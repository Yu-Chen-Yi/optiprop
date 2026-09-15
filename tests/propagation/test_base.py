"""Contract tests for the common propagation request/result API."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from optiprop.core import Field2D, Grid2D
from optiprop.core.validation import Severity, ValidationIssue, ValidationReport
from optiprop.propagation import (
    CancellationToken,
    EvanescentPolicy,
    PaddingMode,
    PaddingSpec,
    PrecisionPolicy,
    PropagationCancelled,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
)
from optiprop.propagation.sampling import assess_sampling


def _field() -> Field2D:
    grid = Grid2D(nx=6, ny=5, dx=2.0e-6, dy=3.0e-6)
    return Field2D(
        data=torch.ones((1, grid.ny, grid.nx), dtype=torch.complex64),
        grid=grid,
        wavelength_m=940e-9,
    )


def test_propagation_methods_have_stable_serializable_values() -> None:
    assert {method.value for method in PropagationMethod} == {
        "asm",
        "blas",
        "fresnel_tf",
        "fresnel_scaled",
        "rs_fft",
        "rs_direct",
    }
    assert {mode.value for mode in PaddingMode} == {
        "none",
        "auto",
        "factor",
        "explicit",
    }
    assert {policy.value for policy in EvanescentPolicy} == {
        "discard",
        "decay",
        "keep",
    }
    assert {policy.value for policy in PrecisionPolicy} == {
        "inherit",
        "complex64",
        "complex128",
    }


@pytest.mark.parametrize("method", list(PropagationMethod))
def test_spec_accepts_every_propagation_method(method: PropagationMethod) -> None:
    spec = PropagationSpec(method=method, distance_m=-1.25e-3)

    assert spec.method is method
    assert spec.distance_m == pytest.approx(-1.25e-3)
    assert spec.padding == PaddingSpec.auto()
    assert spec.medium_index is None
    assert spec.output_grid is None


def test_spec_accepts_explicit_padding_medium_and_output_grid() -> None:
    output_grid = Grid2D(
        nx=8,
        ny=7,
        dx=1.5e-6,
        dy=2.5e-6,
        x_center=1.0e-6,
        y_center=-2.0e-6,
    )
    spec = PropagationSpec(
        method=PropagationMethod.RS_DIRECT,
        distance_m=2.0e-3,
        medium_index=1.45 + 0.001j,
        padding=PaddingSpec.by_factor(2.0),
        output_grid=output_grid,
    )

    assert spec.medium_index == 1.45 + 0.001j
    assert spec.padding.mode is PaddingMode.FACTOR
    assert spec.padding.factor == pytest.approx(2.0)
    assert spec.output_grid == output_grid
    with pytest.raises(FrozenInstanceError):
        spec.distance_m = 1.0


@pytest.mark.parametrize(
    "distance_m",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_spec_rejects_non_finite_distance(distance_m: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        PropagationSpec(method=PropagationMethod.ASM, distance_m=distance_m)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("method", "invalid", id="unknown-method"),
        pytest.param("padding", object(), id="invalid-padding"),
        pytest.param("medium_index", 0.0, id="zero-medium-real"),
        pytest.param("medium_index", -1.0 + 0.1j, id="negative-medium-real"),
        pytest.param(
            "medium_index", complex(1.0, math.nan), id="non-finite-medium"
        ),
        pytest.param("output_grid", object(), id="invalid-output-grid"),
    ],
)
def test_spec_rejects_invalid_options(keyword: str, value) -> None:
    parameters = {
        "method": PropagationMethod.ASM,
        "distance_m": 1.0e-3,
    }
    parameters[keyword] = value

    with pytest.raises((TypeError, ValueError)):
        PropagationSpec(**parameters)


def test_string_enums_are_canonicalized_for_json_facing_callers() -> None:
    spec = PropagationSpec(
        method="asm",
        distance_m=1.0e-3,
        padding=PaddingSpec(mode="none"),
        evanescent_policy="decay",
        precision="complex128",
    )

    assert spec.method is PropagationMethod.ASM
    assert spec.padding.mode is PaddingMode.NONE
    assert spec.evanescent_policy is EvanescentPolicy.DECAY
    assert spec.precision is PrecisionPolicy.COMPLEX128


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(PaddingSpec.none(), id="none"),
        pytest.param(PaddingSpec.auto(), id="auto"),
        pytest.param(PaddingSpec.by_factor(2.0), id="factor"),
        pytest.param(PaddingSpec.explicit((254, 384)), id="explicit"),
    ],
)
def test_padding_spec_constructors_are_canonical(padding: PaddingSpec) -> None:
    assert isinstance(padding.mode, PaddingMode)
    if padding.mode is PaddingMode.FACTOR:
        assert padding.factor == pytest.approx(2.0)
        assert padding.shape is None
    elif padding.mode is PaddingMode.EXPLICIT:
        assert padding.shape == (254, 384)
        assert padding.factor is None
    else:
        assert padding.factor is None
        assert padding.shape is None


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: PaddingSpec.by_factor(0.99), id="factor-below-one"),
        pytest.param(lambda: PaddingSpec.by_factor(math.nan), id="factor-nan"),
        pytest.param(lambda: PaddingSpec.by_factor(math.inf), id="factor-infinite"),
        pytest.param(lambda: PaddingSpec.explicit((0, 4)), id="zero-explicit-ny"),
        pytest.param(lambda: PaddingSpec.explicit((3, -1)), id="negative-explicit-nx"),
        pytest.param(lambda: PaddingSpec.explicit((3.5, 4)), id="fractional-shape"),
        pytest.param(
            lambda: PaddingSpec(
                mode=PaddingMode.NONE, factor=2.0
            ),
            id="none-with-factor",
        ),
    ],
)
def test_invalid_padding_spec_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_cancellation_token_is_idempotent_and_raises_with_reason() -> None:
    token = CancellationToken()

    assert not token.is_cancelled
    token.throw_if_cancelled()
    assert token.cancel("user requested stop") is None
    assert token.cancel("ignored second reason") is None
    assert token.is_cancelled
    assert token.reason == "user requested stop"

    with pytest.raises(PropagationCancelled, match="user requested stop"):
        token.throw_if_cancelled()


def test_propagation_result_is_immutable_and_snapshots_metadata() -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=1.0e-3,
        padding=PaddingSpec.none(),
    )
    issue = ValidationIssue(
        code="sampling.checked",
        severity=Severity.INFO,
        message="Sampling checked",
    )
    report = ValidationReport((issue,))
    metadata = {"backend": "torch"}

    result = PropagationResult(
        field=field,
        spec=spec,
        validation=report,
        elapsed_s=0.25,
        diagnostics=metadata,
    )
    metadata["backend"] = "changed"

    assert result.field is field
    assert result.spec is spec
    assert result.validation is report
    assert result.elapsed_s == pytest.approx(0.25)
    assert result.diagnostics["backend"] == "torch"
    with pytest.raises(FrozenInstanceError):
        result.elapsed_s = 2.0
    with pytest.raises(TypeError):
        result.diagnostics["backend"] = "changed"


@pytest.mark.parametrize("elapsed_s", [-1.0, math.nan, math.inf])
def test_propagation_result_rejects_invalid_elapsed_time(elapsed_s: float) -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=1.0e-3,
    )

    with pytest.raises((TypeError, ValueError)):
        PropagationResult(
            field=field,
            spec=spec,
            validation=ValidationReport(),
            elapsed_s=elapsed_s,
        )


def test_propagation_result_keeps_structured_sampling_and_power_metrics() -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=1.0e-3,
        padding=PaddingSpec.none(),
    )
    sampling = assess_sampling(field, spec)

    result = PropagationResult(
        field=field,
        spec=spec,
        sampling=sampling,
        validation=sampling.validation,
        power_before=2.0,
        power_after=1.5,
        cache_key="sha256:example",
    )

    assert result.sampling is sampling
    assert result.validation is sampling.validation
    assert result.power_before == pytest.approx(2.0)
    assert result.power_after == pytest.approx(1.5)
    assert result.cache_key == "sha256:example"
    assert result.warnings == sampling.warnings


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("power_before", -1.0, id="negative-input-power"),
        pytest.param("power_after", math.inf, id="infinite-output-power"),
        pytest.param("cache_key", "", id="empty-cache-key"),
        pytest.param("validation", object(), id="invalid-validation-report"),
        pytest.param("sampling", object(), id="invalid-sampling-report"),
    ],
)
def test_propagation_result_rejects_invalid_structured_context(
    keyword: str, value
) -> None:
    field = _field()
    parameters = {
        "field": field,
        "spec": PropagationSpec(PropagationMethod.ASM, 1.0e-3),
        keyword: value,
    }

    with pytest.raises((TypeError, ValueError)):
        PropagationResult(**parameters)


def test_legacy_propagators_remain_available_from_all_public_import_paths() -> None:
    from optiprop import (
        ASMPropagation as root_asm,
        FresnelPropagation as root_fresnel,
        RayleighSommerfeldPropagation as root_rs,
    )
    from optiprop.propagation import (
        ASMPropagation as package_asm,
        FresnelPropagation as package_fresnel,
        RayleighSommerfeldPropagation as package_rs,
    )
    from optiprop.propagation.legacy import (
        ASMPropagation as legacy_asm,
        FresnelPropagation as legacy_fresnel,
        RayleighSommerfeldPropagation as legacy_rs,
    )

    assert root_asm is package_asm is legacy_asm
    assert root_fresnel is package_fresnel is legacy_fresnel
    assert root_rs is package_rs is legacy_rs
