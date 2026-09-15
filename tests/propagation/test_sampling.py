"""Contract tests for pre-run sampling and workload assessment."""

import pytest
import torch

from optiprop.core import Field2D, Grid2D
from optiprop.propagation import (
    EvanescentPolicy,
    PaddingSpec,
    PropagationMethod,
    PropagationSpec,
)
from optiprop.propagation.sampling import assess_sampling


def _field(
    *,
    nx: int = 192,
    ny: int = 127,
    dx: float = 0.37e-6,
    dy: float = 0.81e-6,
    components: tuple[str, ...] = ("scalar",),
    dtype: torch.dtype = torch.complex64,
) -> Field2D:
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    return Field2D(
        data=torch.ones((len(components), ny, nx), dtype=dtype),
        grid=grid,
        wavelength_m=1.31e-6,
        medium_index=1.5,
        components=components,
    )


def _codes(assessment) -> set[str]:
    return {issue.code for issue in assessment.report.issues}


def test_rectangular_dx_dy_are_reported_in_the_correct_axes() -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=250e-6,
        padding=PaddingSpec.none(),
    )

    assessment = assess_sampling(field, spec)

    assert assessment.input_shape == (127, 192)
    assert assessment.padded_shape == (127, 192)
    extent_x, extent_y = assessment.physical_extent_m
    nyquist_x, nyquist_y = assessment.nyquist_frequency_per_m
    assert extent_x == pytest.approx(192 * 0.37e-6)
    assert extent_y == pytest.approx(127 * 0.81e-6)
    assert nyquist_x == pytest.approx(1.0 / (2 * 0.37e-6))
    assert nyquist_y == pytest.approx(1.0 / (2 * 0.81e-6))
    assert assessment.medium_wavelength_m == pytest.approx(1.31e-6 / 1.5)
    assert assessment.estimated_memory_bytes > 0


def test_factor_padding_preserves_rectangular_order_and_increases_memory() -> None:
    field = _field()
    without_padding = assess_sampling(
        field,
        PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=250e-6,
            padding=PaddingSpec.none(),
        ),
    )
    with_padding = assess_sampling(
        field,
        PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=250e-6,
            padding=PaddingSpec.by_factor(2.0),
        ),
    )

    assert with_padding.padded_shape == (254, 384)
    assert with_padding.estimated_memory_bytes > without_padding.estimated_memory_bytes
    assert with_padding.input_shape == without_padding.input_shape == (127, 192)


def test_padding_records_half_pixel_center_shift_when_parity_changes() -> None:
    field = _field(nx=5, ny=7, dx=1.0e-6, dy=2.0e-6)

    assessment = assess_sampling(
        field,
        PropagationSpec(
            PropagationMethod.ASM,
            1.0e-3,
            padding=PaddingSpec.auto(),
        ),
    )

    assert assessment.padded_shape == (14, 10)
    assert assessment.computational_grid.x_center == pytest.approx(0.5e-6)
    assert assessment.computational_grid.y_center == pytest.approx(1.0e-6)


def test_complex128_memory_estimate_exceeds_complex64_estimate() -> None:
    spec = PropagationSpec(
        method=PropagationMethod.BLAS,
        distance_m=500e-6,
        padding=PaddingSpec.by_factor(2.0),
    )

    estimate64 = assess_sampling(_field(dtype=torch.complex64), spec)
    estimate128 = assess_sampling(_field(dtype=torch.complex128), spec)

    assert estimate128.estimated_memory_bytes > estimate64.estimated_memory_bytes


def test_auto_padding_reports_recommendation_and_structured_issue() -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.BLAS,
        distance_m=20e-3,
        padding=PaddingSpec.auto(),
    )

    assessment = assess_sampling(field, spec)

    recommended_ny, recommended_nx = assessment.recommended_padded_shape
    assert recommended_ny >= field.grid.ny
    assert recommended_nx >= field.grid.nx
    assert (recommended_ny, recommended_nx) != (field.grid.ny, field.grid.nx)
    assert assessment.padded_shape == assessment.recommended_padded_shape
    assert "sampling.padding_recommended" in _codes(assessment)


def test_zero_distance_is_valid_bypass_with_no_padding() -> None:
    field = _field()
    spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=0.0,
        padding=PaddingSpec.auto(),
    )

    assessment = assess_sampling(field, spec)

    assert assessment.report.is_valid
    assert assessment.report.errors == ()
    assert assessment.padded_shape == (field.grid.ny, field.grid.nx)
    assert assessment.recommended_padded_shape == assessment.padded_shape
    assert "sampling.zero_distance" in _codes(assessment)


def test_negative_distance_is_supported_and_uses_absolute_distance_estimates() -> None:
    field = _field()
    forward = assess_sampling(
        field,
        PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=1.0e-3,
            padding=PaddingSpec.none(),
        ),
    )
    backward = assess_sampling(
        field,
        PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=-1.0e-3,
            padding=PaddingSpec.none(),
        ),
    )

    assert backward.report.is_valid
    assert backward.report.errors == ()
    assert backward.fresnel_number == pytest.approx(forward.fresnel_number)
    assert backward.recommended_padded_shape == forward.recommended_padded_shape
    assert "sampling.negative_distance" in _codes(backward)


def test_rs_direct_reports_exact_component_weighted_interaction_count() -> None:
    field = _field(
        nx=5,
        ny=7,
        dx=1.0e-6,
        dy=2.0e-6,
        components=("Ex", "Ey"),
    )
    output_grid = Grid2D(nx=4, ny=3, dx=0.75e-6, dy=1.25e-6)
    spec = PropagationSpec(
        method=PropagationMethod.RS_DIRECT,
        distance_m=25e-6,
        padding=PaddingSpec.none(),
        output_grid=output_grid,
    )

    assessment = assess_sampling(field, spec)

    assert assessment.estimated_operations == 2 * (7 * 5) * (3 * 4)
    assert "sampling.rs_direct_workload" in _codes(assessment)
    assert assessment.report.is_valid


def test_non_direct_method_does_not_claim_rs_direct_operations() -> None:
    assessment = assess_sampling(
        _field(nx=5, ny=7),
        PropagationSpec(
            method=PropagationMethod.RS_FFT,
            distance_m=25e-6,
            padding=PaddingSpec.none(),
        ),
    )

    assert assessment.estimated_operations is None


def test_sampling_issues_have_stable_structured_codes_and_messages() -> None:
    field = _field()
    assessment = assess_sampling(
        field,
        PropagationSpec(
            method=PropagationMethod.BLAS,
            distance_m=20e-3,
            padding=PaddingSpec.auto(),
        ),
    )

    assert assessment.report.issues
    for issue in assessment.report.issues:
        assert issue.code.startswith("sampling.")
        assert " " not in issue.code
        assert issue.message.strip()
        assert issue.details


def test_memory_limit_is_a_structured_blocking_error() -> None:
    field = _field()
    assessment = assess_sampling(
        field,
        PropagationSpec(PropagationMethod.ASM, 1.0e-3),
        memory_limit_bytes=1024,
    )

    assert not assessment.is_valid
    assert "sampling.memory_estimate_exceeded" in _codes(assessment)


def test_backward_decay_is_a_stable_regularized_request() -> None:
    assessment = assess_sampling(
        _field(),
        PropagationSpec(
            PropagationMethod.ASM,
            -1.0e-3,
            evanescent_policy=EvanescentPolicy.DECAY,
        ),
    )

    assert assessment.is_valid
    assert "sampling.negative_distance" in _codes(assessment)


def test_backward_keep_reports_conditioning_and_blocks_overflow() -> None:
    warning = assess_sampling(
        _field(dtype=torch.complex128),
        PropagationSpec(
            PropagationMethod.ASM,
            -1.0e-6,
            evanescent_policy=EvanescentPolicy.KEEP,
        ),
    )
    blocked = assess_sampling(
        _field(dtype=torch.complex64),
        PropagationSpec(
            PropagationMethod.ASM,
            -1.0e-3,
            evanescent_policy=EvanescentPolicy.KEEP,
        ),
    )

    assert warning.is_valid
    assert "sampling.evanescent_backward_ill_conditioned" in _codes(warning)
    assert not blocked.is_valid
    assert "sampling.evanescent_backward_overflow" in _codes(blocked)


def test_lossy_backward_and_active_media_are_blocked() -> None:
    passive_loss = _field().with_data(
        _field().data,
        medium_index=1.5 + 0.01j,
    )
    active_gain = _field().with_data(
        _field().data,
        medium_index=1.5 - 0.01j,
    )

    backward_loss = assess_sampling(
        passive_loss,
        PropagationSpec(
            PropagationMethod.ASM,
            -1.0e-6,
            evanescent_policy=EvanescentPolicy.DISCARD,
        ),
    )
    forward_gain = assess_sampling(
        active_gain,
        PropagationSpec(
            PropagationMethod.ASM,
            1.0e-6,
            evanescent_policy=EvanescentPolicy.DISCARD,
        ),
    )

    assert not backward_loss.is_valid
    assert "sampling.lossy_medium_backward_unstable" in _codes(backward_loss)
    assert not forward_gain.is_valid
    assert "sampling.active_medium_unsupported" in _codes(forward_gain)


def test_same_grid_method_rejects_custom_output_sampling() -> None:
    assessment = assess_sampling(
        _field(nx=8, ny=5),
        PropagationSpec(
            PropagationMethod.BLAS,
            1.0e-3,
            output_grid=Grid2D(nx=4, ny=3, dx=2.0e-6, dy=3.0e-6),
        ),
    )

    assert not assessment.is_valid
    assert "sampling.output_grid_unsupported" in _codes(assessment)


def test_explicit_padding_cannot_be_smaller_than_the_input() -> None:
    assessment = assess_sampling(
        _field(nx=8, ny=5),
        PropagationSpec(
            PropagationMethod.ASM,
            1.0e-3,
            padding=PaddingSpec.explicit((4, 7)),
        ),
    )

    assert not assessment.is_valid
    assert "sampling.padding_explicit_too_small" in _codes(assessment)
