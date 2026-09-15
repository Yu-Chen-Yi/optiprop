"""Behavioral tests for immutable multi-layer optical-system execution."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from dataclasses import replace
from uuid import UUID

import pytest
import torch

from optiprop import Field2D, Grid2D
from optiprop.propagation import (
    AngularSpectrumPropagator,
    CancellationToken,
    FresnelPropagator,
    PaddingSpec,
    PropagationCancelled,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
    Propagator,
    RayleighSommerfeldPropagator,
    assess_sampling,
)
from optiprop.system import (
    LayerExecutionStatus,
    OpticalLayer,
    OpticalSystem,
    PropagationBackendRegistry,
    PropagationLayer,
    SystemExecutionCancelled,
    SystemExecutionFailed,
    SystemValidationError,
)


def _field(
    *,
    dtype: torch.dtype = torch.complex128,
    device: str = "cpu",
    requires_grad: bool = False,
) -> Field2D:
    grid = Grid2D(nx=13, ny=9, dx=1.2e-6, dy=0.9e-6)
    x, y = grid.meshgrid(
        dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
        device=device,
    )
    data = (
        torch.exp(-((x / 3.8e-6) ** 2 + (y / 2.9e-6) ** 2))
        * torch.exp(1j * (1.3e5 * x - 0.8e5 * y))
    ).to(dtype)[None]
    if requires_grad:
        data.requires_grad_()
    return Field2D(data=data, grid=grid, wavelength_m=633e-9)


def _layer(
    method: PropagationMethod,
    distance_m: float,
    *,
    name: str,
    enabled: bool = True,
    padding: PaddingSpec | None = None,
) -> PropagationLayer:
    return PropagationLayer(
        name=name,
        enabled=enabled,
        spec=PropagationSpec(
            method=method,
            distance_m=distance_m,
            padding=PaddingSpec.none() if padding is None else padding,
        ),
    )


def _manual(field: Field2D, specs: tuple[PropagationSpec, ...]) -> Field2D:
    current = field
    for spec in specs:
        if spec.method in (PropagationMethod.ASM, PropagationMethod.BLAS):
            result = AngularSpectrumPropagator().propagate(current, spec)
        elif spec.method in (
            PropagationMethod.FRESNEL_TF,
            PropagationMethod.FRESNEL_SCALED,
        ):
            result = FresnelPropagator().propagate(current, spec)
        else:
            result = RayleighSommerfeldPropagator().propagate(current, spec)
        current = result.field
    return current


class _CancellingAsmBackend(Propagator):
    def __init__(self) -> None:
        self.delegate = AngularSpectrumPropagator()

    def validate(self, field: Field2D, spec: PropagationSpec):
        return self.delegate.validate(field, spec)

    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        result = self.delegate.propagate(field, spec, cancel_token)
        assert cancel_token is not None
        cancel_token.cancel("between layers")
        return result


class _FailingBackend(Propagator):
    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        raise RuntimeError("backend exploded")


def test_propagation_layer_has_stable_identity_and_is_runtime_protocol() -> None:
    layer = _layer(PropagationMethod.ASM, 4e-6, name="air gap")

    assert isinstance(layer, OpticalLayer)
    assert isinstance(layer.id, UUID)
    assert layer.name == "air gap"
    assert layer.enabled is True
    assert layer.id == layer.id
    assert layer.to_config()["id"] == str(layer.id)

    with pytest.raises(FrozenInstanceError):
        layer.name = "changed"  # type: ignore[misc]


def test_parameter_hash_excludes_identity_but_tracks_physical_parameters() -> None:
    layer = _layer(PropagationMethod.ASM, 4e-6, name="air gap")

    display_only = replace(layer, name="renamed", enabled=False)
    physical_change = replace(
        layer,
        spec=replace(layer.spec, distance_m=5e-6),
    )

    assert display_only.id == layer.id
    assert display_only.parameter_hash == layer.parameter_hash
    assert physical_change.parameter_hash != layer.parameter_hash


def test_layer_config_does_not_change_when_original_nested_options_mutate() -> None:
    nested = {"labels": ["before"]}
    layer = PropagationLayer(
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=1e-6,
            padding=PaddingSpec.none(),
            options=nested,
        )
    )
    config_before = layer.to_config()
    hash_before = layer.parameter_hash

    nested["labels"].append("after")

    assert layer.to_config() == config_before
    assert layer.parameter_hash == hash_before


def test_to_config_is_strict_json_and_storage_policy_is_not_physical_hash() -> None:
    layer = PropagationLayer(
        name="serializable",
        keep_intermediate=True,
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=1e-6,
            medium_index=1.5 + 0.01j,
            padding=PaddingSpec.explicit((19, 23)),
            options={"nested": {"sequence": (1, 2), "enabled": True}},
        ),
    )
    discard_snapshot = replace(layer, keep_intermediate=False)

    encoded = json.dumps(
        layer.to_config(),
        sort_keys=True,
        allow_nan=False,
    )

    assert json.loads(encoded)["id"] == str(layer.id)
    assert discard_snapshot.parameter_hash == layer.parameter_hash


def test_scaled_fresnel_preview_geometry_matches_sampling_without_execution() -> None:
    field = _field()
    before = field.data.clone()
    layer = _layer(
        PropagationMethod.FRESNEL_SCALED,
        30e-6,
        name="scaled preview",
    )

    geometry = layer.preview_geometry(field)
    expected = assess_sampling(field, layer.spec).output_grid

    assert geometry.grid == expected
    assert geometry.z_m == pytest.approx(field.z_m + layer.spec.distance_m)
    assert not hasattr(geometry, "data")
    torch.testing.assert_close(field.data, before, rtol=0, atol=0)


def test_arbitrary_ordered_layers_match_manual_sequential_execution() -> None:
    field = _field()
    layers = (
        _layer(PropagationMethod.ASM, 3e-6, name="ASM"),
        _layer(PropagationMethod.FRESNEL_TF, 4e-6, name="Fresnel"),
        _layer(
            PropagationMethod.RS_FFT,
            5e-6,
            name="RS",
            padding=PaddingSpec.auto(),
        ),
    )
    system = OpticalSystem(name="mixed methods", layers=layers)

    result = system.execute(field)
    expected = _manual(field, tuple(layer.spec for layer in layers))

    assert result.input_field is field
    assert len(result.layer_results) == 3
    assert tuple(item.layer_id for item in result.layer_results) == tuple(
        layer.id for layer in layers
    )
    torch.testing.assert_close(result.output_field.data, expected.data)
    assert result.output_field.z_m == pytest.approx(12e-6)


@pytest.mark.parametrize("method", tuple(PropagationMethod))
def test_default_registry_dispatches_every_public_method(
    method: PropagationMethod,
) -> None:
    """Zero distance isolates registry dispatch without conflating algorithms."""

    field = _field()
    layer = _layer(method, 0.0, name=method.value)

    result = OpticalSystem(layers=(layer,)).execute(field)

    assert result.layer_results[0].status is LayerExecutionStatus.COMPLETED
    assert result.layer_results[0].diagnostics["method"] == method.value
    torch.testing.assert_close(result.output_field.data, field.data, rtol=0, atol=0)


def test_results_keep_input_and_per_layer_output_snapshots() -> None:
    field = _field()
    layers = (
        _layer(PropagationMethod.ASM, 2e-6, name="first"),
        _layer(PropagationMethod.FRESNEL_TF, 3e-6, name="second"),
    )

    result = OpticalSystem(layers=layers).run(field)

    first, second = result.layer_results
    assert first.input_field is field
    assert second.input_field is first.output_field
    assert result.output_field is second.output_field
    assert first.status is LayerExecutionStatus.COMPLETED
    assert second.status is LayerExecutionStatus.COMPLETED
    assert first.elapsed_s >= 0.0
    assert first.parameter_hash
    assert first.diagnostics["method"] == PropagationMethod.ASM.value


def test_disabled_layer_is_reported_as_skipped_and_is_exact_identity() -> None:
    field = _field()
    disabled = _layer(
        PropagationMethod.RS_DIRECT,
        1.0,
        name="disabled expensive layer",
        enabled=False,
    )

    result = OpticalSystem(layers=(disabled,)).execute(field)
    snapshot = result.layer_results[0]

    assert snapshot.status is LayerExecutionStatus.SKIPPED
    assert snapshot.input_field is field
    assert snapshot.output_field is field
    assert result.output_field is field
    torch.testing.assert_close(result.output_field.data, field.data, rtol=0, atol=0)


def test_execution_does_not_mutate_input_and_preserves_autograd() -> None:
    field = _field(requires_grad=True)
    before = field.data.detach().clone()
    system = OpticalSystem(
        layers=(_layer(PropagationMethod.ASM, 7e-6, name="gap"),)
    )

    result = system.execute(field)
    loss = result.output_field.intensity().sum()
    loss.backward()

    torch.testing.assert_close(field.data.detach(), before, rtol=0, atol=0)
    assert field.data.grad is not None
    assert torch.isfinite(field.data.grad).all()


def test_validation_readdresses_backend_issues_to_layer_and_spec_parameter() -> None:
    field = _field()
    # ASM rejects an output grid even though the generic PropagationSpec accepts it.
    bad = PropagationLayer(
        name="bad output",
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=2e-6,
            padding=PaddingSpec.none(),
            output_grid=Grid2D(nx=7, ny=5, dx=2e-6, dy=2e-6),
        ),
    )

    report = OpticalSystem(layers=(bad,)).validate(field)

    assert not report.is_valid
    issue = report.errors[0]
    assert issue.layer_id == str(bad.id)
    assert issue.parameter_path == "spec.output_grid"


def test_wrong_backend_method_is_layer_addressed() -> None:
    field = _field()
    layer = _layer(PropagationMethod.ASM, 2e-6, name="wrong dispatch")

    report = layer.validate(
        field,
        backend=FresnelPropagator(),
    )

    assert not report.is_valid
    assert all(issue.layer_id == str(layer.id) for issue in report.errors)
    assert report.errors[0].parameter_path == "spec.method"


def test_invalid_negative_rs_distance_is_addressed_to_distance_parameter() -> None:
    field = _field()
    layer = _layer(PropagationMethod.RS_FFT, -2e-6, name="backward RS")

    report = OpticalSystem(layers=(layer,)).validate(field)

    assert not report.is_valid
    issue = next(
        item
        for item in report.errors
        if item.code == "rs.negative_distance_unsupported"
    )
    assert issue.layer_id == str(layer.id)
    assert issue.parameter_path == "spec.distance_m"


def test_system_validation_error_carries_input_only_partial_result() -> None:
    field = _field()
    bad = PropagationLayer(
        name="invalid ASM output",
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=2e-6,
            padding=PaddingSpec.none(),
            output_grid=Grid2D(nx=7, ny=5, dx=2e-6, dy=2e-6),
        ),
    )

    with pytest.raises(SystemValidationError) as caught:
        OpticalSystem(layers=(bad,)).execute(field)

    partial = caught.value.partial_result
    assert partial.input_field is field
    assert partial.output_field is field
    assert partial.layer_results == ()
    assert not partial.validation.is_valid
    assert partial.validation.errors[0].layer_id == str(bad.id)


def test_cancelled_before_run_raises_with_empty_partial_result() -> None:
    field = _field()
    token = CancellationToken()
    token.cancel("test cancellation")
    system = OpticalSystem(
        layers=(_layer(PropagationMethod.ASM, 2e-6, name="never runs"),)
    )

    with pytest.raises(SystemExecutionCancelled) as caught:
        system.execute(field, cancel_token=token)

    assert isinstance(caught.value, PropagationCancelled)
    assert caught.value.partial_result.input_field is field
    assert caught.value.partial_result.output_field is field
    assert caught.value.partial_result.layer_results == ()


def test_cancellation_between_layers_keeps_completed_immutable_snapshot() -> None:
    field = _field()
    token = CancellationToken()
    backend = _CancellingAsmBackend()
    registry = PropagationBackendRegistry(
        {PropagationMethod.ASM: backend}
    )
    first = _layer(PropagationMethod.ASM, 2e-6, name="completed")
    second = _layer(PropagationMethod.ASM, 3e-6, name="not started")

    with pytest.raises(SystemExecutionCancelled) as caught:
        OpticalSystem(layers=(first, second)).execute(
            field,
            cancel_token=token,
            registry=registry,
        )

    partial = caught.value.partial_result
    assert partial.completed_layer_ids == (first.id,)
    assert len(partial.layer_results) == 1
    assert partial.layer_results[0].layer_id == first.id
    assert partial.output_field is partial.layer_results[0].output_field
    assert partial.output_field.z_m == pytest.approx(2e-6)


def test_backend_failure_carries_layer_id_cause_and_completed_provenance() -> None:
    field = _field()
    failing = _FailingBackend()
    registry = PropagationBackendRegistry(
        {PropagationMethod.ASM: failing}
    )
    layer = _layer(PropagationMethod.ASM, 2e-6, name="fails")

    with pytest.raises(SystemExecutionFailed) as caught:
        OpticalSystem(layers=(layer,)).execute(field, registry=registry)

    assert caught.value.layer_id == layer.id
    assert isinstance(caught.value.cause, RuntimeError)
    assert "backend exploded" in str(caught.value.cause)
    assert caught.value.partial_result.input_field is field
    assert caught.value.partial_result.output_field is field
    assert caught.value.partial_result.layer_results == ()


def test_immutable_add_remove_replace_and_move_semantics() -> None:
    first = _layer(PropagationMethod.ASM, 1e-6, name="first")
    second = _layer(PropagationMethod.FRESNEL_TF, 2e-6, name="second")
    third = _layer(PropagationMethod.RS_FFT, 3e-6, name="third")
    original = OpticalSystem(name="editor", layers=(first, second))

    appended = original.append(third)
    inserted = original.insert(1, third)
    moved = appended.move(third.id, 0)
    replacement = _layer(PropagationMethod.BLAS, 9e-6, name="replacement")
    replaced = original.replace(first.id, replacement)
    removed = original.remove(second.id)

    assert original.layers == (first, second)
    assert appended.layers == (first, second, third)
    assert inserted.layers == (first, third, second)
    assert moved.layers == (third, first, second)
    assert replaced.layers[0].id == first.id
    assert replaced.layers[0].spec == replacement.spec
    assert removed.layers == (first,)
    assert all(system.id == original.id for system in (
        appended,
        inserted,
        moved,
        replaced,
        removed,
    ))


def test_duplicate_ids_and_unknown_edit_targets_are_rejected() -> None:
    layer = _layer(PropagationMethod.ASM, 1e-6, name="one")

    with pytest.raises(ValueError, match="duplicate|unique"):
        OpticalSystem(layers=(layer, layer))

    system = OpticalSystem(layers=(layer,))
    unknown = _layer(PropagationMethod.ASM, 1e-6, name="unknown")
    with pytest.raises(KeyError):
        system.remove(unknown.id)
    with pytest.raises(KeyError):
        system.move(unknown.id, 0)


def test_set_enabled_and_update_are_immutable_and_preserve_layer_id() -> None:
    layer = _layer(PropagationMethod.ASM, 1e-6, name="gap")
    system = OpticalSystem(layers=(layer,))

    disabled = system.set_enabled(layer.id, False)
    renamed = system.update(layer.id, name="renamed")

    assert system.layers[0].enabled is True
    assert system.layers[0].name == "gap"
    assert disabled.layers[0].enabled is False
    assert renamed.layers[0].name == "renamed"
    assert disabled.layers[0].id == layer.id
    assert renamed.layers[0].id == layer.id


def test_validation_tracks_scaled_fresnel_output_geometry_downstream() -> None:
    field = _field()
    distance_m = 1.0e-3
    natural_grid = Grid2D(
        nx=field.grid.nx,
        ny=field.grid.ny,
        dx=(
            field.wavelength_m
            * distance_m
            / (field.grid.nx * field.grid.dx)
        ),
        dy=(
            field.wavelength_m
            * distance_m
            / (field.grid.ny * field.grid.dy)
        ),
    )
    scaled = _layer(
        PropagationMethod.FRESNEL_SCALED,
        distance_m,
        name="scaled",
    )
    downstream = PropagationLayer(
        name="downstream",
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=2.0e-6,
            padding=PaddingSpec.none(),
            output_grid=natural_grid,
        ),
    )
    system = OpticalSystem(layers=(scaled, downstream))

    report = system.validate(field)
    result = system.execute(field)

    assert report.is_valid
    assert result.output_field.grid == natural_grid


def test_execution_range_validates_only_the_selected_layer_range() -> None:
    field = _field()
    incompatible = Grid2D(nx=7, ny=5, dx=2e-6, dy=2e-6)
    invalid = PropagationLayer(
        name="outside range",
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=2.0e-6,
            padding=PaddingSpec.none(),
            output_grid=incompatible,
        ),
    )
    valid = _layer(
        PropagationMethod.FRESNEL_TF,
        3.0e-6,
        name="selected",
    )
    system = OpticalSystem(layers=(invalid, valid))

    result = system.execute(field, start_at=valid.id)

    assert tuple(item.layer_id for item in result.layer_results) == (valid.id,)
    assert result.output_field.z_m == pytest.approx(3.0e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_system_execution_preserves_cuda_device() -> None:
    field = _field(dtype=torch.complex64, device="cuda")
    system = OpticalSystem(
        layers=(
            _layer(PropagationMethod.ASM, 2e-6, name="ASM"),
            _layer(PropagationMethod.FRESNEL_TF, 3e-6, name="Fresnel"),
        )
    )

    result = system.execute(field)

    assert result.output_field.device.type == "cuda"
    assert all(
        item.output_field.device.type == "cuda" for item in result.layer_results
    )
