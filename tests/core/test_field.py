"""Contract tests for the canonical scalar and polarized field container."""

import math

import pytest
import torch

from optiprop.core import Field2D, Grid2D


def _grid(
    *,
    nx: int = 192,
    ny: int = 127,
    dx: float = 0.37e-6,
    dy: float = 0.81e-6,
) -> Grid2D:
    return Grid2D(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x_center=4.25e-6,
        y_center=-2.75e-6,
    )


def _scalar_field(
    *,
    dtype: torch.dtype = torch.complex64,
    metadata: dict | None = None,
) -> Field2D:
    grid = _grid()
    real = torch.arange(grid.ny * grid.nx, dtype=torch.float32).reshape(
        1, grid.ny, grid.nx
    )
    data = torch.complex(real, -0.25 * real).to(dtype)
    return Field2D(
        data=data,
        grid=grid,
        wavelength_m=1.31e-6,
        medium_index=1.5 + 0.01j,
        components=("scalar",),
        z_m=6.5e-6,
        metadata={} if metadata is None else metadata,
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_scalar_field_preserves_canonical_shape_and_complex_dtype(
    dtype: torch.dtype,
) -> None:
    grid = _grid()
    data = torch.ones((1, grid.ny, grid.nx), dtype=dtype)
    before = data.clone()

    field = Field2D(data=data, grid=grid, wavelength_m=532e-9)

    assert field.data.shape == (1, 127, 192)
    assert field.data.dtype == dtype
    assert field.components == ("scalar",)
    assert field.grid == grid
    torch.testing.assert_close(data, before)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_ex_ey_field_preserves_component_order_shape_and_dtype(
    dtype: torch.dtype,
) -> None:
    grid = _grid()
    data = torch.zeros((2, grid.ny, grid.nx), dtype=dtype)
    data[0] = 1.0 + 2.0j
    data[1] = -3.0 + 4.0j
    before = data.clone()

    field = Field2D(
        data=data,
        grid=grid,
        wavelength_m=1.31e-6,
        components=("Ex", "Ey"),
    )

    assert field.data.shape == (2, 127, 192)
    assert field.data.dtype == dtype
    assert field.components == ("Ex", "Ey")
    torch.testing.assert_close(field.data, before)
    torch.testing.assert_close(data, before)


def test_amplitude_phase_and_component_intensity_are_derived_without_mutation() -> None:
    grid = _grid(nx=4, ny=3)
    data = torch.empty((2, grid.ny, grid.nx), dtype=torch.complex128)
    data[0] = 3.0 + 4.0j
    data[1] = 1.0 - 1.0j
    before = data.clone()
    field = Field2D(
        data=data,
        grid=grid,
        wavelength_m=633e-9,
        components=("Ex", "Ey"),
    )

    amplitude = field.amplitude
    phase = field.phase()
    ex_intensity = field.intensity(component="Ex")
    ey_intensity = field.intensity(component="Ey")
    total_intensity = field.intensity()

    assert amplitude.shape == data.shape
    assert phase.shape == data.shape
    torch.testing.assert_close(amplitude, torch.abs(data))
    torch.testing.assert_close(phase, torch.angle(data))
    torch.testing.assert_close(
        ex_intensity,
        torch.full((3, 4), 25.0, dtype=ex_intensity.dtype),
    )
    torch.testing.assert_close(
        ey_intensity,
        torch.full((3, 4), 2.0, dtype=ey_intensity.dtype),
    )
    torch.testing.assert_close(total_intensity, ex_intensity + ey_intensity)
    torch.testing.assert_close(field.data, before)
    torch.testing.assert_close(data, before)


def test_power_is_total_intensity_integrated_with_dx_dy() -> None:
    grid = _grid(nx=6, ny=5, dx=2.0e-6, dy=3.0e-6)
    data = torch.empty((2, grid.ny, grid.nx), dtype=torch.complex128)
    data[0] = 1.0 + 0.0j
    data[1] = 0.0 + 2.0j
    before = data.clone()
    field = Field2D(
        data=data,
        grid=grid,
        wavelength_m=940e-9,
        components=("Ex", "Ey"),
    )
    expected = 5.0 * grid.nx * grid.ny * grid.dx * grid.dy

    assert float(field.power()) == pytest.approx(expected, rel=1e-12)
    torch.testing.assert_close(data, before)


def test_to_returns_a_new_field_and_does_not_mutate_source() -> None:
    metadata = {"source": "synthetic", "run": 3}
    field = _scalar_field(dtype=torch.complex64, metadata=metadata)
    before = field.data.clone()

    converted = field.to(device="cpu", dtype=torch.complex128)

    assert isinstance(converted, Field2D)
    assert converted is not field
    assert converted.data.dtype == torch.complex128
    assert converted.data.device.type == "cpu"
    assert converted.grid == field.grid
    assert converted.wavelength_m == field.wavelength_m
    assert converted.medium_index == field.medium_index
    assert converted.components == field.components
    assert converted.z_m == field.z_m
    assert converted.metadata == field.metadata
    torch.testing.assert_close(converted.data, before.to(torch.complex128))
    assert field.data.dtype == torch.complex64
    torch.testing.assert_close(field.data, before)
    assert metadata == {"source": "synthetic", "run": 3}


def test_with_data_preserves_metadata_and_does_not_mutate_either_input() -> None:
    field = _scalar_field(metadata={"plane": "input"})
    source_before = field.data.clone()
    replacement = torch.full_like(field.data, 2.0 - 3.0j)
    replacement_before = replacement.clone()

    changed = field.with_data(replacement)

    assert isinstance(changed, Field2D)
    assert changed is not field
    assert changed.grid == field.grid
    assert changed.wavelength_m == field.wavelength_m
    assert changed.medium_index == field.medium_index
    assert changed.components == field.components
    assert changed.z_m == field.z_m
    assert changed.metadata == field.metadata
    torch.testing.assert_close(changed.data, replacement_before)
    torch.testing.assert_close(field.data, source_before)
    torch.testing.assert_close(replacement, replacement_before)


def test_to_rejects_real_dtype_without_mutating_source() -> None:
    field = _scalar_field()
    before = field.data.clone()

    with pytest.raises((TypeError, ValueError)):
        field.to(dtype=torch.float32)

    torch.testing.assert_close(field.data, before)


def test_unknown_intensity_component_is_rejected() -> None:
    grid = _grid(nx=4, ny=3)
    field = Field2D(
        data=torch.ones((2, 3, 4), dtype=torch.complex64),
        grid=grid,
        wavelength_m=633e-9,
        components=("Ex", "Ey"),
    )

    with pytest.raises((KeyError, ValueError)):
        field.intensity(component="Ez")


@pytest.mark.parametrize(
    ("data", "components"),
    [
        pytest.param(
            torch.ones((1, 127, 192), dtype=torch.float32),
            ("scalar",),
            id="real-dtype",
        ),
        pytest.param(
            torch.ones((1, 127, 192), dtype=torch.int64),
            ("scalar",),
            id="integer-dtype",
        ),
        pytest.param(
            torch.ones((127, 192), dtype=torch.complex64),
            ("scalar",),
            id="missing-component-axis",
        ),
        pytest.param(
            torch.ones((1, 192, 127), dtype=torch.complex64),
            ("scalar",),
            id="silent-transpose-forbidden",
        ),
        pytest.param(
            torch.ones((2, 127, 192), dtype=torch.complex64),
            ("scalar",),
            id="component-count-mismatch",
        ),
        pytest.param(
            torch.ones((1, 127, 192), dtype=torch.complex64),
            ("Ex",),
            id="unsupported-single-component-name",
        ),
        pytest.param(
            torch.ones((2, 127, 192), dtype=torch.complex64),
            ("Ey", "Ex"),
            id="polarization-order-must-be-canonical",
        ),
        pytest.param(
            torch.ones((3, 127, 192), dtype=torch.complex64),
            ("Ex", "Ey", "Ez"),
            id="unsupported-component-count",
        ),
    ],
)
def test_invalid_data_shape_dtype_or_components_are_rejected(
    data: torch.Tensor, components: tuple[str, ...]
) -> None:
    before = data.clone()

    with pytest.raises((TypeError, ValueError)):
        Field2D(
            data=data,
            grid=_grid(),
            wavelength_m=1.31e-6,
            components=components,
        )

    torch.testing.assert_close(data, before)


def test_non_tensor_data_is_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        Field2D(
            data=[[[1.0 + 0.0j]]],
            grid=Grid2D(nx=1, ny=1, dx=1.0, dy=1.0),
            wavelength_m=1.0,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("grid", object(), id="invalid-grid"),
        pytest.param("metadata", [], id="metadata-must-be-dict"),
    ],
)
def test_invalid_structural_metadata_is_rejected(keyword: str, value) -> None:
    parameters = {
        "data": torch.ones((1, 3, 4), dtype=torch.complex64),
        "grid": _grid(nx=4, ny=3),
        "wavelength_m": 1.31e-6,
        "components": ("scalar",),
        "metadata": {},
    }
    parameters[keyword] = value

    with pytest.raises((TypeError, ValueError)):
        Field2D(**parameters)


def test_component_sequence_is_canonicalized_without_mutating_input() -> None:
    components = ["Ex", "Ey"]
    field = Field2D(
        data=torch.ones((2, 3, 4), dtype=torch.complex64),
        grid=_grid(nx=4, ny=3),
        wavelength_m=1.31e-6,
        components=components,
    )

    assert field.components == ("Ex", "Ey")
    assert components == ["Ex", "Ey"]


@pytest.mark.parametrize(
    "wavelength_m",
    [0.0, -1.0, math.nan, math.inf],
    ids=["zero", "negative", "nan", "infinite"],
)
def test_invalid_wavelength_is_rejected(wavelength_m: float) -> None:
    grid = _grid(nx=4, ny=3)
    data = torch.ones((1, 3, 4), dtype=torch.complex64)

    with pytest.raises((TypeError, ValueError)):
        Field2D(data=data, grid=grid, wavelength_m=wavelength_m)


@pytest.mark.parametrize(
    "medium_index",
    [
        0.0,
        -1.0,
        -1.0 + 0.2j,
        complex(math.nan, 0.0),
        complex(1.0, math.nan),
        complex(math.inf, 0.0),
        complex(1.0, math.inf),
    ],
    ids=[
        "zero-real",
        "negative-real",
        "negative-real-complex",
        "nan-real",
        "nan-imag",
        "infinite-real",
        "infinite-imag",
    ],
)
def test_invalid_medium_index_is_rejected(medium_index: complex) -> None:
    grid = _grid(nx=4, ny=3)
    data = torch.ones((1, 3, 4), dtype=torch.complex64)

    with pytest.raises((TypeError, ValueError)):
        Field2D(
            data=data,
            grid=grid,
            wavelength_m=1.31e-6,
            medium_index=medium_index,
        )


@pytest.mark.parametrize("z_m", [math.nan, math.inf, -math.inf])
def test_non_finite_z_is_rejected(z_m: float) -> None:
    grid = _grid(nx=4, ny=3)
    data = torch.ones((1, 3, 4), dtype=torch.complex64)

    with pytest.raises((TypeError, ValueError)):
        Field2D(data=data, grid=grid, wavelength_m=1.31e-6, z_m=z_m)


def test_with_data_revalidates_shape_and_does_not_mutate_source() -> None:
    field = _scalar_field()
    before = field.data.clone()
    invalid = torch.ones((1, field.grid.nx, field.grid.ny), dtype=field.data.dtype)
    invalid_before = invalid.clone()

    with pytest.raises((TypeError, ValueError)):
        field.with_data(invalid)

    torch.testing.assert_close(field.data, before)
    torch.testing.assert_close(invalid, invalid_before)
