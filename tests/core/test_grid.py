"""Contract tests for the canonical two-dimensional sampling grid."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from optiprop.core import Grid2D


def _as_float64_tensor(value) -> torch.Tensor:
    """Compare coordinate implementations without constraining their storage dtype."""
    return torch.as_tensor(value, dtype=torch.float64)


def test_rectangular_127_by_192_grid_uses_yx_axis_convention() -> None:
    """A [127, 192] field has 127 y samples and 192 x samples."""
    grid = Grid2D(
        nx=192,
        ny=127,
        dx=0.37e-6,
        dy=0.81e-6,
        x_center=4.25e-6,
        y_center=-2.75e-6,
    )

    x = _as_float64_tensor(grid.x())
    y = _as_float64_tensor(grid.y())
    expected_x = (
        torch.arange(192, dtype=torch.float64) - (192 - 1) / 2
    ) * grid.dx + grid.x_center
    expected_y = (
        torch.arange(127, dtype=torch.float64) - (127 - 1) / 2
    ) * grid.dy + grid.y_center

    assert x.shape == (192,)
    assert y.shape == (127,)
    torch.testing.assert_close(x, expected_x, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(y, expected_y, rtol=0.0, atol=1e-15)
    assert x[96] - x[95] == pytest.approx(grid.dx)
    assert y[64] - y[63] == pytest.approx(grid.dy)


@pytest.mark.parametrize(
    ("nx", "ny"),
    [
        pytest.param(5, 7, id="odd-by-odd"),
        pytest.param(4, 6, id="even-by-even"),
        pytest.param(4, 7, id="even-by-odd"),
        pytest.param(5, 6, id="odd-by-even"),
    ],
)
def test_odd_and_even_coordinates_have_the_requested_center(
    nx: int, ny: int
) -> None:
    grid = Grid2D(
        nx=nx,
        ny=ny,
        dx=2.0e-6,
        dy=3.0e-6,
        x_center=11.0e-6,
        y_center=-13.0e-6,
    )

    x = _as_float64_tensor(grid.x())
    y = _as_float64_tensor(grid.y())

    assert float((x[0] + x[-1]) / 2) == pytest.approx(grid.x_center)
    assert float((y[0] + y[-1]) / 2) == pytest.approx(grid.y_center)
    if nx % 2:
        assert float(x[nx // 2]) == pytest.approx(grid.x_center)
    else:
        assert float((x[nx // 2 - 1] + x[nx // 2]) / 2) == pytest.approx(
            grid.x_center
        )
        assert not bool(
            torch.isclose(
                x,
                torch.tensor(grid.x_center, dtype=x.dtype),
                rtol=0.0,
                atol=1e-15,
            ).any()
        )
    if ny % 2:
        assert float(y[ny // 2]) == pytest.approx(grid.y_center)
    else:
        assert float((y[ny // 2 - 1] + y[ny // 2]) / 2) == pytest.approx(
            grid.y_center
        )
        assert not bool(
            torch.isclose(
                y,
                torch.tensor(grid.y_center, dtype=y.dtype),
                rtol=0.0,
                atol=1e-15,
            ).any()
        )


def test_grid_is_frozen() -> None:
    grid = Grid2D(nx=3, ny=5, dx=1.0, dy=2.0)

    with pytest.raises(FrozenInstanceError):
        grid.nx = 9


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("nx", 0, id="zero-nx"),
        pytest.param("nx", -1, id="negative-nx"),
        pytest.param("ny", 0, id="zero-ny"),
        pytest.param("ny", -1, id="negative-ny"),
        pytest.param("nx", 4.5, id="fractional-nx"),
        pytest.param("ny", True, id="boolean-ny"),
        pytest.param("dx", 0.0, id="zero-dx"),
        pytest.param("dx", -1.0, id="negative-dx"),
        pytest.param("dy", 0.0, id="zero-dy"),
        pytest.param("dy", -1.0, id="negative-dy"),
        pytest.param("dx", math.nan, id="nan-dx"),
        pytest.param("dy", math.inf, id="infinite-dy"),
        pytest.param("x_center", math.nan, id="nan-x-center"),
        pytest.param("y_center", math.inf, id="infinite-y-center"),
    ],
)
def test_invalid_grid_parameters_are_rejected(keyword: str, value) -> None:
    parameters = {
        "nx": 5,
        "ny": 7,
        "dx": 1.0e-6,
        "dy": 2.0e-6,
        "x_center": 0.0,
        "y_center": 0.0,
    }
    parameters[keyword] = value

    with pytest.raises((TypeError, ValueError)):
        Grid2D(**parameters)
