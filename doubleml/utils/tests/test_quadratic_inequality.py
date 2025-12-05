import numpy as np
import pytest

from doubleml.utils._estimation import _solve_quadratic_inequality


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (1, 0, -4, [(-2.0, 2.0)]),  # happy quadratic, determinant > 0
        (-1, 0, 4, [(-np.inf, -2), (2, np.inf)]),  # sad quadratic, determinant > 0
        (1, 0, 4, []),  # happy quadratic, determinant < 0
        (-1, 0, -4, [(-np.inf, np.inf)]),  # sad quadratic, determinant < 0
        (1, 0, 0, [(0.0, 0.0)]),  # happy quadratic, determinant = 0
        (-1, 0, 0, [(-np.inf, np.inf)]),  # sad quadratic, determinant = 0
        (1, 3, -4, [(-4.0, 1.0)]),  # happy quadratic, determinant > 0
        (-1, 3, 4, [(-np.inf, -1), (4, np.inf)]),  # sad quadratic, determinant > 0
        (-1, -3, 4, [(-np.inf, -4), (1, np.inf)]),  # sad quadratic, determinant > 0
        (1, 3, 4, []),  # happy quadratic, determinant < 0
        (-1, 3, -4, [(-np.inf, np.inf)]),  # sad quadratic, determinant < 0
        (1, 4, 4, [(-2.0, -2.0)]),  # happy quadratic, determinant = 0
        (-1, 4, -4, [(-np.inf, np.inf)]),  # sad quadratic, determinant = 0
        (0, 0, 0, [(-np.inf, np.inf)]),  # constant and equal to zero
        (0, 0, 1, []),  # constant and larger than zero
        (0, 1, 0, [(-np.inf, 0.0)]),  # increasing linear function
        (0, -1, -1, [(-1.0, np.inf)]),  # decreasing linear function
    ],
)
def test_solve_quadratic_inequation(a, b, c, expected):
    np.random.seed(42)
    result = _solve_quadratic_inequality(a, b, c)

    assert len(result) == len(expected), f"Expected {len(expected)} intervals but got {len(result)}"

    for i, tpl in enumerate(result):
        if tpl[0] == -np.inf:
            assert np.isinf(tpl[0])
        if tpl[1] == np.inf:
            assert np.isinf(tpl[1])
        else:
            assert np.isclose(tpl[0], expected[i][0]), f"Expected {expected[i][0]} but got {tpl[0]}"
            assert np.isclose(tpl[1], expected[i][1]), f"Expected {expected[i][1]} but got {tpl[1]}"
