import numpy as np
import pandas as pd
import pytest

from .._did_utils import _check_g_t_values, _get_id_positions, _get_never_treated_value, _is_never_treated, _set_id_positions

valid_args = {
    "g_values": np.array([1, 2]),
    "t_values": np.array([0, 1, 2]),
    "control_group": "never_treated",
}


@pytest.mark.ci
def test_get_never_treated_value():
    assert _get_never_treated_value(np.array([1, 2])) == 0
    assert np.isnan(_get_never_treated_value(np.array([1.0, 2.0])))
    assert np.isnan(_get_never_treated_value(np.array([1.0, 2])))
    assert _get_never_treated_value(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64")) is pd.NaT
    assert _get_never_treated_value(np.array(["2024-01-01", "2024-01-02"])) == 0


@pytest.mark.ci
def test_is_never_treated():
    # check single values
    arguments = (
        (0, 0, True),
        (1, 0, False),
        (np.nan, np.nan, True),
        (0, np.nan, False),
        (pd.NaT, pd.NaT, True),
        (0, pd.NaT, False),
    )
    for x, never_treated_value, expected in arguments:
        assert _is_never_treated(x, never_treated_value) == expected

    # check arrays
    arguments = (
        (np.array([0, 1]), 0, np.array([True, False])),
        (np.array([0, 1]), np.nan, np.array([False, False])),
        (np.array([0, 1]), pd.NaT, np.array([False, False])),
        (np.array([0, np.nan]), 0, np.array([True, False])),
        (np.array([0, np.nan]), np.nan, np.array([False, True])),
        (np.array([0, pd.NaT]), 0, np.array([True, False])),
        (np.array([0, pd.NaT]), pd.NaT, np.array([False, True])),
    )
    for x, never_treated_value, expected in arguments:
        assert np.all(_is_never_treated(x, never_treated_value) == expected)


@pytest.mark.ci
def test_input_check_g_t_values():
    invalid_args = [
        ({"g_values": ["test"]}, TypeError, r"Invalid type for g_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"t_values": ["test"]}, TypeError, r"Invalid type for t_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"g_values": np.array([[1, 2]])}, ValueError, "g_values must be a vector. Number of dimensions is 2."),
        ({"t_values": np.array([[0, 1, 2]])}, ValueError, "t_values must be a vector. Number of dimensions is 2."),
        ({"g_values": None}, TypeError, "Invalid type for g_values."),
        ({"t_values": None}, TypeError, "Invalid type for t_values."),
        ({"t_values": np.array([0, 1, np.nan])}, ValueError, "t_values contains missing values."),
        (
            {"g_values": np.array([0, 1]), "t_values": np.array([0.0, 1.0, 2.0])},
            ValueError,
            "g_values and t_values must have the same data type. Got int64 and float64.",
        ),
    ]

    for arg, error, msg in invalid_args:
        with pytest.raises(error, match=msg):
            _check_g_t_values(**(valid_args | arg))


@pytest.mark.ci
def test_modify_g_values_check_g_t_values():
    arguments = [
        ({"g_values": [0, 1]}, UserWarning, "The never treated group 0 is removed from g_values."),
        ({"t_values": [1, 2]}, UserWarning, "Values before/equal the first period 1 are removed from g_values."),
        ({"g_values": [1, 2, 3]}, UserWarning, "Values after the last period 2 are removed from g_values."),
        (
            {"g_values": [1, 2], "control_group": "not_yet_treated"},
            UserWarning,
            r"Individuals treated in the last period are excluded from the analysis \(no comparison group available\).",
        ),
    ]

    for arg, error, msg in arguments:
        with pytest.warns(error, match=msg):
            _check_g_t_values(**(valid_args | arg))


def test_get_id_positions():
    # Test case 1: Normal array with valid positions
    a = np.array([1, 2, 3, 4, 5])
    id_positions = np.array([0, 2, 4])
    expected = np.array([1, 3, 5])
    result = _get_id_positions(a, id_positions)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: 2D array with valid positions
    a_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    id_positions = np.array([1, 3])
    expected_2d = np.array([[3, 4], [7, 8]])
    result_2d = _get_id_positions(a_2d, id_positions)
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test case 3: None input
    a_none = None
    id_positions = np.array([0, 1, 2])
    result_none = _get_id_positions(a_none, id_positions)
    assert result_none is None


def test_set_id_positions():
    # Test case 1: Basic 1D array
    a = np.array([1, 2, 3])
    n_obs = 5
    id_positions = np.array([1, 3, 4])
    fill_value = 0
    expected = np.array([0, 1, 0, 2, 3])
    result = _set_id_positions(a, n_obs, id_positions, fill_value)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: 2D array
    a_2d = np.array([[1, 2], [3, 4], [5, 6]])
    n_obs = 5
    id_positions = np.array([0, 2, 4])
    fill_value = -1
    expected_2d = np.array([[1, 2], [-1, -1], [3, 4], [-1, -1], [5, 6]])
    result_2d = _set_id_positions(a_2d, n_obs, id_positions, fill_value)
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test case 3: None input
    a_none = None
    n_obs = 3
    id_positions = np.array([0, 1])
    fill_value = 0
    result_none = _set_id_positions(a_none, n_obs, id_positions, fill_value)
    assert result_none is None
