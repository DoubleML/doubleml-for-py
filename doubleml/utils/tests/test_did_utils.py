import numpy as np
import pandas as pd
import pytest

from .._did_utils import _check_g_t_values, _get_never_treated_value, _is_never_treated

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
