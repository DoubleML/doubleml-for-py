import pytest

import numpy as np

from .._did_utils import _check_preprocess_g_t

valid_args = {
    "g_values": np.array([1, 2]),
    "t_values": np.array([0, 1, 2]),
    "control_group": "never_treated",
}


@pytest.mark.ci
def test_input_check_preprocess_g_t():
    invalid_args = [
        ({"g_values": ["test"]}, TypeError, r"Invalid type for g_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"t_values": ["test"]}, TypeError, r"Invalid type for t_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"g_values": np.array([[1, 2]])}, ValueError, "g_values must be a vector. Number of dimensions is 2."),
        ({"t_values": np.array([[0, 1, 2]])}, ValueError, "t_values must be a vector. Number of dimensions is 2."),
        ({"g_values": None}, TypeError, "Invalid type for g_values."),
        ({"t_values": None}, TypeError, "Invalid type for t_values."),
    ]

    for arg, error, msg in invalid_args:
        with pytest.raises(error, match=msg):
            _check_preprocess_g_t(**(valid_args | arg))


@pytest.mark.ci
def test_modify_g_values_check_preprocess_g_t():
    arguments = [
        ({"g_values": [0, 1]}, UserWarning, "The never treated group 0 is removed from g_values."),
        ({"t_values": [1, 2]}, UserWarning, "Values before/equal the first period 1 are removed from g_values."),
        ({"g_values": [1, 2, 3]}, UserWarning, "Values after the last period 2 are removed from g_values."),
        ({"g_values": [1, 2], "control_group": "not_yet_treated"}, UserWarning, r"Individuals treated in the last period are excluded from the analysis \(no comparison group available\)."),
    ]

    for arg, error, msg in arguments:
        with pytest.warns(error, match=msg):
            _check_preprocess_g_t(**(valid_args | arg))
