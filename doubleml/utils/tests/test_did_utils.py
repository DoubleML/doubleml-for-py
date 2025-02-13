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
    msg = r"Invalid type for g_values: expected one of \(<class 'int'>, <class 'float'>, <class 'numpy.ndarray'>\)."
    with pytest.raises(TypeError, match=msg):
        _check_preprocess_g_t(**(valid_args | {"g_values": ["test"]}))

    msg = r"Invalid type for t_values: expected one of \(<class 'int'>, <class 'float'>, <class 'numpy.ndarray'>\)."
    with pytest.raises(TypeError, match=msg):
        _check_preprocess_g_t(**(valid_args | {"t_values": ["test"]}))

    msg = "g_values must be a vector. Number of dimensions is 2."
    with pytest.raises(ValueError, match=msg):
        _check_preprocess_g_t(**(valid_args | {"g_values": np.array([[1, 2]])}))

    msg = "t_values must be a vector. Number of dimensions is 2."
    with pytest.raises(ValueError, match=msg):
        _check_preprocess_g_t(**(valid_args | {"t_values": np.array([[0, 1, 2]])}))


@pytest.mark.ci
def test_modify_g_values_check_preprocess_g_t():
    msg = "The never treated group 0 is removed from g_values."
    with pytest.warns(UserWarning, match=msg):
        _check_preprocess_g_t(**(valid_args | {"g_values": [0, 1]}))

    msg = "Values before/equal the first period 1 are removed from g_values."
    with pytest.warns(UserWarning, match=msg):
        _check_preprocess_g_t(**(valid_args | {"t_values": [1, 2]}))

    msg = "Values after the last period 2 are removed from g_values."
    with pytest.warns(UserWarning, match=msg):
        _check_preprocess_g_t(**(valid_args | {"g_values": [1, 2, 3]}))