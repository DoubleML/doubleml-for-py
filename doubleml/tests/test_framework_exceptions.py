import pytest
import numpy as np

from doubleml.double_ml_framework import DoubleMLFramework, concat
from ._utils import generate_dml_dict

n_obs = 10
n_thetas = 2
n_rep = 5

# generate score samples
psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
doubleml_dict = generate_dml_dict(psi_a, psi_b)

# combine objects and estimate parameters
dml_framework_obj_1 = DoubleMLFramework(doubleml_dict)


@pytest.mark.ci
def test_input_exceptions():
    msg = r"The dict must contain the following keys: thetas, ses, all_thetas, all_ses, var_scaling_factors, scaled_psi"
    with pytest.raises(ValueError, match=msg):
        test_dict = {}
        DoubleMLFramework(test_dict)

    msg = r"The shape of thetas does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['thetas'] = np.ones(shape=(1,))
        DoubleMLFramework(test_dict)

    msg = r"The shape of ses does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['ses'] = np.ones(shape=(1,))
        DoubleMLFramework(test_dict)

    msg = r"The shape of all_thetas does not match the expected shape \(2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['all_thetas'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of all_ses does not match the expected shape \(2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['all_ses'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of var_scaling_factors does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['var_scaling_factors'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of scaled_psi does not match the expected shape \(10, 2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['scaled_psi'] = np.ones(shape=(10, 2, 5, 3))
        DoubleMLFramework(test_dict)

    msg = "doubleml_dict must be a dictionary."
    with pytest.raises(AssertionError, match=msg):
        DoubleMLFramework(1.0)


def test_operation_exceptions():
    # addition
    msg = "Unsupported operand type: <class 'float'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 + 1.0
    with pytest.raises(TypeError, match=msg):
        _ = 1.0 + dml_framework_obj_1
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2
    msg = 'The number of parameters theta in DoubleMLFrameworks must be the same. Got 2 and 3.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas + 1, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas + 1, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2

    # subtraction
    msg = "Unsupported operand type: <class 'float'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 - 1.0
    with pytest.raises(TypeError, match=msg):
        _ = 1.0 - dml_framework_obj_1
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2
    msg = 'The number of parameters theta in DoubleMLFrameworks must be the same. Got 2 and 3.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas + 1, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas + 1, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2

    # multiplication
    msg = "Unsupported operand type: <class 'dict'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 * {}
    with pytest.raises(TypeError, match=msg):
        _ = {} * dml_framework_obj_1

    # concatenation
    msg = 'Need at least one object to concatenate.'
    with pytest.raises(TypeError, match=msg):
        concat([])
    msg = 'All objects must be of type DoubleMLFramework.'
    with pytest.raises(TypeError, match=msg):
        concat([dml_framework_obj_1, 1.0])
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = concat([dml_framework_obj_1, dml_framework_obj_2])
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = concat([dml_framework_obj_1, dml_framework_obj_2])


@pytest.mark.ci
def test_p_adjust_exceptions():
    msg = "The p_adjust method must be of str type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.p_adjust(method=1)

    msg = r'Apply bootstrap\(\) before p_adjust\("rw"\)\.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.p_adjust(method='rw')
