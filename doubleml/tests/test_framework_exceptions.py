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
psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep))
psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep)) + 1.0
doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)

# combine objects and estimate parameters
dml_framework_obj_1 = DoubleMLFramework(doubleml_dict)
dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)


@pytest.mark.ci
def test_framework_input_exceptions():
    msg = r"The dict must contain the following keys: thetas, ses, all_thetas, all_ses, var_scaling_factors, scaled_psi"
    with pytest.raises(ValueError, match=msg):
        test_dict = {}
        DoubleMLFramework(test_dict)

    msg = "doubleml_obj must be of type DoubleML or dictionary."
    with pytest.raises(AssertionError, match=msg):
        DoubleMLFramework(1.0)
