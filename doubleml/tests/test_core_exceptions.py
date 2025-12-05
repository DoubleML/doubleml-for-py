import numpy as np
import pytest

from doubleml.double_ml_framework import DoubleMLCore
from doubleml.tests._utils import generate_dml_dict

n_obs = 10
n_thetas = 2
n_rep = 5


def valid_core_kwargs():
    np.random.seed(42)
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    return doubleml_dict


@pytest.mark.ci
def test_scaled_psi_shape_and_type():
    kwargs = valid_core_kwargs()
    msg = "scaled_psi must be a 3-dimensional numpy.ndarray."
    kwargs["scaled_psi"] = "not_an_array"
    with pytest.raises(ValueError, match=msg):
        DoubleMLCore(**kwargs)
    kwargs["scaled_psi"] = np.ones((10,))
    with pytest.raises(ValueError, match=msg):
        DoubleMLCore(**kwargs)
    kwargs["scaled_psi"] = np.ones((10, 2))
    with pytest.raises(ValueError, match=msg):
        DoubleMLCore(**kwargs)


@pytest.mark.ci
def test_arrays():
    kwargs = valid_core_kwargs()
    # Type checks
    for key in ["all_thetas", "all_ses", "var_scaling_factors"]:
        bad_kwargs = kwargs.copy()
        bad_kwargs[key] = "not_an_array"
        with pytest.raises(TypeError, match=f"{key} must be a numpy.ndarray"):
            DoubleMLCore(**bad_kwargs)
    # Shape checks
    shapes = {
        "all_thetas": (3, 5),
        "all_ses": (3, 5),
        "var_scaling_factors": (3,),
    }
    for key, shape in shapes.items():
        bad_kwargs = kwargs.copy()
        bad_kwargs[key] = np.ones(shape)
        with pytest.raises(ValueError, match=".*does not match expected.*"):
            DoubleMLCore(**bad_kwargs)


@pytest.mark.ci
def test_cluster_dict_exceptions():
    kwargs = valid_core_kwargs()
    kwargs["is_cluster_data"] = True

    # 1. cluster_dict missing
    bad_kwargs = kwargs.copy()
    bad_kwargs.pop("cluster_dict", None)
    with pytest.raises(ValueError, match="If is_cluster_data is True, cluster_dict must be provided."):
        DoubleMLCore(**bad_kwargs)

    # 2. cluster_dict not a dict
    bad_kwargs = kwargs.copy()
    bad_kwargs["cluster_dict"] = "not_a_dict"
    with pytest.raises(TypeError, match="cluster_dict must be a dictionary."):
        DoubleMLCore(**bad_kwargs)

    # 3. cluster_dict missing keys
    bad_kwargs = kwargs.copy()
    bad_kwargs["cluster_dict"] = {"smpls": [], "smpls_cluster": [], "cluster_vars": []}  # missing n_folds_per_cluster
    msg = "cluster_dict must contain keys: smpls, smpls_cluster, cluster_vars, n_folds_per_cluster."
    with pytest.raises(ValueError, match=msg):
        DoubleMLCore(**bad_kwargs)

    # 4. cluster_dict wrong value types
    type_cases = [
        ("smpls", "not_a_list", "cluster_dict\\['smpls'\\] must be a list."),
        ("smpls_cluster", "not_a_list", "cluster_dict\\['smpls_cluster'\\] must be a list."),
        ("cluster_vars", "not_a_list", "cluster_dict\\['cluster_vars'\\] must be a numpy.ndarray."),
        ("n_folds_per_cluster", "not_an_int", "cluster_dict\\['n_folds_per_cluster'\\] must be an int."),
    ]
    for key, bad_value, msg in type_cases:
        cluster_dict = {
            "smpls": [],
            "smpls_cluster": [],
            "cluster_vars": np.array([]),
            "n_folds_per_cluster": 1,
        }
        cluster_dict[key] = bad_value
        bad_kwargs = kwargs.copy()
        bad_kwargs["cluster_dict"] = cluster_dict
        with pytest.raises(TypeError, match=msg):
            DoubleMLCore(**bad_kwargs)


@pytest.mark.ci
def test_sensitivity_elements_exceptions():
    kwargs = valid_core_kwargs()

    # Not a dict
    bad_kwargs = kwargs.copy()
    bad_kwargs["sensitivity_elements"] = "not_a_dict"
    with pytest.raises(TypeError, match="sensitivity_elements must be a dict if provided."):
        DoubleMLCore(**bad_kwargs)

    # Missing required key
    bad_kwargs = kwargs.copy()
    bad_kwargs["sensitivity_elements"] = {"max_bias": np.ones((1, n_thetas, n_rep))}
    with pytest.raises(ValueError, match="sensitivity_elements must contain key 'psi_max_bias'."):
        DoubleMLCore(**bad_kwargs)

    # Wrong type for required key
    bad_kwargs = kwargs.copy()
    bad_kwargs["sensitivity_elements"] = {
        "max_bias": "not_an_array",
        "psi_max_bias": np.ones((n_obs, n_thetas, n_rep)),
    }
    with pytest.raises(TypeError, match="sensitivity_elements\\['max_bias'\\] must be a numpy.ndarray."):
        DoubleMLCore(**bad_kwargs)

    # Wrong shape for required key
    bad_kwargs = kwargs.copy()
    bad_kwargs["sensitivity_elements"] = {
        "max_bias": np.ones((2, n_thetas, n_rep)),  # should be (1, n_thetas, n_rep)
        "psi_max_bias": np.ones((n_obs, n_thetas, n_rep)),
    }
    with pytest.raises(
        ValueError, match=r"sensitivity_elements\['max_bias'\] shape \(2, 2, 5\) does not match expected \(1, 2, 5\)\."
    ):
        DoubleMLCore(**bad_kwargs)

    bad_kwargs = kwargs.copy()
    bad_kwargs["sensitivity_elements"] = {
        "max_bias": np.ones((1, n_thetas, n_rep)),
        "psi_max_bias": np.ones((n_obs + 1, n_thetas, n_rep)),  # wrong n_obs
    }
    with pytest.raises(
        ValueError, match=r"sensitivity_elements\['psi_max_bias'\] shape \(11, 2, 5\) does not match expected \(10, 2, 5\)\."
    ):
        DoubleMLCore(**bad_kwargs)

    # sigma2 and nu2 wrong type
    for key in ["sigma2", "nu2"]:
        bad_kwargs = kwargs.copy()
        sens = {
            "max_bias": np.ones((1, n_thetas, n_rep)),
            "psi_max_bias": np.ones((n_obs, n_thetas, n_rep)),
            key: "not_an_array",
        }
        bad_kwargs["sensitivity_elements"] = sens
        with pytest.raises(TypeError, match=rf"sensitivity_elements\['{key}'\] must be a numpy.ndarray."):
            DoubleMLCore(**bad_kwargs)

    # sigma2 and nu2 negative values
    for key in ["sigma2", "nu2"]:
        bad_kwargs = kwargs.copy()
        sens = {
            "max_bias": np.ones((1, n_thetas, n_rep)),
            "psi_max_bias": np.ones((n_obs, n_thetas, n_rep)),
            key: -np.ones((1, n_thetas, n_rep)),
        }
        bad_kwargs["sensitivity_elements"] = sens
        with pytest.raises(ValueError, match=rf"sensitivity_elements\['{key}'\] must be positive.*"):
            DoubleMLCore(**bad_kwargs)

    # sigma2 and nu2 wrong shape
    for key in ["sigma2", "nu2"]:
        bad_kwargs = kwargs.copy()
        sens = {
            "max_bias": np.ones((1, n_thetas, n_rep)),
            "psi_max_bias": np.ones((n_obs, n_thetas, n_rep)),
            key: np.ones((2, n_thetas, n_rep)),
        }
        bad_kwargs["sensitivity_elements"] = sens
        with pytest.raises(
            ValueError, match=rf"sensitivity_elements\['{key}'\] shape \(2, 2, 5\) does not match expected \(1, 2, 5\)\."
        ):
            DoubleMLCore(**bad_kwargs)
