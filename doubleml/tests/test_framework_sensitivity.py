import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.double_ml_framework import concat
from doubleml.irm.irm import DoubleMLIRM


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_framework_sensitivity_fixture(n_rep, generate_data_simple):
    dml_data, dml_data_2 = generate_data_simple

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    dml_irm_obj = DoubleMLIRM(dml_data, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj.fit()
    dml_framework_obj = dml_irm_obj.construct_framework()
    dml_framework_obj.sensitivity_analysis()

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj
    dml_framework_obj_add_obj.sensitivity_analysis()

    # substract objects
    dml_irm_obj_2 = DoubleMLIRM(dml_data_2, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj_2.fit()
    dml_framework_obj_2 = dml_irm_obj_2.construct_framework()
    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2
    dml_framework_obj_sub_obj2 = dml_framework_obj - dml_framework_obj
    dml_framework_obj_sub_obj2.sensitivity_analysis()

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2
    dml_framework_obj_mul_obj.sensitivity_analysis()
    dml_framework_obj_mul_zero_obj = 0 * dml_framework_obj
    dml_framework_obj_mul_zero_obj.sensitivity_analysis()

    # concat objects
    dml_framework_obj_concat = concat([dml_framework_obj, dml_framework_obj])

    result_dict = {
        "dml_obj": dml_irm_obj,
        "dml_obj_2": dml_irm_obj_2,
        "dml_framework_obj": dml_framework_obj,
        "dml_framework_obj_2": dml_framework_obj_2,
        "dml_framework_obj_add_obj": dml_framework_obj_add_obj,
        "dml_framework_obj_sub_obj": dml_framework_obj_sub_obj,
        "dml_framework_obj_sub_obj2": dml_framework_obj_sub_obj2,
        "dml_framework_obj_mul_obj": dml_framework_obj_mul_obj,
        "dml_framework_obj_mul_zero_obj": dml_framework_obj_mul_zero_obj,
        "dml_framework_obj_concat": dml_framework_obj_concat,
        "n_rep": n_rep,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_sensitivity_shapes(dml_framework_sensitivity_fixture):
    n_rep = dml_framework_sensitivity_fixture["dml_framework_obj"].n_rep
    n_thetas = dml_framework_sensitivity_fixture["dml_framework_obj"].n_thetas
    n_obs = dml_framework_sensitivity_fixture["dml_framework_obj"].n_obs

    object_list = [
        "dml_framework_obj",
        "dml_framework_obj_2",
        "dml_framework_obj_add_obj",
        "dml_framework_obj_sub_obj",
        "dml_framework_obj_mul_obj",
    ]
    var_keys = ["max_bias"]
    score_keys = ["psi_max_bias"]

    for obj in object_list:
        assert dml_framework_sensitivity_fixture[obj]._sensitivity_implemented
        for key in var_keys:
            assert dml_framework_sensitivity_fixture[obj].sensitivity_elements[key].shape == (1, n_thetas, n_rep)
        for key in score_keys:
            assert dml_framework_sensitivity_fixture[obj].sensitivity_elements[key].shape == (n_obs, n_thetas, n_rep)

    # separate test for concat
    for key in var_keys:
        assert dml_framework_sensitivity_fixture["dml_framework_obj_concat"].sensitivity_elements[key].shape == (1, 2, n_rep)
    for key in score_keys:
        assert dml_framework_sensitivity_fixture["dml_framework_obj_concat"].sensitivity_elements[key].shape == (
            n_obs,
            2,
            n_rep,
        )


@pytest.mark.ci
def test_dml_framework_sensitivity_summary(dml_framework_sensitivity_fixture):
    # summary without sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture["dml_framework_obj_2"].sensitivity_summary
    substring = "Apply sensitivity_analysis() to generate sensitivity_summary."
    assert substring in sensitivity_summary

    # summary with sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture["dml_framework_obj"].sensitivity_summary
    assert isinstance(sensitivity_summary, str)
    substrings = [
        "\n------------------ Scenario          ------------------\n",
        "\n------------------ Bounds with CI    ------------------\n",
        "\n------------------ Robustness Values ------------------\n",
        "Significance Level: level=",
        "Sensitivity parameters: cf_y=",
    ]
    for substring in substrings:
        assert substring in sensitivity_summary


@pytest.mark.ci
def test_dml_framework_sensitivity_operations(dml_framework_sensitivity_fixture):
    add_obj = dml_framework_sensitivity_fixture["dml_framework_obj_add_obj"]
    mul_obj = dml_framework_sensitivity_fixture["dml_framework_obj_mul_obj"]
    mul_zero_obj = dml_framework_sensitivity_fixture["dml_framework_obj_mul_zero_obj"]

    for key in ["theta", "se", "ci"]:
        assert np.allclose(add_obj.sensitivity_params[key]["upper"], mul_obj.sensitivity_params[key]["upper"])
        assert np.allclose(add_obj.sensitivity_params[key]["lower"], mul_obj.sensitivity_params[key]["lower"])

        assert mul_zero_obj.sensitivity_params[key]["upper"] == 0
        assert mul_zero_obj.sensitivity_params[key]["lower"] == 0

    assert np.allclose(add_obj.sensitivity_params["rv"], mul_obj.sensitivity_params["rv"])
    assert mul_zero_obj.sensitivity_params["rv"] > 0.99  # due to degenerated variance
    assert mul_zero_obj.sensitivity_params["rva"] > 0.99  # due to degenerated variance

    sub_obj = dml_framework_sensitivity_fixture["dml_framework_obj_sub_obj2"]
    assert np.allclose(sub_obj.sensitivity_params["theta"]["upper"], -1 * sub_obj.sensitivity_params["theta"]["lower"])
    assert np.allclose(sub_obj.sensitivity_params["se"]["upper"], sub_obj.sensitivity_params["se"]["lower"])
    assert np.allclose(sub_obj.sensitivity_params["ci"]["upper"], -1 * sub_obj.sensitivity_params["ci"]["lower"])
    assert sub_obj.sensitivity_params["rv"] < 0.01
    assert sub_obj.sensitivity_params["rva"] < 0.01
