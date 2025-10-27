import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_cs_CS2021
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250)],
        [
            RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        ],
    ],
)
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["observational", "experimental"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def in_sample_normalization(request):
    return request.param


@pytest.fixture(scope="module", params=[0.1])
def clipping_threshold(request):
    return request.param


@pytest.fixture(scope="module", params=["datetime", "float"])
def time_type(request):
    return request.param


@pytest.fixture(scope="module", params=[0.5, 0.1])
def lambda_t(request):
    return request.param


@pytest.fixture(scope="module")
def dml_did_binary_vs_did_multi_fixture(time_type, lambda_t, learner, score, in_sample_normalization, clipping_threshold):
    n_obs = 500
    dpg = 1
    boot_methods = ["normal"]
    n_rep_boot = 500

    # collect data
    df = make_did_cs_CS2021(n_obs=n_obs, dgp_type=dpg, time_type=time_type, lambda_t=lambda_t)
    dml_panel_data = dml.data.DoubleMLPanelData(
        df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
    )

    dml_args = {
        "n_folds": 3,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "ps_processor_config": dml.utils.PSProcessorConfig(clipping_threshold=clipping_threshold),
        "draw_sample_splitting": True,
    }
    gt_combination = [(dml_panel_data.g_values[0], dml_panel_data.t_values[0], dml_panel_data.t_values[1])]
    dml_did_multi_obj = dml.did.DoubleMLDIDMulti(
        dml_panel_data,
        ml_g=learner[0],
        ml_m=learner[1],
        gt_combinations=gt_combination,
        panel=False,
        **dml_args,
    )
    dml_did_multi_obj.fit()

    treatment_col = dml_panel_data.d_cols[0]
    ext_pred_dict = {treatment_col: {}}
    all_keys = ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1"]
    for key in all_keys:
        ext_pred_dict["d"][key] = dml_did_multi_obj.modellist[0].predictions[key][:, :, 0]
    if score == "observational":
        ext_pred_dict[treatment_col]["ml_m"] = dml_did_multi_obj.modellist[0].predictions["ml_m"][:, :, 0]

    dml_did_binary_obj = dml.did.DoubleMLDIDCSBinary(
        dml_panel_data,
        g_value=gt_combination[0][0],
        t_value_pre=gt_combination[0][1],
        t_value_eval=gt_combination[0][2],
        ml_g=DMLDummyRegressor(),
        ml_m=DMLDummyClassifier(),
        **dml_args,
    )
    dml_did_binary_obj.fit(external_predictions=ext_pred_dict)

    res_dict = {
        "coef_multi": dml_did_multi_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "se_multi": dml_did_multi_obj.se,
        "se_binary": dml_did_binary_obj.se,
        "boot_methods": boot_methods,
        "nuisance_loss_multi": dml_did_multi_obj.nuisance_loss,
        "nuisance_loss_binary": dml_did_binary_obj.nuisance_loss,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_did_multi_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_did_binary_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        # approximately same ci (bootstrap not identical due to size of score)
        res_dict["boot_ci" + bootstrap + "_multi"] = dml_did_multi_obj.confint(joint=True)
        res_dict["boot_ci" + bootstrap + "_binary"] = dml_did_binary_obj.confint(joint=True)

    # sensitivity tests
    res_dict["sensitivity_elements_multi"] = dml_did_multi_obj.sensitivity_elements
    res_dict["sensitivity_elements_binary"] = dml_did_binary_obj.framework.sensitivity_elements

    dml_did_multi_obj.sensitivity_analysis()
    dml_did_binary_obj.sensitivity_analysis()

    res_dict["sensitivity_params_multi"] = dml_did_multi_obj.sensitivity_params
    res_dict["sensitivity_params_binary"] = dml_did_binary_obj.sensitivity_params

    return res_dict


@pytest.mark.ci
def test_coefs(dml_did_binary_vs_did_multi_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_multi_fixture["coef_binary"][0],
        dml_did_binary_vs_did_multi_fixture["coef_multi"][0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


@pytest.mark.ci
def test_se(dml_did_binary_vs_did_multi_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_multi_fixture["se_binary"][0],
        dml_did_binary_vs_did_multi_fixture["se_multi"][0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


@pytest.mark.ci
def test_boot(dml_did_binary_vs_did_multi_fixture):
    for bootstrap in dml_did_binary_vs_did_multi_fixture["boot_methods"]:
        assert np.allclose(
            dml_did_binary_vs_did_multi_fixture["boot_ci" + bootstrap + "_multi"].values,
            dml_did_binary_vs_did_multi_fixture["boot_ci" + bootstrap + "_binary"].values,
            atol=1e-2,
        )


@pytest.mark.ci
def test_nuisance_loss(dml_did_binary_vs_did_multi_fixture):
    assert (
        dml_did_binary_vs_did_multi_fixture["nuisance_loss_multi"].keys()
        == dml_did_binary_vs_did_multi_fixture["nuisance_loss_binary"].keys()
    )
    for key, value in dml_did_binary_vs_did_multi_fixture["nuisance_loss_multi"].items():
        assert np.allclose(value, dml_did_binary_vs_did_multi_fixture["nuisance_loss_binary"][key], rtol=1e-9, atol=1e-3)


@pytest.mark.ci
def test_sensitivity_elements(dml_did_binary_vs_did_multi_fixture):
    elements_multi = dml_did_binary_vs_did_multi_fixture["sensitivity_elements_multi"]
    elements_binary = dml_did_binary_vs_did_multi_fixture["sensitivity_elements_binary"]
    sensitivity_element_names = ["max_bias", "psi_max_bias", "sigma2", "nu2"]
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(
            elements_multi[sensitivity_element],
            elements_binary[sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_sensitivity_params(dml_did_binary_vs_did_multi_fixture):
    multi_params = dml_did_binary_vs_did_multi_fixture["sensitivity_params_multi"]
    binary_params = dml_did_binary_vs_did_multi_fixture["sensitivity_params_binary"]
    for key in ["theta", "se", "ci"]:
        assert np.allclose(
            multi_params[key]["lower"],
            binary_params[key]["lower"],
            rtol=1e-9,
            atol=1e-4,
        )
        assert np.allclose(
            multi_params[key]["upper"],
            binary_params[key]["upper"],
            rtol=1e-9,
            atol=1e-4,
        )

    for key in ["rv", "rva"]:
        assert np.allclose(
            multi_params[key],
            binary_params[key],
            rtol=1e-9,
            atol=1e-4,
        )
