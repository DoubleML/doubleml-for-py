import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021
from doubleml.utils._did_utils import _get_id_positions


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
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module", params=["datetime", "float"])
def time_type(request):
    return request.param


@pytest.fixture(scope="module")
def dml_did_binary_vs_did_fixture(time_type, learner, score, in_sample_normalization, trimming_threshold):
    n_obs = 500
    dpg = 1

    boot_methods = ["normal"]
    n_rep_boot = 50000

    # collect data
    df = make_did_CS2021(n_obs=n_obs, dgp_type=dpg, time_type=time_type)
    dml_panel_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

    dml_args = {
        "ml_g": clone(learner[0]),
        "ml_m": clone(learner[1]),
        "n_folds": 3,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": True,
    }

    dml_did_binary_obj = dml.did.DoubleMLDIDBinary(
        dml_panel_data,
        g_value=dml_panel_data.g_values[0],
        t_value_pre=dml_panel_data.t_values[0],
        t_value_eval=dml_panel_data.t_values[1],
        **dml_args,
    )
    dml_did_binary_obj.fit()

    df_wide = dml_did_binary_obj._panel_data_wide.copy()
    dml_data = dml.data.DoubleMLData(df_wide, y_col="y_diff", d_cols="G_indicator", x_cols=["Z1", "Z2", "Z3", "Z4"])
    dml_did_obj = dml.DoubleMLDID(
        dml_data,
        **dml_args,
    )

    # use external predictions (sample splitting is hard to synchronize)
    ext_predictions = {"G_indicator": {}}
    ext_predictions["G_indicator"]["ml_g0"] = _get_id_positions(
        dml_did_binary_obj.predictions["ml_g0"][:, :, 0], dml_did_binary_obj._id_positions)
    ext_predictions["G_indicator"]["ml_g1"] = _get_id_positions(dml_did_binary_obj.predictions["ml_g1"][:, :, 0], dml_did_binary_obj._id_positions)
    if score == "observational":
        ext_predictions["G_indicator"]["ml_m"] = _get_id_positions(dml_did_binary_obj.predictions["ml_m"][:, :, 0], dml_did_binary_obj._id_positions)
    dml_did_obj.fit(external_predictions=ext_predictions)

    res_dict = {
        "coef": dml_did_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "se": dml_did_obj.se,
        "se_binary": dml_did_binary_obj.se,
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_did_binary_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        # approximately same ci (bootstrap not identical due to size of score)
        res_dict["boot_ci" + bootstrap] = dml_did_obj.confint(joint=True)
        res_dict["boot_ci" + bootstrap + "_binary"] = dml_did_binary_obj.confint(joint=True)

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_did_obj.sensitivity_elements
    res_dict["sensitivity_elements_binary"] = dml_did_binary_obj.sensitivity_elements

    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_did_obj.sensitivity_analysis(rho=0.0)
    dml_did_binary_obj.sensitivity_analysis(rho=0.0)

    res_dict["sensitivity_ses"] = dml_did_obj.sensitivity_params["se"]
    res_dict["sensitivity_ses_binary"] = dml_did_binary_obj.sensitivity_params["se"]

    return res_dict


@pytest.mark.ci
def test_coefs(dml_did_binary_vs_did_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_fixture["coef_binary"][0], dml_did_binary_vs_did_fixture["coef"][0], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_ses(dml_did_binary_vs_did_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_fixture["se_binary"][0], dml_did_binary_vs_did_fixture["se"][0], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_boot(dml_did_binary_vs_did_fixture):
    for bootstrap in dml_did_binary_vs_did_fixture["boot_methods"]:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["boot_ci" + bootstrap].values,
            dml_did_binary_vs_did_fixture["boot_ci" + bootstrap + "_binary"].values,
            atol=1e-2,
        )


@pytest.mark.ci
@pytest.mark.skip(reason="Sensitivity still to implement in detal")
def test_sensitivity(dml_did_binary_vs_did_fixture):
    sensitivity_element_names = ["sigma2", "nu2", "psi_sigma2", "psi_nu2"]
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_elements"][sensitivity_element],
            dml_did_binary_vs_did_fixture["sensitivity_elements_binary"][sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_sensitivity_rho0(dml_did_binary_vs_did_fixture):
    assert np.allclose(
        dml_did_binary_vs_did_fixture["se"], dml_did_binary_vs_did_fixture["sensitivity_ses"]["lower"], rtol=1e-9, atol=1e-4
    )
    assert np.allclose(
        dml_did_binary_vs_did_fixture["se"], dml_did_binary_vs_did_fixture["sensitivity_ses"]["upper"], rtol=1e-9, atol=1e-4
    )

    assert np.allclose(
        dml_did_binary_vs_did_fixture["se_binary"],
        dml_did_binary_vs_did_fixture["sensitivity_ses_binary"]["lower"],
        rtol=1e-9,
        atol=1e-4,
    )
    assert np.allclose(
        dml_did_binary_vs_did_fixture["se_binary"],
        dml_did_binary_vs_did_fixture["sensitivity_ses_binary"]["upper"],
        rtol=1e-9,
        atol=1e-4,
    )
