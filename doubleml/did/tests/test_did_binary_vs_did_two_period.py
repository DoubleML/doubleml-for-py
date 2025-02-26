import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_manual import boot_did, fit_did, fit_sensitivity_elements_did


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


@pytest.fixture(scope="module")
def dml_did_binary_vs_did_fixture(generate_data_did_binary, learner, score, in_sample_normalization, trimming_threshold):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 499

    # collect data
    dml_panel_data = generate_data_did_binary
    df = dml_panel_data._data.sort_values(by=["id", "t"])
    df_panel = df.groupby("id").agg(
        {"y": lambda x: x.iloc[1] - x.iloc[0], "d": "first", "Z1": "first", "Z2": "first", "Z3": "first", "Z4": "first"}
    )

    n_obs = df_panel.shape[0]
    all_smpls = draw_smpls(n_obs, n_folds)
    obj_dml_data = dml.DoubleMLData(df_panel, y_col="y", d_cols="d", x_cols=["Z1", "Z2", "Z3", "Z4"])

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    dml_args = {
        "ml_g": ml_g,
        "ml_m": ml_m,
        "n_folds": n_folds,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": False,
    }

    dml_did_binary_obj = dml.did.DoubleMLDIDBinary(
        dml_panel_data,
        g_value=1,
        t_value_pre=0,
        t_value_eval=1,
        **dml_args,
    )

    dml_did_obj = dml.DoubleMLDID(
        obj_dml_data,
        **dml_args,
    )

    # synchronize the sample splitting
    dml_did_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_did_binary_obj.set_sample_splitting(all_smpls=all_smpls)

    dml_did_obj.fit()
    dml_did_binary_obj.fit()

    # manual fit
    y = df_panel["y"].values
    d = df_panel["d"].values
    x = df_panel[["Z1", "Z2", "Z3", "Z4"]].values

    np.random.seed(3141)
    res_manual = fit_did(
        y,
        x,
        d,
        clone(learner[0]),
        clone(learner[1]),
        all_smpls,
        score,
        in_sample_normalization,
        trimming_threshold=trimming_threshold,
    )

    res_dict = {
        "coef": dml_did_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "coef_manual": res_manual["theta"],
        "se": dml_did_obj.se,
        "se_binary": dml_did_binary_obj.se,
        "se_manual": res_manual["se"],
        "nuisance_loss": dml_did_obj.nuisance_loss,
        "nuisance_loss_binary": dml_did_binary_obj.nuisance_loss,
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_did(
            y,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_psi_a"],
            res_manual["all_psi_b"],
            all_smpls,
            bootstrap,
            n_rep_boot,
        )

        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_did_binary_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict["boot_t_stat" + bootstrap] = dml_did_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_binary"] = dml_did_binary_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_did_obj.sensitivity_elements
    res_dict["sensitivity_elements_binary"] = dml_did_binary_obj.sensitivity_elements
    res_dict["sensitivity_elements_manual"] = fit_sensitivity_elements_did(
        y,
        d,
        all_coef=dml_did_obj.all_coef,
        predictions=dml_did_obj.predictions,
        score=score,
        in_sample_normalization=in_sample_normalization,
        n_rep=1,
    )

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_did_obj.sensitivity_elements
    res_dict["sensitivity_elements_binary"] = dml_did_binary_obj.sensitivity_elements

    dml_did_obj.sensitivity_analysis()
    dml_did_binary_obj.sensitivity_analysis()

    res_dict["sensitivity_params"] = dml_did_obj.sensitivity_params
    res_dict["sensitivity_params_binary"] = dml_did_binary_obj.sensitivity_params

    return res_dict


@pytest.mark.ci
def test_coefs(dml_did_binary_vs_did_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_fixture["coef"][0], dml_did_binary_vs_did_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4
    )
    assert math.isclose(
        dml_did_binary_vs_did_fixture["coef_binary"][0], dml_did_binary_vs_did_fixture["coef"][0], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_ses(dml_did_binary_vs_did_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_fixture["se"][0], dml_did_binary_vs_did_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4
    )
    assert math.isclose(
        dml_did_binary_vs_did_fixture["se_binary"][0], dml_did_binary_vs_did_fixture["se"][0], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_boot(dml_did_binary_vs_did_fixture):
    for bootstrap in dml_did_binary_vs_did_fixture["boot_methods"]:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["boot_t_stat" + bootstrap],
            dml_did_binary_vs_did_fixture["boot_t_stat" + bootstrap + "_manual"],
            atol=1e-4,
        )
        assert np.allclose(
            dml_did_binary_vs_did_fixture["boot_t_stat" + bootstrap],
            dml_did_binary_vs_did_fixture["boot_t_stat" + bootstrap + "_binary"],
            atol=1e-4,
        )


@pytest.mark.ci
def test_nuisance_loss(dml_did_binary_vs_did_fixture):
    assert (
        dml_did_binary_vs_did_fixture["nuisance_loss"].keys() == dml_did_binary_vs_did_fixture["nuisance_loss_binary"].keys()
    )
    for key, value in dml_did_binary_vs_did_fixture["nuisance_loss"].items():
        assert np.allclose(value, dml_did_binary_vs_did_fixture["nuisance_loss_binary"][key], rtol=1e-9, atol=1e-3)


@pytest.mark.ci
def test_sensitivity_elements(dml_did_binary_vs_did_fixture):
    sensitivity_element_names = ["sigma2", "nu2", "psi_sigma2", "psi_nu2"]
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_elements"][sensitivity_element],
            dml_did_binary_vs_did_fixture["sensitivity_elements_manual"][sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_elements"][sensitivity_element],
            dml_did_binary_vs_did_fixture["sensitivity_elements_binary"][sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )
    for sensitivity_element in ["riesz_rep"]:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_elements"][sensitivity_element],
            dml_did_binary_vs_did_fixture["sensitivity_elements_binary"][sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_sensitivity_params(dml_did_binary_vs_did_fixture):
    for key in ["theta", "se", "ci"]:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_params"][key]["lower"],
            dml_did_binary_vs_did_fixture["sensitivity_params_binary"][key]["lower"],
            rtol=1e-9,
            atol=1e-4,
        )
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_params"][key]["upper"],
            dml_did_binary_vs_did_fixture["sensitivity_params_binary"][key]["upper"],
            rtol=1e-9,
            atol=1e-4,
        )

    for key in ["rv", "rva"]:
        assert np.allclose(
            dml_did_binary_vs_did_fixture["sensitivity_params"][key],
            dml_did_binary_vs_did_fixture["sensitivity_params_binary"][key],
            rtol=1e-9,
            atol=1e-4,
        )
