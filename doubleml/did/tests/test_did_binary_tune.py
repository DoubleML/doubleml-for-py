import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_manual import boot_did, fit_did, tune_nuisance_did


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42)])
def learner_g(request):
    return request.param


@pytest.fixture(scope="module", params=[LogisticRegression()])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=["observational", "experimental"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def in_sample_normalization(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestRegressor]:
        par_grid = {"n_estimators": [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression]
        par_grid = {"C": np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope="module")
def dml_did_fixture(generate_data_did_binary, learner_g, learner_m, score, in_sample_normalization, tune_on_folds):
    par_grid = {"ml_g": get_par_grid(learner_g), "ml_m": get_par_grid(learner_m)}
    n_folds_tune = 4

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
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=df_panel["d"])
    obj_dml_data = dml.DoubleMLData(df_panel, y_col="y", d_cols="d", x_cols=["Z1", "Z2", "Z3", "Z4"])

    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    dml_args = {
        "ml_g": ml_g,
        "ml_m": ml_m,
        "n_folds": n_folds,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
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

    # tune hyperparameters
    np.random.seed(3141)
    tune_res = dml_did_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLDID)
    np.random.seed(3141)
    tune_res_binary = dml_did_binary_obj.tune(
        par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False
    )
    assert isinstance(tune_res_binary, dml.did.DoubleMLDIDBinary)

    dml_did_obj.fit()
    dml_did_binary_obj.fit()

    # manual fit
    y = df_panel["y"].values
    d = df_panel["d"].values
    x = df_panel[["Z1", "Z2", "Z3", "Z4"]].values
    np.random.seed(3141)
    smpls = all_smpls[0]

    if tune_on_folds:
        g0_params, g1_params, m_params = tune_nuisance_did(
            y, x, d, clone(learner_g), clone(learner_m), smpls, score, n_folds_tune, par_grid["ml_g"], par_grid["ml_m"]
        )
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_params, g1_params, m_params = tune_nuisance_did(
            y, x, d, clone(learner_g), clone(learner_m), xx, score, n_folds_tune, par_grid["ml_g"], par_grid["ml_m"]
        )
        g0_params = g0_params * n_folds
        if score == "experimental":
            g1_params = g1_params * n_folds
            m_params = None
        else:
            assert score == "observational"
            g1_params = None
            m_params = m_params * n_folds

    res_manual = fit_did(
        y,
        x,
        d,
        clone(learner_g),
        clone(learner_m),
        all_smpls,
        score,
        in_sample_normalization,
        g0_params=g0_params,
        g1_params=g1_params,
        m_params=m_params,
    )

    res_dict = {
        "coef": dml_did_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "coef_manual": res_manual["theta"],
        "se": dml_did_obj.se,
        "se_binary": dml_did_binary_obj.se,
        "se_manual": res_manual["se"],
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

    return res_dict


@pytest.mark.ci
def test_dml_did_coef(dml_did_fixture):
    assert math.isclose(dml_did_fixture["coef"][0], dml_did_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_did_fixture["coef_binary"][0], dml_did_fixture["coef"][0], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_se(dml_did_fixture):
    assert math.isclose(dml_did_fixture["se"][0], dml_did_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_did_fixture["se_binary"][0], dml_did_fixture["se"][0], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_boot(dml_did_fixture):
    for bootstrap in dml_did_fixture["boot_methods"]:
        assert np.allclose(
            dml_did_fixture["boot_t_stat" + bootstrap],
            dml_did_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )

        assert np.allclose(
            dml_did_fixture["boot_t_stat" + bootstrap],
            dml_did_fixture["boot_t_stat" + bootstrap + "_binary"],
            rtol=1e-9,
            atol=1e-4,
        )
