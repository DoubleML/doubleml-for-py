import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_cs_manual import fit_did_cs, tune_nuisance_did_cs
from ._utils_did_manual import boot_did


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
def dml_did_cs_fixture(generate_data_did_cs, learner_g, learner_m, score, in_sample_normalization, tune_on_folds):
    par_grid = {"ml_g": get_par_grid(learner_g), "ml_m": get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d, t) = generate_data_did_cs

    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d + 2 * t)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, t=t)
    dml_did_cs_obj = dml.DoubleMLDIDCS(
        obj_dml_data,
        ml_g,
        ml_m,
        n_folds,
        score=score,
        in_sample_normalization=in_sample_normalization,
        draw_sample_splitting=False,
    )
    # synchronize the sample splitting
    dml_did_cs_obj.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(3141)
    # tune hyperparameters
    tune_res = dml_did_cs_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLDIDCS)

    dml_did_cs_obj.fit()

    np.random.seed(3141)
    smpls = all_smpls[0]

    if tune_on_folds:
        g_d0_t0_params, g_d0_t1_params, g_d1_t0_params, g_d1_t1_params, m_params = tune_nuisance_did_cs(
            y, x, d, t, clone(learner_g), clone(learner_m), smpls, score, n_folds_tune, par_grid["ml_g"], par_grid["ml_m"]
        )
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_d0_t0_params, g_d0_t1_params, g_d1_t0_params, g_d1_t1_params, m_params = tune_nuisance_did_cs(
            y, x, d, t, clone(learner_g), clone(learner_m), xx, score, n_folds_tune, par_grid["ml_g"], par_grid["ml_m"]
        )
        g_d0_t0_params = g_d0_t0_params * n_folds
        g_d0_t1_params = g_d0_t1_params * n_folds
        g_d1_t0_params = g_d1_t0_params * n_folds
        g_d1_t1_params = g_d1_t1_params * n_folds
        if score == "observational":
            m_params = m_params * n_folds
        else:
            assert score == "experimental"
            m_params = None

    res_manual = fit_did_cs(
        y,
        x,
        d,
        t,
        clone(learner_g),
        clone(learner_m),
        all_smpls,
        score,
        in_sample_normalization,
        g_d0_t0_params=g_d0_t0_params,
        g_d0_t1_params=g_d0_t1_params,
        g_d1_t0_params=g_d1_t0_params,
        g_d1_t1_params=g_d1_t1_params,
        m_params=m_params,
    )

    res_dict = {
        "coef": dml_did_cs_obj.coef,
        "coef_manual": res_manual["theta"],
        "se": dml_did_cs_obj.se,
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
        dml_did_cs_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_did_cs_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_did_cs_coef(dml_did_cs_fixture):
    assert math.isclose(dml_did_cs_fixture["coef"][0], dml_did_cs_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_cs_se(dml_did_cs_fixture):
    assert math.isclose(dml_did_cs_fixture["se"][0], dml_did_cs_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_cs_boot(dml_did_cs_fixture):
    for bootstrap in dml_did_cs_fixture["boot_methods"]:
        assert np.allclose(
            dml_did_cs_fixture["boot_t_stat" + bootstrap],
            dml_did_cs_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )
