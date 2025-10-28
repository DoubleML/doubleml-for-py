import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_lplr_manual import fit_selection, tune_nuisance

@pytest.fixture(scope="module", params=[RandomForestClassifier(random_state=42)])
def learner_M(request):
    return request.param

@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42)])
def learner_t(request):
    return request.param


@pytest.fixture(scope="module", params=[LogisticRegression(random_state=42)])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestRegressor, RandomForestClassifier]:
        par_grid = {"n_estimators": [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression, Lasso]
        par_grid = {"C": np.logspace(-2, 2, 10)}
    return par_grid


@pytest.fixture(scope="module")
def dml_lplr_fixture(
    generate_data_selection,
    learner_M,
    learner_t,
    learner_m,
    score,
    tune_on_folds,
):
    par_grid = {"ml_M": get_par_grid(learner_M), "ml_t": get_par_grid(learner_t), "ml_m": get_par_grid(learner_m)}
    n_folds_tune = 4
    n_folds = 2

    # collect data
    np.random.seed(42)
    x, y, d = generate_data_selection


    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    ml_M = clone(learner_M)
    ml_t = clone(learner_t)
    ml_m = clone(learner_m)

    np.random.seed(42)

    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_sel_obj = dml.DoubleMLLPLR(
        obj_dml_data,
        ml_M,
        ml_t,
        ml_m,
        n_folds=n_folds,
        score=score,
        draw_sample_splitting=False,
    )


    # synchronize the sample splitting
    np.random.seed(42)
    dml_sel_obj.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(42)
    # tune hyperparameters
    tune_res = dml_sel_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLSSM)

    dml_sel_obj.fit()

    np.random.seed(42)
    smpls = all_smpls[0]
    if tune_on_folds:

        M_best_params, t_best_params, m_best_params = tune_nuisance(
                y,
                x,
                d,
                clone(learner_M),
                clone(learner_t),
                clone(learner_m),
                smpls,
                n_folds_tune,
                par_grid["ml_M"],
                par_grid["ml_t"],
                par_grid["ml_m"],
            )

    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_best_params, g1_best_params, pi_best_params, m_best_params = tune_nuisance(
                y,
                x,
                d,
                clone(learner_M),
                clone(learner_t),
                clone(learner_m),
                xx,
                n_folds_tune,
                par_grid["ml_M"],
                par_grid["ml_t"],
                par_grid["ml_m"],
            )


    M_best_params = M_best_params * n_folds
    t_best_params = t_best_params * n_folds
    m_best_params = m_best_params * n_folds

    np.random.seed(42)
    res_manual = fit_selection(
        y,
        x,
        d,
        clone(learner_M),
        clone(learner_t),
        clone(learner_m),
        all_smpls,
        score,
        M_params=M_best_params,
        t_params=t_best_params,
        m_params=m_best_params,
    )

    res_dict = {
        "coef": dml_sel_obj.coef[0],
        "coef_manual": res_manual["theta"],
        "se": dml_sel_obj.se[0],
        "se_manual": res_manual["se"],
    }

    return res_dict


@pytest.mark.ci
def test_dml_ssm_coef(dml_ssm_fixture):
    assert math.isclose(dml_lplr_fixture["coef"], dml_lplr_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_ssm_se(dml_ssm_fixture):
    assert math.isclose(dml_lplr_fixture["se"], dml_lplr_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
