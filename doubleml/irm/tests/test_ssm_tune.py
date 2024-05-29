import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_ssm_manual import fit_selection, tune_nuisance_ssm


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(random_state=42)])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression(random_state=42)])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=['missing-at-random', 'nonignorable'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestRegressor]:
        par_grid = {'n_estimators': [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression]
        par_grid = {'C': np.logspace(-2, 2, 10)}
    return par_grid


@pytest.fixture(scope='module')
def dml_ssm_fixture(generate_data_selection_mar, generate_data_selection_nonignorable,
                    learner_g, learner_m, score,
                    normalize_ipw, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_pi': get_par_grid(learner_m),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4
    n_folds = 2

    # collect data
    np.random.seed(42)
    if score == 'missing-at-random':
        (x, y, d, z, s) = generate_data_selection_mar
    else:
        (x, y, d, z, s) = generate_data_selection_nonignorable

    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    ml_g = clone(learner_g)
    ml_pi = clone(learner_m)
    ml_m = clone(learner_m)

    np.random.seed(42)
    if score == 'missing-at-random':
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=None, s=s)
        dml_sel_obj = dml.DoubleMLSSM(obj_dml_data,
                                      ml_g, ml_pi, ml_m,
                                      n_folds=n_folds,
                                      score=score,
                                      normalize_ipw=normalize_ipw,
                                      draw_sample_splitting=False)
    else:
        assert score == 'nonignorable'
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=z, s=s)
        dml_sel_obj = dml.DoubleMLSSM(obj_dml_data,
                                      ml_g, ml_pi, ml_m,
                                      n_folds=n_folds,
                                      score=score,
                                      normalize_ipw=normalize_ipw,
                                      draw_sample_splitting=False)

    # synchronize the sample splitting
    np.random.seed(42)
    dml_sel_obj.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(42)
    # tune hyperparameters
    tune_res = dml_sel_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLSSM)

    dml_sel_obj.fit()

    np.random.seed(42)
    smpls = all_smpls[0]
    if tune_on_folds:
        g0_best_params, g1_best_params, pi_best_params, m_best_params = tune_nuisance_ssm(
            y, x, d, z, s,
            clone(learner_g), clone(learner_m), clone(learner_m),
            smpls, score, n_folds_tune,
            par_grid['ml_g'], par_grid['ml_pi'], par_grid['ml_m'])

    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_best_params, g1_best_params, pi_best_params, m_best_params = tune_nuisance_ssm(
            y, x, d, z, s,
            clone(learner_g), clone(learner_m), clone(learner_m),
            xx, score, n_folds_tune,
            par_grid['ml_g'], par_grid['ml_pi'], par_grid['ml_m'])

        g0_best_params = g0_best_params * n_folds
        g1_best_params = g1_best_params * n_folds
        pi_best_params = pi_best_params * n_folds
        m_best_params = m_best_params * n_folds

    np.random.seed(42)
    res_manual = fit_selection(y, x, d, z, s,
                               clone(learner_g), clone(learner_m), clone(learner_m),
                               all_smpls, score,
                               normalize_ipw=normalize_ipw,
                               g_d0_params=g0_best_params, g_d1_params=g1_best_params,
                               pi_params=pi_best_params, m_params=m_best_params)

    res_dict = {'coef': dml_sel_obj.coef[0],
                'coef_manual': res_manual['theta'],
                'se': dml_sel_obj.se[0],
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_ssm_coef(dml_ssm_fixture):
    assert math.isclose(dml_ssm_fixture['coef'],
                        dml_ssm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_ssm_se(dml_ssm_fixture):
    assert math.isclose(dml_ssm_fixture['se'],
                        dml_ssm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
