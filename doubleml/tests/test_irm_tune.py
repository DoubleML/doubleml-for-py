import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ._utils_irm_manual import irm_dml1, irm_dml2, fit_nuisance_irm, boot_irm, tune_nuisance_irm


@pytest.fixture(scope='module',
                params=[RandomForestRegressor()])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml2'])
def dml_procedure(request):
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
        par_grid = {'C': np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope='module')
def dml_irm_fixture(generate_data_irm, learner_g, learner_m, score, dml_procedure, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d) = generate_data_irm

    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    # tune hyperparameters
    _ = dml_irm_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    dml_irm_obj.fit()

    np.random.seed(3141)
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(x)]

    if tune_on_folds:
        g0_params, g1_params, m_params = tune_nuisance_irm(y, x, d,
                                                           clone(learner_m), clone(learner_g), smpls, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])

        g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                        clone(learner_m), clone(learner_g), smpls,
                                                        score,
                                                        g0_params, g1_params, m_params)
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_params, g1_params, m_params = tune_nuisance_irm(y, x, d,
                                                           clone(learner_m), clone(learner_g), xx, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])
        if score == 'ATE':
            g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                            clone(learner_m), clone(learner_g), smpls,
                                                            score,
                                                            g0_params * n_folds, g1_params * n_folds, m_params * n_folds)
        else:
            assert score == 'ATTE'
            g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                            clone(learner_m), clone(learner_g), smpls,
                                                            score,
                                                            g0_params * n_folds, None, m_params * n_folds)

    if dml_procedure == 'dml1':
        res_manual, se_manual = irm_dml1(y, x, d,
                                         g_hat0, g_hat1, m_hat, p_hat,
                                         smpls, score)
    else:
        assert dml_procedure == 'dml2'
        res_manual, se_manual = irm_dml2(y, x, d,
                                         g_hat0, g_hat1, m_hat, p_hat,
                                         smpls, score)

    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual,
                'se': dml_irm_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_irm(res_manual,
                                           y, d,
                                           g_hat0, g_hat1, m_hat, p_hat,
                                           smpls, score,
                                           se_manual,
                                           bootstrap, n_rep_boot,
                                           dml_procedure)

        np.random.seed(3141)
        dml_irm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_irm_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_irm_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_irm_coef(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['coef'],
                        dml_irm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['se'],
                        dml_irm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_boot(dml_irm_fixture):
    for bootstrap in dml_irm_fixture['boot_methods']:
        assert np.allclose(dml_irm_fixture['boot_coef' + bootstrap],
                           dml_irm_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_fixture['boot_t_stat' + bootstrap],
                           dml_irm_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
