import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import Lasso, ElasticNet

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_plr_manual import plr_dml1, plr_dml2, boot_plr, tune_nuisance_plr, fit_nuisance_plr


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(),
                        ElasticNet()])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(),
                        ElasticNet()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params = [True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == Lasso:
        par_grid = {'alpha': np.linspace(0.05, .95, 7)}
    elif learner.__class__ == ElasticNet:
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data2, idx, learner_g, learner_m, score, dml_procedure, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # collect data
    obj_dml_data = generate_data2[idx]

    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    # tune hyperparameters
    res_tuning = dml_plr_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    # fit with tuned parameters
    dml_plr_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    X = obj_dml_data.x
    d = obj_dml_data.d
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]

    if tune_on_folds:
        g_params, m_params = tune_nuisance_plr(y, X, d,
                                               clone(learner_m), clone(learner_g), smpls, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])

        g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                        clone(learner_m), clone(learner_g), smpls,
                                        g_params, m_params)
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_params, m_params = tune_nuisance_plr(y, X, d,
                                               clone(learner_m), clone(learner_g), xx, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])

        g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                        clone(learner_m), clone(learner_g),
                                        smpls,
                                        g_params * n_folds, m_params * n_folds)


    if dml_procedure == 'dml1':
        res_manual, se_manual = plr_dml1(y, X, d,
                                         g_hat, m_hat,
                                         smpls, score)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = plr_dml2(y, X, d,
                                         g_hat, m_hat,
                                         smpls, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(res_manual,
                                           y, d,
                                           g_hat, m_hat,
                                           smpls, score,
                                           se_manual,
                                           bootstrap, n_rep_boot,
                                           dml_procedure)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
