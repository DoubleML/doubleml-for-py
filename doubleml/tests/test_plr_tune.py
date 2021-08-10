import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import Lasso, ElasticNet

import doubleml as dml

from ._utils import draw_smpls
from ._utils_plr_manual import fit_plr, boot_plr, tune_nuisance_plr


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
                params=['dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == Lasso:
        par_grid = {'alpha': np.linspace(0.05, .95, 7)}
    else:
        assert learner.__class__ == ElasticNet
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data2, learner_g, learner_m, score, dml_procedure, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # collect data
    obj_dml_data = generate_data2

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
    _ = dml_plr_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    # fit with tuned parameters
    dml_plr_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)
    smpls = all_smpls[0]

    if tune_on_folds:
        g_params, m_params = tune_nuisance_plr(y, x, d,
                                               clone(learner_g), clone(learner_m), smpls, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_params, m_params = tune_nuisance_plr(y, x, d,
                                               clone(learner_g), clone(learner_m), xx, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])
        g_params = g_params * n_folds
        m_params = m_params * n_folds

    res_manual = fit_plr(y, x, d, clone(learner_g), clone(learner_m),
                         all_smpls, dml_procedure, score,
                         g_params=g_params, m_params=m_params)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat'], res_manual['all_m_hat'],
                                           all_smpls, score, bootstrap, n_rep_boot)

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
