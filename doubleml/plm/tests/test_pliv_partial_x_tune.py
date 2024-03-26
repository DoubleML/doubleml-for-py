import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_pliv_partial_x_manual import fit_pliv_partial_x, boot_pliv_partial_x, tune_nuisance_pliv_partial_x


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_l(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_r(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == RandomForestRegressor:
        par_grid = {'n_estimators': [5, 10, 20]}
    else:
        assert learner.__class__ == ElasticNet
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope='module')
def dml_pliv_partial_x_fixture(generate_data_pliv_partialX, learner_l, learner_m, learner_r, score,
                               tune_on_folds):
    par_grid = {'ml_l': get_par_grid(learner_l),
                'ml_m': get_par_grid(learner_m),
                'ml_r': get_par_grid(learner_r)}
    n_folds_tune = 4

    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    obj_dml_data = generate_data_pliv_partialX

    # Set machine learning methods for l, m & r
    ml_l = clone(learner_l)
    ml_m = clone(learner_m)
    ml_r = clone(learner_r)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV._partialX(obj_dml_data,
                                              ml_l, ml_m, ml_r,
                                              n_folds=n_folds)

    # tune hyperparameters
    _ = dml_pliv_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    z = obj_dml_data.z
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)
    smpls = all_smpls[0]

    if tune_on_folds:
        l_params, m_params, r_params = tune_nuisance_pliv_partial_x(y, x, d, z,
                                                                    clone(learner_l),
                                                                    clone(learner_m),
                                                                    clone(learner_r),
                                                                    smpls, n_folds_tune,
                                                                    par_grid['ml_l'],
                                                                    par_grid['ml_m'],
                                                                    par_grid['ml_r'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        l_params, m_params, r_params = tune_nuisance_pliv_partial_x(y, x, d, z,
                                                                    clone(learner_l),
                                                                    clone(learner_m),
                                                                    clone(learner_r),
                                                                    xx, n_folds_tune,
                                                                    par_grid['ml_l'],
                                                                    par_grid['ml_m'],
                                                                    par_grid['ml_r'])
        l_params = l_params * n_folds
        m_params = [xx * n_folds for xx in m_params]
        r_params = r_params * n_folds

    res_manual = fit_pliv_partial_x(y, x, d, z,
                                    clone(learner_l), clone(learner_m), clone(learner_r),
                                    all_smpls, score,
                                    l_params=l_params, m_params=m_params, r_params=r_params)

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_pliv_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_pliv_partial_x(y, d, z, res_manual['thetas'], res_manual['ses'],
                                          res_manual['all_l_hat'], res_manual['all_m_hat'],
                                          res_manual['all_r_hat'],
                                          all_smpls, score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


def test_dml_pliv_coef(dml_pliv_partial_x_fixture):
    assert math.isclose(dml_pliv_partial_x_fixture['coef'],
                        dml_pliv_partial_x_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_se(dml_pliv_partial_x_fixture):
    assert math.isclose(dml_pliv_partial_x_fixture['se'],
                        dml_pliv_partial_x_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_partial_x_fixture):
    for bootstrap in dml_pliv_partial_x_fixture['boot_methods']:
        assert np.allclose(dml_pliv_partial_x_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_partial_x_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
