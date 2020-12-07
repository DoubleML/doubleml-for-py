import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_pliv_manual import pliv_dml1, pliv_dml2, fit_nuisance_pliv, boot_pliv, tune_nuisance_pliv


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_g(request):
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
                params=['dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == RandomForestRegressor:
        par_grid = {'n_estimators': [5, 10, 20]}
    elif learner.__class__ == ElasticNet:
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope='module')
def dml_pliv_fixture(generate_data_iv, idx, learner_g, learner_m, learner_r, score, dml_procedure, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m),
                'ml_r': get_par_grid(learner_r)}
    n_folds_tune = 4

    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    data = generate_data_iv[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for g, m & r
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)
    ml_r = clone(learner_r)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, 'Z1')
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)

    # tune hyperparameters
    res_tuning = dml_pliv_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    dml_pliv_obj.fit()
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    z = data['Z1'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]

    if tune_on_folds:
        g_params, m_params, r_params = tune_nuisance_pliv(y, X, d, z,
                                                          clone(learner_m), clone(learner_g), clone(learner_r),
                                                          smpls, n_folds_tune,
                                                          par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'])

        g_hat, m_hat, r_hat = fit_nuisance_pliv(y, X, d, z,
                                                clone(learner_m), clone(learner_g), clone(learner_r),
                                                smpls,
                                                g_params, m_params, r_params)
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_params, m_params, r_params = tune_nuisance_pliv(y, X, d, z,
                                                          clone(learner_m), clone(learner_g), clone(learner_r),
                                                          xx, n_folds_tune,
                                                          par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'])

        g_hat, m_hat, r_hat = fit_nuisance_pliv(y, X, d, z,
                                                clone(learner_m), clone(learner_g), clone(learner_r),
                                                smpls,
                                                g_params * n_folds, m_params * n_folds, r_params * n_folds)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = pliv_dml1(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, score)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = pliv_dml2(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, score)
    
    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual,
                'se': dml_pliv_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_pliv(res_manual,
                                            y, d,
                                            z,
                                            g_hat, m_hat, r_hat,
                                            smpls, score,
                                            se_manual,
                                            bootstrap, n_rep_boot,
                                            dml_procedure)
        
        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_pliv_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat
    
    return res_dict


@pytest.mark.ci
def test_dml_pliv_coef(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['coef'],
                        dml_pliv_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_se(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['se'],
                        dml_pliv_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_boot(dml_pliv_fixture):
    for bootstrap in dml_pliv_fixture['boot_methods']:
        assert np.allclose(dml_pliv_fixture['boot_coef' + bootstrap],
                           dml_pliv_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_pliv_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

