import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_iivm_manual import iivm_dml1, iivm_dml2, fit_nuisance_iivm, boot_iivm, tune_nuisance_iivm


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param

@pytest.fixture(scope='module',
                params=[RandomForestRegressor()])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[RandomForestClassifier()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression()])
def learner_r(request):
    return request.param


@pytest.fixture(scope='module',
                params=['LATE'])
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
    if learner.__class__ in [RandomForestRegressor, RandomForestClassifier]:
        par_grid = {'n_estimators': [5, 10, 20]}
    elif learner.__class__ in [LogisticRegression]:
        par_grid = {'C': np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope="module")
def dml_iivm_fixture(generate_data_iivm, idx, learner_g, learner_m, learner_r, score, dml_procedure, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m),
                'ml_r': get_par_grid(learner_r)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 491

    # collect data
    data = generate_data_iivm[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m, g & r
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)
    ml_r = clone(learner_r)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, 'z')
    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)
    # tune hyperparameters
    res_tuning = dml_iivm_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    dml_iivm_obj.fit()
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    z = data['z'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]

    if tune_on_folds:
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, X, d, z,
                               clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'])

        g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = \
            fit_nuisance_iivm(y, X, d, z,
                              clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                              g0_params, g1_params, m_params,  r0_params, r1_params)
    else:
        xx = [(np.arange(data.shape[0]), np.array([]))]
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, X, d, z,
                               clone(learner_m), clone(learner_g), clone(learner_r), xx,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'])

        g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = \
            fit_nuisance_iivm(y, X, d, z,
                              clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                              g0_params * n_folds, g1_params * n_folds, m_params * n_folds,
                              r0_params * n_folds, r1_params * n_folds)

    if dml_procedure == 'dml1':
        res_manual, se_manual = iivm_dml1(y, X, d, z,
                                          g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                          smpls, score)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = iivm_dml2(y, X, d, z,
                                          g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                          smpls, score)
    
    res_dict = {'coef': dml_iivm_obj.coef,
                'coef_manual': res_manual,
                'se': dml_iivm_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_iivm(res_manual,
                                            y, d, z,
                                            g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                            smpls, score,
                                            se_manual,
                                            bootstrap, n_rep_boot,
                                            dml_procedure)
        
        np.random.seed(3141)
        dml_iivm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_iivm_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_iivm_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat
    
    return res_dict


@pytest.mark.ci
def test_dml_iivm_coef(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture['coef'],
                        dml_iivm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_se(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture['se'],
                        dml_iivm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_boot(dml_iivm_fixture):
    for bootstrap in dml_iivm_fixture['boot_methods']:
        assert np.allclose(dml_iivm_fixture['boot_coef' + bootstrap],
                           dml_iivm_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_iivm_fixture['boot_t_stat' + bootstrap],
                           dml_iivm_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

