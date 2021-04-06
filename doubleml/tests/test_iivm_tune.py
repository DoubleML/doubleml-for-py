import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ._utils_iivm_manual import iivm_dml1, iivm_dml2, fit_nuisance_iivm, boot_iivm, tune_nuisance_iivm


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
                params=[{'always_takers': True, 'never_takers': True},
                        {'always_takers': False, 'never_takers': False}])
def subgroups(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestRegressor, RandomForestClassifier]:
        par_grid = {'n_estimators': [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression]
        par_grid = {'C': np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope="module")
def dml_iivm_fixture(generate_data_iivm, learner_g, learner_m, learner_r, score, dml_procedure, subgroups,
                     tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m),
                'ml_r': get_par_grid(learner_r)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 491

    # collect data
    data = generate_data_iivm
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m, g & r
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)
    ml_r = clone(learner_r)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols, 'z')
    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    subgroups=subgroups,
                                    dml_procedure=dml_procedure)
    # tune hyperparameters
    _ = dml_iivm_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    dml_iivm_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    z = data['z'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(x)]

    if tune_on_folds:
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, x, d, z,
                               clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'],
                               always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])

        g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = \
            fit_nuisance_iivm(y, x, d, z,
                              clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                              g0_params, g1_params, m_params,  r0_params, r1_params,
                              always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])
    else:
        xx = [(np.arange(data.shape[0]), np.array([]))]
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, x, d, z,
                               clone(learner_m), clone(learner_g), clone(learner_r), xx,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'],
                               always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])
        if subgroups['always_takers']:
            r0_params_rep = r0_params * n_folds
        else:
            r0_params_rep = r0_params
        if subgroups['never_takers']:
            r1_params_rep = r1_params * n_folds
        else:
            r1_params_rep = r1_params

        g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = \
            fit_nuisance_iivm(y, x, d, z,
                              clone(learner_m), clone(learner_g), clone(learner_r), smpls,
                              g0_params * n_folds, g1_params * n_folds, m_params * n_folds,
                              r0_params_rep, r1_params_rep,
                              always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])

    if dml_procedure == 'dml1':
        res_manual, se_manual = iivm_dml1(y, x, d, z,
                                          g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                          smpls, score)
    else:
        assert dml_procedure == 'dml2'
        res_manual, se_manual = iivm_dml2(y, x, d, z,
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
