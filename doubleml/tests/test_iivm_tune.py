import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ._utils import draw_smpls
from ._utils_iivm_manual import fit_iivm, boot_iivm, tune_nuisance_iivm


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
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)
    smpls = all_smpls[0]

    if tune_on_folds:
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, x, d, z,
                               clone(learner_g), clone(learner_m), clone(learner_r), smpls,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'],
                               always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])
    else:
        xx = [(np.arange(data.shape[0]), np.array([]))]
        g0_params, g1_params, m_params,  r0_params, r1_params = \
            tune_nuisance_iivm(y, x, d, z,
                               clone(learner_g), clone(learner_m), clone(learner_r), xx,
                               n_folds_tune,
                               par_grid['ml_g'], par_grid['ml_m'], par_grid['ml_r'],
                               always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])
        g0_params = g0_params * n_folds
        g1_params = g1_params * n_folds
        m_params = m_params * n_folds
        if subgroups['always_takers']:
            r0_params = r0_params * n_folds
        else:
            r0_params = r0_params
        if subgroups['never_takers']:
            r1_params = r1_params * n_folds
        else:
            r1_params = r1_params

    res_manual = fit_iivm(y, x, d, z,
                          clone(learner_g), clone(learner_m), clone(learner_r),
                          all_smpls, dml_procedure, score,
                          g0_params=g0_params, g1_params=g1_params,
                          m_params=m_params, r0_params=r0_params, r1_params=r1_params,
                          always_takers=subgroups['always_takers'], never_takers=subgroups['never_takers'])

    res_dict = {'coef': dml_iivm_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_iivm_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_iivm(y, d, z, res_manual['thetas'], res_manual['ses'],
                                            res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                                            res_manual['all_m_hat'], res_manual['all_r_hat0'], res_manual['all_r_hat1'],
                                            all_smpls, score, bootstrap, n_rep_boot)

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
