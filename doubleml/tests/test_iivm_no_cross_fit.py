import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ._utils import draw_smpls
from ._utils_iivm_manual import fit_iivm, boot_iivm


@pytest.fixture(scope='module',
                params=[[RandomForestRegressor(max_depth=2, n_estimators=10),
                         RandomForestClassifier(max_depth=2, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['LATE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 2])
def n_folds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_iivm_no_cross_fit_fixture(generate_data_iivm, learner, score, n_folds):
    boot_methods = ['normal']
    n_rep_boot = 491
    dml_procedure = 'dml1'

    # collect data
    data = generate_data_iivm
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])
    ml_r = clone(learner[1])

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols, 'z')
    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure,
                                    apply_cross_fitting=False)

    dml_iivm_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    z = data['z'].values
    if n_folds == 1:
        smpls = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        n_obs = len(y)
        all_smpls = draw_smpls(n_obs, n_folds)
        smpls = all_smpls[0]
        smpls = [smpls[0]]

    res_manual = fit_iivm(y, x, d, z,
                          clone(learner[0]), clone(learner[1]), clone(learner[1]),
                          [smpls], dml_procedure, score)

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
                                            [smpls], score, bootstrap, n_rep_boot,
                                            apply_cross_fitting=False)

        np.random.seed(3141)
        dml_iivm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_iivm_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_iivm_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_iivm_no_cross_fit_coef(dml_iivm_no_cross_fit_fixture):
    assert math.isclose(dml_iivm_no_cross_fit_fixture['coef'],
                        dml_iivm_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_no_cross_fit_se(dml_iivm_no_cross_fit_fixture):
    assert math.isclose(dml_iivm_no_cross_fit_fixture['se'],
                        dml_iivm_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_no_cross_fit_boot(dml_iivm_no_cross_fit_fixture):
    for bootstrap in dml_iivm_no_cross_fit_fixture['boot_methods']:
        assert np.allclose(dml_iivm_no_cross_fit_fixture['boot_coef' + bootstrap],
                           dml_iivm_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_iivm_no_cross_fit_fixture['boot_t_stat' + bootstrap],
                           dml_iivm_no_cross_fit_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
