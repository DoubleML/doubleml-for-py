import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ._utils_irm_manual import irm_dml1, fit_nuisance_irm, boot_irm


@pytest.fixture(scope='module',
                params=[[RandomForestClassifier(max_depth=2, n_estimators=10),
                         RandomForestRegressor(max_depth=2, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 2])
def n_folds(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_no_cross_fit_fixture(generate_data_irm, learner, score, n_folds):
    boot_methods = ['normal']
    n_rep_boot = 499
    dml_procedure = 'dml1'

    # collect data
    (x, y, d) = generate_data_irm

    # Set machine learning methods for m & g
    ml_g = clone(learner[1])
    ml_m = clone(learner[0])

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  apply_cross_fitting=False)

    dml_irm_obj.fit()

    np.random.seed(3141)
    if n_folds == 1:
        smpls = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        resampling = KFold(n_splits=n_folds,
                           shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(x)]
        smpls = [smpls[0]]

    g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                    clone(learner[0]), clone(learner[1]), smpls,
                                                    score)

    assert dml_procedure == 'dml1'
    res_manual, se_manual = irm_dml1(y, x, d,
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
                                           apply_cross_fitting=False)

        np.random.seed(3141)
        dml_irm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_irm_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_irm_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_irm_no_cross_fit_coef(dml_irm_no_cross_fit_fixture):
    assert math.isclose(dml_irm_no_cross_fit_fixture['coef'],
                        dml_irm_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_no_cross_fit_se(dml_irm_no_cross_fit_fixture):
    assert math.isclose(dml_irm_no_cross_fit_fixture['se'],
                        dml_irm_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_no_cross_fit_boot(dml_irm_no_cross_fit_fixture):
    for bootstrap in dml_irm_no_cross_fit_fixture['boot_methods']:
        assert np.allclose(dml_irm_no_cross_fit_fixture['boot_coef' + bootstrap],
                           dml_irm_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_no_cross_fit_fixture['boot_t_stat' + bootstrap],
                           dml_irm_no_cross_fit_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
