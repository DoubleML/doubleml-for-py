import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_irm_manual import fit_irm, boot_irm


@pytest.fixture(scope='module',
                params=[[RandomForestRegressor(max_depth=2, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1])
def n_folds(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_no_cross_fit_fixture(generate_data_irm, learner, score, normalize_ipw, n_folds):
    boot_methods = ['normal']
    n_rep_boot = 499
    dml_procedure = 'dml1'

    # collect data
    (x, y, d) = generate_data_irm
    n_obs = len(y)
    if n_folds == 1:
        all_smpls = [(np.arange(n_obs), np.arange(n_obs))]
    else:
        all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  normalize_ipw=normalize_ipw,
                                  apply_cross_fitting=False,
                                  draw_sample_splitting=False)
    # synchronize the sample splitting
    dml_irm_obj.set_sample_splitting(all_smpls=all_smpls)
    np.random.seed(3141)
    dml_irm_obj.fit()

    np.random.seed(3141)
    if n_folds == 1:
        smpls = all_smpls
    else:
        smpls = all_smpls[0]
        smpls = [smpls[0]]

    np.random.seed(3141)
    res_manual = fit_irm(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         [smpls], dml_procedure, score,
                         normalize_ipw=normalize_ipw)

    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_irm_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_irm(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                                           res_manual['all_m_hat'], res_manual['all_p_hat'],
                                           [smpls], score, bootstrap, n_rep_boot,
                                           dml_procedure=dml_procedure,
                                           normalize_ipw=normalize_ipw,
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