import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data

from ._utils import draw_smpls
from ._utils_did_manual import fit_did, boot_did


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['PA-1', 'PA-2', 'DR'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_did_fixture(generate_data_did, learner, score, dml_procedure, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d) = generate_data_did

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    np.random.seed(3141)
    dml_did_obj = dml.DoubleMLDID(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  draw_sample_splitting=False,
                                  trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_did_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_did_obj.fit()

    np.random.seed(3141)
    res_manual = fit_did(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         all_smpls, dml_procedure, score,
                         trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_did_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_did_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_did(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                                           res_manual['all_m_hat'], res_manual['all_p_hat'],
                                           all_smpls, score, bootstrap, n_rep_boot,
                                           dml_procedure=dml_procedure)

        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_did_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_did_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_did_coef(dml_did_fixture):
    assert math.isclose(dml_did_fixture['coef'],
                        dml_did_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    

@pytest.mark.ci
def test_dml_did_se(dml_did_fixture):
    assert math.isclose(dml_did_fixture['se'],
                        dml_did_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    

@pytest.mark.ci
def test_dml_did_boot(dml_did_fixture):
    for bootstrap in dml_did_fixture['boot_methods']:
        assert np.allclose(dml_did_fixture['boot_coef' + bootstrap],
                           dml_did_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_did_fixture['boot_t_stat' + bootstrap],
                           dml_did_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)