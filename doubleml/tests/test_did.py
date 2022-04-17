import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ._utils import draw_smpls
from ._utils_did_ro_manual import fit_did_ro, boot_did_ro


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=2, n_estimators=10),
                         RandomForestClassifier(max_depth=2, n_estimators=10)]])
def learner(request):
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
def dml_did_ro_fixture(generate_data_did_ro, learner, dml_procedure, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y0, y1, d) = generate_data_did_ro

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    obj_dml_data = dml.DiffInDiffRODoubleMLData.from_arrays(x, y0, y1, d)
    dml_did_obj = dml.DoubleMLDiD(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score="ortho_ro",
                                  dml_procedure=dml_procedure,
                                  trimming_threshold=trimming_threshold)

    dml_did_obj.fit()

    np.random.seed(3141)
    n_obs = len(y0)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_did_ro(y0, y1, x, d,
                            clone(learner[0]), clone(learner[1]),
                            all_smpls, dml_procedure, trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_did_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_did_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_did_ro(y0, y1, d, res_manual['thetas'], res_manual['ses'],
                                              res_manual['all_g_hat'], res_manual['all_m_hat'], res_manual['all_p_hat'],
                                              all_smpls, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_did_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_did_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_did_coef(dml_did_ro_fixture):
    assert math.isclose(dml_did_ro_fixture['coef'],
                        dml_did_ro_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_ro_se(dml_did_ro_fixture):
    assert math.isclose(dml_did_ro_fixture['se'],
                        dml_did_ro_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_ro_boot(dml_did_ro_fixture):
    for bootstrap in dml_did_ro_fixture['boot_methods']:
        assert np.allclose(dml_did_ro_fixture['boot_coef' + bootstrap],
                           dml_did_ro_fixture['boot_coef' +
                                              bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_did_ro_fixture['boot_t_stat' + bootstrap],
                           dml_did_ro_fixture['boot_t_stat' +
                                              bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
