import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import Lasso

import doubleml as dml

from ._utils_pliv_partial_x_manual import pliv_partial_x_dml1, pliv_partial_x_dml2, \
    fit_nuisance_pliv_partial_x, boot_pliv_partial_x


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_partial_x_fixture(generate_data_pliv_partialX, learner, score, dml_procedure):
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    obj_dml_data = generate_data_pliv_partialX

    # Set machine learning methods for g, m & r
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV._partialX(obj_dml_data,
                                              ml_g, ml_m, ml_r,
                                              n_folds,
                                              dml_procedure=dml_procedure)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    z = obj_dml_data.z
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(x)]

    g_hat, r_hat, r_hat_tilde = fit_nuisance_pliv_partial_x(y, x, d, z,
                                                            clone(learner), clone(learner), clone(learner),
                                                            smpls)

    if dml_procedure == 'dml1':
        res_manual, se_manual = pliv_partial_x_dml1(y, x, d,
                                                    z,
                                                    g_hat, r_hat, r_hat_tilde,
                                                    smpls, score)
    else:
        assert dml_procedure == 'dml2'
        res_manual, se_manual = pliv_partial_x_dml2(y, x, d,
                                                    z,
                                                    g_hat, r_hat, r_hat_tilde,
                                                    smpls, score)

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual,
                'se': dml_pliv_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_pliv_partial_x(res_manual,
                                                      y, d,
                                                      z,
                                                      g_hat, r_hat, r_hat_tilde,
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
        assert np.allclose(dml_pliv_partial_x_fixture['boot_coef' + bootstrap],
                           dml_pliv_partial_x_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_pliv_partial_x_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_partial_x_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
