import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import doubleml as dml
from doubleml.datasets import fetch_bonus

from ._utils import draw_smpls
from ._utils_plr_manual import fit_plr, boot_plr

bonus_data = fetch_bonus()


@pytest.fixture(scope='module',
                params=[Lasso(),
                        RandomForestClassifier(max_depth=2, n_estimators=10),
                        LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_binary_classifier_fixture(learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # Set machine learning methods for m & g
    ml_g = Lasso()
    ml_m = clone(learner)

    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(bonus_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = bonus_data.y
    x = bonus_data.x
    d = bonus_data.d
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_plr(y, x, d, clone(ml_g), clone(ml_m),
                         all_smpls, dml_procedure, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat'], res_manual['all_m_hat'],
                                           all_smpls, score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_binary_classifier_coef(dml_plr_binary_classifier_fixture):
    assert math.isclose(dml_plr_binary_classifier_fixture['coef'],
                        dml_plr_binary_classifier_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_binary_classifier_se(dml_plr_binary_classifier_fixture):
    assert math.isclose(dml_plr_binary_classifier_fixture['se'],
                        dml_plr_binary_classifier_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_binary_classifier_boot(dml_plr_binary_classifier_fixture):
    for bootstrap in dml_plr_binary_classifier_fixture['boot_methods']:
        assert np.allclose(dml_plr_binary_classifier_fixture['boot_coef' + bootstrap],
                           dml_plr_binary_classifier_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_binary_classifier_fixture['boot_t_stat' + bootstrap],
                           dml_plr_binary_classifier_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
