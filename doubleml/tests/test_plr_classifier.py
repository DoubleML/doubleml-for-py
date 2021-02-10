import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone, is_classifier

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import doubleml as dml
from doubleml.datasets import fetch_bonus

from ._utils_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr, fit_nuisance_plr_classifier

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
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(x)]

    if is_classifier(ml_m):
        g_hat, m_hat = fit_nuisance_plr_classifier(y, x, d,
                                                   clone(ml_m), clone(ml_g), smpls)
    else:
        g_hat, m_hat = fit_nuisance_plr(y, x, d,
                                        clone(ml_m), clone(ml_g), smpls)

    if dml_procedure == 'dml1':
        res_manual, se_manual = plr_dml1(y, x, d,
                                         g_hat, m_hat,
                                         smpls, score)
    else:
        assert dml_procedure == 'dml2'
        res_manual, se_manual = plr_dml2(y, x, d,
                                         g_hat, m_hat,
                                         smpls, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(res_manual,
                                           y, d,
                                           g_hat, m_hat,
                                           smpls, score,
                                           se_manual,
                                           bootstrap, n_rep_boot,
                                           dml_procedure)

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
