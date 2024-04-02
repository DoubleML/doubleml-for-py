import numpy as np
import pytest
import math

from sklearn.base import clone

# TODO: Maybe add some learner which cannot handle missings in x and test the exception
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_irm_manual import fit_irm, boot_irm


@pytest.fixture(scope='module',
                params=[[XGBRegressor(n_jobs=1, objective="reg:squarederror",
                                      eta=0.1, n_estimators=10),
                         XGBClassifier(use_label_encoder=False, n_jobs=1,
                                       objective="binary:logistic", eval_metric="logloss",
                                       eta=0.1, n_estimators=10)]])
def learner_xgboost(request):
    return request.param


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)]])
def learner_sklearn(request):
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
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_w_missing_fixture(generate_data_irm_w_missings, learner_xgboost, score,
                              normalize_ipw, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d) = generate_data_irm_w_missings

    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    # Set machine learning methods for m & g
    ml_g = clone(learner_xgboost[0])
    ml_m = clone(learner_xgboost[1])

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d,
                                                force_all_x_finite='allow-nan')
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  normalize_ipw=normalize_ipw,
                                  trimming_threshold=trimming_threshold)
    # synchronize the sample splitting
    dml_irm_obj.set_sample_splitting(all_smpls=all_smpls)
    np.random.seed(3141)
    dml_irm_obj.fit()

    np.random.seed(3141)
    res_manual = fit_irm(y, x, d,
                         clone(learner_xgboost[0]), clone(learner_xgboost[1]),
                         all_smpls, score,
                         normalize_ipw=normalize_ipw,
                         trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_irm_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_irm(y, d, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                               res_manual['all_m_hat'], res_manual['all_p_hat'],
                               all_smpls, score, bootstrap, n_rep_boot,
                               normalize_ipw=normalize_ipw)

        np.random.seed(3141)
        dml_irm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_irm_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_irm_w_missing_coef(dml_irm_w_missing_fixture):
    assert math.isclose(dml_irm_w_missing_fixture['coef'],
                        dml_irm_w_missing_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_w_missing_se(dml_irm_w_missing_fixture):
    assert math.isclose(dml_irm_w_missing_fixture['se'],
                        dml_irm_w_missing_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_w_missing_boot(dml_irm_w_missing_fixture):
    for bootstrap in dml_irm_w_missing_fixture['boot_methods']:
        assert np.allclose(dml_irm_w_missing_fixture['boot_t_stat' + bootstrap],
                           dml_irm_w_missing_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


def test_irm_exception_with_missings(generate_data_irm_w_missings, learner_sklearn):
    # collect data
    (x, y, d) = generate_data_irm_w_missings

    # Set machine learning methods for m & g
    ml_g = clone(learner_sklearn[0])
    ml_m = clone(learner_sklearn[1])

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d,
                                                force_all_x_finite='allow-nan')
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m)

    msg = r"Input X contains NaN.\nLinearRegression does not accept missing values encoded as NaN natively."
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.fit()
