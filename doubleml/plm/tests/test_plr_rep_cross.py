import numpy as np
import pytest
import math

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls, _clone
from ._utils_plr_manual import fit_plr, boot_plr


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data1, learner, score, n_rep):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 498

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for l, m & g
    ml_l = _clone(learner)
    ml_m = _clone(learner)
    if score == 'IV-type':
        ml_g = _clone(learner)
    else:
        ml_g = None

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l, ml_m, ml_g,
                                  n_folds,
                                  n_rep,
                                  score)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep)

    res_manual = fit_plr(y, x, d, _clone(learner), _clone(learner), _clone(learner),
                         all_smpls, score, n_rep)

    np.random.seed(3141)
    # test with external nuisance predictions
    if score == 'partialling out':
        dml_plr_obj_ext = dml.DoubleMLPLR(obj_dml_data,
                                          ml_l, ml_m,
                                          n_folds,
                                          n_rep,
                                          score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj_ext = dml.DoubleMLPLR(obj_dml_data,
                                          ml_l, ml_m, ml_g,
                                          n_folds,
                                          n_rep,
                                          score=score)

    # synchronize the sample splitting
    dml_plr_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    if score == 'partialling out':
        prediction_dict = {'d': {'ml_l': dml_plr_obj.predictions['ml_l'].reshape(-1, n_rep),
                                 'ml_m': dml_plr_obj.predictions['ml_m'].reshape(-1, n_rep)}}
    else:
        assert score == 'IV-type'
        prediction_dict = {'d': {'ml_l': dml_plr_obj.predictions['ml_l'].reshape(-1, n_rep),
                                 'ml_m': dml_plr_obj.predictions['ml_m'].reshape(-1, n_rep),
                                 'ml_g': dml_plr_obj.predictions['ml_g'].reshape(-1, n_rep)}}

    dml_plr_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'coef_ext': dml_plr_obj_ext.coef,
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'se_ext': dml_plr_obj_ext.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_plr(y, d, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_l_hat'], res_manual['all_m_hat'], res_manual['all_g_hat'],
                               all_smpls, score, bootstrap, n_rep_boot, n_rep)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, n_rep)

    return res_dict


@pytest.mark.ci
def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
