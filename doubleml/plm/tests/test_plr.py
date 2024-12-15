import pytest
import math
import scipy
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_plr_manual import fit_plr, plr_dml2, boot_plr, fit_sensitivity_elements_plr


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression(),
                        Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data1, learner, score):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    if score == 'partialling out':
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds,
                                      score=score)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_plr(y, x, d, clone(learner), clone(learner), clone(learner),
                         all_smpls, score)

    np.random.seed(3141)
    # test with external nuisance predictions
    if score == 'partialling out':
        dml_plr_obj_ext = dml.DoubleMLPLR(obj_dml_data,
                                          ml_l, ml_m,
                                          n_folds,
                                          score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj_ext = dml.DoubleMLPLR(obj_dml_data,
                                          ml_l, ml_m, ml_g,
                                          n_folds,
                                          score=score)

    # synchronize the sample splitting
    dml_plr_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    if score == 'partialling out':
        prediction_dict = {'d': {'ml_l': dml_plr_obj.predictions['ml_l'].reshape(-1, 1),
                                 'ml_m': dml_plr_obj.predictions['ml_m'].reshape(-1, 1)}}
    else:
        assert score == 'IV-type'
        prediction_dict = {'d': {'ml_l': dml_plr_obj.predictions['ml_l'].reshape(-1, 1),
                                 'ml_m': dml_plr_obj.predictions['ml_m'].reshape(-1, 1),
                                 'ml_g': dml_plr_obj.predictions['ml_g'].reshape(-1, 1)}}

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
                               all_smpls, score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_plr_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)
        res_dict['boot_t_stat' + bootstrap + '_ext'] = dml_plr_obj_ext.boot_t_stat

    # sensitivity tests
    res_dict['sensitivity_elements'] = dml_plr_obj.sensitivity_elements
    res_dict['sensitivity_elements_manual'] = fit_sensitivity_elements_plr(y, d.reshape(-1, 1),
                                                                           all_coef=dml_plr_obj.all_coef,
                                                                           predictions=dml_plr_obj.predictions,
                                                                           score=score,
                                                                           n_rep=1)
    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_plr_obj.sensitivity_analysis(rho=0.0)
    res_dict['sensitivity_ses'] = dml_plr_obj.sensitivity_params['se']
    return res_dict


@pytest.mark.ci
def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_ext'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_ext'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_ext'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_plr_sensitivity(dml_plr_fixture):
    sensitivity_element_names = ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2']
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(dml_plr_fixture['sensitivity_elements'][sensitivity_element],
                           dml_plr_fixture['sensitivity_elements_manual'][sensitivity_element])


@pytest.mark.ci
def test_dml_plr_sensitivity_rho0(dml_plr_fixture):
    assert np.allclose(dml_plr_fixture['se'],
                       dml_plr_fixture['sensitivity_ses']['lower'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_fixture['se'],
                       dml_plr_fixture['sensitivity_ses']['upper'],
                       rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_ols_manual_fixture(generate_data1, score):
    learner = LinearRegression()
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 501

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_g = clone(learner)
    ml_m = clone(learner)

    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    if score == 'partialling out':
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds,
                                      score=score)

    n = data.shape[0]
    this_smpl = list()
    xx = int(n/2)
    this_smpl.append((np.arange(xx, n), np.arange(0, xx)))
    this_smpl.append((np.arange(0, xx), np.arange(xx, n)))
    smpls = [this_smpl]
    dml_plr_obj.set_sample_splitting(smpls)

    dml_plr_obj.fit()

    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values

    # add column of ones for intercept
    o = np.ones((n, 1))
    x = np.append(x, o, axis=1)

    smpls = dml_plr_obj.smpls[0]

    l_hat = []
    l_hat_vec = np.full_like(y, np.nan)
    for (train_index, test_index) in smpls:
        ols_est = scipy.linalg.lstsq(x[train_index], y[train_index])[0]
        preds = np.dot(x[test_index], ols_est)
        l_hat.append(preds)
        l_hat_vec[test_index] = preds

    m_hat = []
    m_hat_vec = np.full_like(d, np.nan)
    for (train_index, test_index) in smpls:
        ols_est = scipy.linalg.lstsq(x[train_index], d[train_index])[0]
        preds = np.dot(x[test_index], ols_est)
        m_hat.append(preds)
        m_hat_vec[test_index] = preds

    g_hat = []
    if score == 'IV-type':
        theta_initial = scipy.linalg.lstsq((d - m_hat_vec).reshape(-1, 1), y - l_hat_vec)[0]
        for (train_index, test_index) in smpls:
            ols_est = scipy.linalg.lstsq(x[train_index],
                                         y[train_index] - d[train_index] * theta_initial)[0]
            g_hat.append(np.dot(x[test_index], ols_est))

    res_manual, se_manual = plr_dml2(y, x, d,
                                     l_hat, m_hat, g_hat,
                                     smpls, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_plr(y, d, [res_manual], [se_manual],
                               [l_hat], [m_hat], [g_hat],
                               [smpls], score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_plr_ols_manual_coef(dml_plr_ols_manual_fixture):
    assert math.isclose(dml_plr_ols_manual_fixture['coef'],
                        dml_plr_ols_manual_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_ols_manual_se(dml_plr_ols_manual_fixture):
    assert math.isclose(dml_plr_ols_manual_fixture['se'],
                        dml_plr_ols_manual_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_ols_manual_boot(dml_plr_ols_manual_fixture):
    for bootstrap in dml_plr_ols_manual_fixture['boot_methods']:
        assert np.allclose(dml_plr_ols_manual_fixture['boot_t_stat' + bootstrap],
                           dml_plr_ols_manual_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope='module',
                params=["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def cov_type(request):
    return request.param


@pytest.mark.ci
def test_dml_plr_cate_gate(score, cov_type):
    n = 9

    # collect data
    np.random.seed(42)
    obj_dml_data = dml.datasets.make_plr_CCDDHNR2018(n_obs=n)
    ml_l = LinearRegression()
    ml_g = LinearRegression()
    ml_m = LinearRegression()

    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m, ml_l,
                                  n_folds=2,
                                  score=score)
    dml_plr_obj.fit()
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 5)))
    cate = dml_plr_obj.cate(random_basis, cov_type=cov_type)
    assert isinstance(cate, dml.DoubleMLBLP)
    assert isinstance(cate.confint(), pd.DataFrame)
    assert cate.blp_model.cov_type == cov_type

    groups_1 = pd.DataFrame(
        np.column_stack([obj_dml_data.data['X1'] <= 0,
                         obj_dml_data.data['X1'] > 0.2]),
        columns=['Group 1', 'Group 2'])
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_1 = dml_plr_obj.gate(groups_1, cov_type=cov_type)
    assert isinstance(gate_1, dml.utils.blp.DoubleMLBLP)
    assert isinstance(gate_1.confint(), pd.DataFrame)
    assert all(gate_1.confint().index == groups_1.columns.tolist())
    assert gate_1.blp_model.cov_type == cov_type

    np.random.seed(42)
    groups_2 = pd.DataFrame(np.random.choice(["1", "2"], n))
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_2 = dml_plr_obj.gate(groups_2, cov_type=cov_type)
    assert isinstance(gate_2, dml.utils.blp.DoubleMLBLP)
    assert isinstance(gate_2.confint(), pd.DataFrame)
    assert all(gate_2.confint().index == ["Group_1", "Group_2"])
    assert gate_2.blp_model.cov_type == cov_type
