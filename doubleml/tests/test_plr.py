import numpy as np
import pytest
import math
import scipy

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ._utils import draw_smpls
from ._utils_plr_manual import fit_plr, plr_dml1, plr_dml2, boot_plr


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


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data1, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_plr(y, x, d, clone(learner), clone(learner),
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
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_ols_manual_fixture(generate_data1, score, dml_procedure):
    learner = LinearRegression()
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 501

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

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

    g_hat = []
    for (train_index, test_index) in smpls:
        ols_est = scipy.linalg.lstsq(x[train_index], y[train_index])[0]
        g_hat.append(np.dot(x[test_index], ols_est))

    m_hat = []
    for (train_index, test_index) in smpls:
        ols_est = scipy.linalg.lstsq(x[train_index], d[train_index])[0]
        m_hat.append(np.dot(x[test_index], ols_est))

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
        boot_theta, boot_t_stat = boot_plr(y, d, [res_manual], [se_manual],
                                           [g_hat], [m_hat],
                                           [smpls], score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

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
        assert np.allclose(dml_plr_ols_manual_fixture['boot_coef' + bootstrap],
                           dml_plr_ols_manual_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_ols_manual_fixture['boot_t_stat' + bootstrap],
                           dml_plr_ols_manual_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
