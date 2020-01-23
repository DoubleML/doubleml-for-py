import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_plr import DoubleMLPLR

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = [RandomForestRegressor(max_depth=2, n_estimators=10),
                          LinearRegression(),
                          Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['IV-type', 'DML2018'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data1, idx, learner, inf_model, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}
    
    dml_plr_obj = DoubleMLPLR(n_folds,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    data = generate_data1[idx]
    np.random.seed(3141)
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'])
    dml_plr_obj.fit(obj_dml_data)
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                    clone(learner), clone(learner), smpls)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = plr_dml1(y, X, d,
                                         g_hat, m_hat,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = plr_dml2(y, X, d,
                                         g_hat, m_hat,
                                         smpls, inf_model)
    
    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = boot_plr(res_manual,
                              y, d,
                              g_hat, m_hat,
                              smpls, inf_model,
                              se_manual,
                              bootstrap, n_rep_boot,
                              dml_procedure)
        
        np.random.seed(3141)
        dml_plr_obj.bootstrap(method = bootstrap, n_rep=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_ols_manual_fixture(generate_data1, idx, inf_model, dml_procedure):
    learner = LinearRegression()
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 501
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}


    dml_plr_obj = DoubleMLPLR(n_folds,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    data = generate_data1[idx]
    N = data.shape[0]
    smpls = []
    xx = int(N/2)
    all_train = [np.arange(xx, N), np.arange(0, xx)]
    all_test = [np.arange(0, xx), np.arange(xx, N)]
    dml_plr_obj.depreciated_set_samples(all_train, all_test)
    
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'])
    dml_plr_obj.fit(obj_dml_data)
    
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    
    # add column of ones for intercept
    o = np.ones((N,1))
    X = np.append(X, o, axis=1)

    smpls = dml_plr_obj.smpls[0]
    
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        ols_est = scipy.linalg.lstsq(X[train_index], y[train_index])[0]
        g_hat.append(np.dot(X[test_index], ols_est))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        ols_est = scipy.linalg.lstsq(X[train_index], d[train_index])[0]
        m_hat.append(np.dot(X[test_index], ols_est))
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = plr_dml1(y, X, d,
                                         g_hat, m_hat,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = plr_dml2(y, X, d,
                                         g_hat, m_hat,
                                         smpls, inf_model)
    
    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = boot_plr(res_manual,
                              y, d,
                              g_hat, m_hat,
                              smpls, inf_model,
                              se_manual,
                              bootstrap, n_rep_boot,
                              dml_procedure)
        
        np.random.seed(3141)
        dml_plr_obj.bootstrap(method = bootstrap, n_rep=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_plr_ols_manual_coef(dml_plr_ols_manual_fixture):
    assert math.isclose(dml_plr_ols_manual_fixture['coef'],
                        dml_plr_ols_manual_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_ols_manual_se(dml_plr_ols_manual_fixture):
    assert math.isclose(dml_plr_ols_manual_fixture['se'],
                        dml_plr_ols_manual_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_ols_manual_boot(dml_plr_ols_manual_fixture):
    for bootstrap in dml_plr_ols_manual_fixture['boot_methods']:
        assert np.allclose(dml_plr_ols_manual_fixture['boot_coef' + bootstrap],
                           dml_plr_ols_manual_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

