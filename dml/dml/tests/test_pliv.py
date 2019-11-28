import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_pliv import DoubleMLPLIV

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_pliv_manual import pliv_dml1, pliv_dml2, fit_nuisance_pliv, boot_pliv


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
                params = ['DML2018'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_fixture(generate_data_iv, idx, learner, inf_model, dml_procedure):
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner),
                   'ml_r': clone(learner)}
    
    dml_pliv_obj = DoubleMLPLIV(n_folds,
                                ml_learners,
                                dml_procedure,
                                inf_model)
    data = generate_data_iv[idx]
    np.random.seed(3141)
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'], 'z')
    dml_pliv_obj.fit(obj_dml_data)
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    z = data['z'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat, m_hat, r_hat = fit_nuisance_pliv(y, X, d, z,
                                            clone(learner), clone(learner), clone(learner),
                                            smpls)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = pliv_dml1(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = pliv_dml2(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, inf_model)
    
    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual,
                'se': dml_pliv_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = boot_pliv(res_manual,
                               y, d,
                               z,
                               g_hat, m_hat, r_hat,
                               smpls, inf_model,
                               se_manual,
                               bootstrap, 500,
                               dml_procedure)
        
        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method = bootstrap, n_rep=500)
        res_dict['boot_coef' + bootstrap] = dml_pliv_obj.boot_coef
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_pliv_coef(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['coef'],
                        dml_pliv_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_se(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['se'],
                        dml_pliv_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_fixture):
    for bootstrap in dml_pliv_fixture['boot_methods']:
        assert np.allclose(dml_pliv_fixture['boot_coef' + bootstrap],
                           dml_pliv_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

