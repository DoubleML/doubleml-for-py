import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_irm import DoubleMLIRM

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_irm_manual import irm_dml1, irm_dml2, fit_nuisance_irm, boot_irm


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = [[LogisticRegression(solver='lbfgs', max_iter=250),
                           LinearRegression()],
                          [RandomForestClassifier(max_depth=2, n_estimators=10),
                           RandomForestRegressor(max_depth=2, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['ATE', 'ATTE'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_fixture(generate_data_irm, idx, learner, inf_model, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2

    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner[0]),
                   'ml_g': clone(learner[1])}
    
    dml_irm_obj = DoubleMLIRM(n_folds,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    data = generate_data_irm[idx]
    np.random.seed(3141)
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'])
    dml_irm_obj.fit(obj_dml_data)
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, X, d,
                                                    clone(learner[0]), clone(learner[1]), smpls,
                                                    inf_model)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = irm_dml1(y, X, d,
                                         g_hat0, g_hat1, m_hat, p_hat,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = irm_dml2(y, X, d,
                                         g_hat0, g_hat1, m_hat, p_hat,
                                         smpls, inf_model)
    
    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual,
                'se': dml_irm_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = boot_irm(res_manual,
                              y, d,
                              g_hat0, g_hat1, m_hat, p_hat,
                              smpls, inf_model,
                              se_manual,
                              bootstrap, 500)
        
        np.random.seed(3141)
        dml_irm_obj.bootstrap(method = bootstrap, n_rep=500)
        res_dict['boot_coef' + bootstrap] = dml_irm_obj.boot_coef
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_irm_coef(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['coef'],
                        dml_irm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['se'],
                        dml_irm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_boot(dml_irm_fixture):
    for bootstrap in dml_irm_fixture['boot_methods']:
        assert np.allclose(dml_irm_fixture['boot_coef' + bootstrap],
                           dml_irm_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

