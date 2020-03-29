import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_iivm import DoubleMLIIVM

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_iivm_manual import iivm_dml1, iivm_dml2, fit_nuisance_iivm, boot_iivm


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
                params = ['LATE'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_iivm_fixture(generate_data_iivm, idx, learner, inf_model, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 491

    # collect data
    data = generate_data_iivm[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner[0]),
                   'ml_g': clone(learner[1]),
                   'ml_r': clone(learner[0])}

    np.random.seed(3141)
    dml_iivm_obj = DoubleMLIIVM(data, X_cols, 'y', ['d'], 'z',
                                n_folds,
                                ml_learners,
                                dml_procedure,
                                inf_model)

    dml_iivm_obj.fit()
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    z = data['z'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = fit_nuisance_iivm(y, X, d, z,
                                                              clone(learner[0]), clone(learner[1]), clone(learner[0]), smpls)
    
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = iivm_dml1(y, X, d, z,
                                         g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = iivm_dml2(y, X, d, z,
                                         g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                         smpls, inf_model)
    
    res_dict = {'coef': dml_iivm_obj.coef,
                'coef_manual': res_manual,
                'se': dml_iivm_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = boot_iivm(res_manual,
                              y, d, z,
                              g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                              smpls, inf_model,
                              se_manual,
                              bootstrap, n_rep_boot,
                              dml_procedure)
        
        np.random.seed(3141)
        dml_iivm_obj.bootstrap(method = bootstrap, n_rep=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_iivm_obj.boot_coef
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_iivm_coef(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture['coef'],
                        dml_iivm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_iivm_se(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture['se'],
                        dml_iivm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_iivm_boot(dml_iivm_fixture):
    for bootstrap in dml_iivm_fixture['boot_methods']:
        assert np.allclose(dml_iivm_fixture['boot_coef' + bootstrap],
                           dml_iivm_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

