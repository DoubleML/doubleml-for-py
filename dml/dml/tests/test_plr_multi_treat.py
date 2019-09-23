import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_plr import DoubleMLPLR

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(2*n_datasets))
def idx(request):
    return request.param

@pytest.fixture(scope='module',
                params = [Lasso(alpha=0.1)])
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

@pytest.fixture(scope='module')
def dml_plr_multitreat_fixture(generate_data_bivariate, generate_data_toeplitz, idx, learner, inf_model, dml_procedure):
    boot_methods = ['normal']
    resampling = KFold(n_splits=2, shuffle=True)
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(clone(learner)),
                   'ml_g': clone(clone(learner))}
    
    dml_plr_obj = DoubleMLPLR(resampling,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    if idx < n_datasets:
        data = generate_data_bivariate[idx]
    else:
        data = generate_data_toeplitz[idx-n_datasets]
        
    np.random.seed(3141)
    dml_plr_obj.fit(data.loc[:, data.columns.str.startswith('X')].values,
                    data['y'].values,
                    data.loc[:, data.columns.str.startswith('d')].values)
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, data.columns.str.startswith('X')].values
    d = data.loc[:, data.columns.str.startswith('d')].values
    
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    n_d = d.shape[1]
    
    coef_manual = np.full(n_d, np.nan)
    se_manual = np.full(n_d, np.nan)
    
    all_g_hat = []
    all_m_hat = []
    
    for i_d in range(n_d):
        
        Xd = np.hstack((X, np.delete(d, i_d , axis=1)))
        
        g_hat, m_hat = fit_nuisance_plr(y, Xd, d[:, i_d],
                                        clone(learner), clone(learner), smpls)
        
        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)
        
        if dml_procedure == 'dml1':
            coef_manual[i_d], se_manual[i_d] = plr_dml1(y, Xd, d[:, i_d],
                                                        g_hat, m_hat,
                                                        smpls, inf_model)
        elif dml_procedure == 'dml2':
            coef_manual[i_d], se_manual[i_d] = plr_dml2(y, Xd, d[:, i_d],
                                                        g_hat, m_hat,
                                                        smpls, inf_model)
                   
    res_dict = {'coef': dml_plr_obj.coef_,
                'coef_manual': coef_manual,
                'se': dml_plr_obj.se_,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta = np.full((n_d, 500), np.nan)
        for i_d in range(n_d):
            boot_theta[i_d, :] = boot_plr(coef_manual[i_d],
                                          y, d[:, i_d],
                                          all_g_hat[i_d], all_m_hat[i_d],
                                          smpls, inf_model,
                                          se_manual[i_d],
                                          bootstrap, 500)
        
        np.random.seed(3141)
        dml_plr_obj.bootstrap(method = bootstrap, n_rep=500)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef_
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict

def test_dml_plr_multitreat_coef(dml_plr_multitreat_fixture):
    assert np.allclose(dml_plr_multitreat_fixture['coef'],
                       dml_plr_multitreat_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)

def test_dml_plr_multitreat_se(dml_plr_multitreat_fixture):
    assert np.allclose(dml_plr_multitreat_fixture['se'],
                       dml_plr_multitreat_fixture['se_manual'],
                       rtol=1e-9, atol=1e-4)

def test_dml_plr_multitreat_boot(dml_plr_multitreat_fixture):
    for bootstrap in dml_plr_multitreat_fixture['boot_methods']:
        assert np.allclose(dml_plr_multitreat_fixture['boot_coef' + bootstrap],
                           dml_plr_multitreat_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

