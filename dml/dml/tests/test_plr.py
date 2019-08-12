import numpy as np
import pytest
import math

from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.dml_plr import DoubleMLPLR

def g(x):
    return np.power(np.sin(x),2)

def m(x,nu=0.,gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))

# number of datasets per dgp
n_datasets = 10

@pytest.fixture(scope="module")
def generate_data1():
    # setting parameters
    N = 500
    k=10
    theta=0.5
    b= [1/k for k in range(1,11)]
    sigma = make_spd_matrix(k)
    
    # generating data
    np.random.seed(1111)
    datasets = []
    for i in range(n_datasets):
        X = np.random.multivariate_normal(np.ones(k),sigma,size=[N,])
        G = g(np.dot(X,b))
        M = m(np.dot(X,b))
        D = M+np.random.standard_normal(size=[500,])
        Y = np.dot(theta,D)+G+np.random.standard_normal(size=[500,])
        xx = {'X': X, 'y': Y, 'd': D}
        datasets.append(xx)
    
    return datasets


@pytest.mark.parametrize('idx', range(n_datasets))
@pytest.mark.parametrize('learner', [RandomForestRegressor(max_depth=2, n_estimators=10),
                                     LinearRegression(),
                                     Lasso(alpha=0.1)])
@pytest.mark.parametrize('inf_model', ['IV-type'])
@pytest.mark.parametrize('dml_procedure', ['dml1', 'dml2'])
def test_dml_plr(generate_data1, idx, learner, inf_model, dml_procedure):
    resampling = KFold(n_splits=2, shuffle=False)
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(clone(learner)),
                   'ml_g': clone(clone(learner))}
    
    dml_plr_obj = DoubleMLPLR(resampling,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    data = generate_data1[idx]
    np.random.seed(3141)
    res = dml_plr_obj.fit(data['X'], data['y'], data['d'])
    
    np.random.seed(3141)
    if dml_procedure == 'dml1':
        res_manual = plr_dml1(data['y'], data['X'], data['d'],
                              clone(learner), clone(learner), resampling)
    elif dml_procedure == 'dml2':
        res_manual = plr_dml2(data['y'], data['X'], data['d'],
                              clone(learner), clone(learner), resampling)
    
    assert math.isclose(res.coef_, res_manual, rel_tol=1e-9, abs_tol=0.0)
    
    return
    
def fit_nuisance(Y, X, D, ml_m, ml_g, smpls):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index],Y[train_index]).predict(X[test_index]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],D[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat


def plr_dml1(Y, X, D, ml_m, ml_g, resampling):
    thetas = np.zeros(resampling.get_n_splits())
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat, m_hat = fit_nuisance(Y, X, D, ml_m, ml_g, smpls)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat = D[test_index] - m_hat[idx]
        thetas[idx] = np.mean(np.dot(v_hat, (Y[test_index] - g_hat[idx])))/np.mean(np.dot(v_hat, D[test_index]))
    theta_hat = np.mean(thetas)
    
    return theta_hat

def plr_dml2(Y, X, D, ml_m, ml_g, resampling):
    thetas = np.zeros(resampling.get_n_splits())
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat, m_hat = fit_nuisance(Y, X, D, ml_m, ml_g, smpls)
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat[test_index] = D[test_index] - m_hat[idx]
        u_hat[test_index] = Y[test_index] - g_hat[idx]
    
    theta_hat = np.mean(np.dot(v_hat, u_hat))/np.mean(np.dot(v_hat, D))
    
    return theta_hat
