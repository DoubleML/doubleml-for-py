import numpy as np
import pandas as pd

import pytest
from scipy.linalg import toeplitz

from sklearn.datasets import make_spd_matrix

from doubleml.tests.helper_general import get_n_datasets
from doubleml.datasets import make_plr_data, make_pliv_data, make_irm_data, make_iivm_data, make_pliv_CHS2015


def g(x):
    return np.power(np.sin(x),2)

def m(x,nu=0.,gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))

def m2(x):
    return np.power(x,2)

def m3(x,nu=0.,gamma=1.):
    return 1./np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='session',
                params = [(500, 10),
                          (1000, 20),
                          (1000, 100)])
def generate_data1(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta=0.5
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_plr_data(N, p, theta)
        datasets.append(data)
    
    return datasets

@pytest.fixture(scope='session',
                params = [(1000, 20)])
def generate_data_bivariate(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta=np.array([0.5, 0.9])
    b= [1/k for k in range(1,p+1)]
    sigma = make_spd_matrix(p)
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        X = np.random.multivariate_normal(np.zeros(p),sigma,size=[N,])
        G = g(np.dot(X,b))
        M0 = m(np.dot(X,b))
        M1 = m2(np.dot(X,b))
        D0 = M0 + np.random.standard_normal(size=[N,])
        D1 = M1 + np.random.standard_normal(size=[N,])
        Y = theta[0]*D0 + theta[1]*D1 +G+np.random.standard_normal(size=[N,])
        D = np.column_stack((D0, D1))
        xx = {'X': X, 'y': Y, 'd': D}
        column_names = [f'X{i+1}' for i in np.arange(p)] \
                       + ['y'] + [f'd{i+1}' for i in np.arange(2)]
        data = pd.DataFrame(np.column_stack((X, Y, D)),
                            columns = column_names)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[(1000, 20)])
def generate_data_toeplitz(request, betamax=4, decay=0.99, threshold=0, noisevar=10):
    N_p = request.param
    np.random.seed(3141)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    
    beta = np.array([betamax * np.power(j+1, -decay) for j in range(p)])
    beta[beta < threshold] = 0
    
    cols_treatment = [0, 4, 9]
    
    sigma = toeplitz([np.power(0.9, k) for k in range(p)])
    mu = np.zeros(p)
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        X = np.random.multivariate_normal(mu,sigma,size=[N,])
        Y = np.dot(X, beta) + np.random.normal(loc=0.0, scale=np.sqrt(noisevar), size=[N,])
        D = X[:, cols_treatment]
        X = np.delete(X, cols_treatment, axis=1)
        xx = {'X': X, 'y': Y, 'd': D}
        column_names = [f'X{i+1}' for i in np.arange(X.shape[1])] \
                       + ['y'] + [f'd{i+1}' for i in np.arange(len(cols_treatment))]
        data = pd.DataFrame(np.column_stack((X, Y, D)),
                            columns = column_names)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[(1000, 20)])
def generate_data_iv(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5
    gamma_z = 0.4
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_pliv_data(N, p, theta, gamma_z)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_irm(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_irm_data(N, p, theta, return_X_y_d=True)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[(500, 11)])
def generate_data_iivm(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5
    gamma_z = 0.4
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_iivm_data(N, p, theta, gamma_z)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[500])
def generate_data_pliv_partialXZ(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p
    theta = 1.

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_pliv_CHS2015(N, alpha=theta)
        datasets.append(data)

    return datasets


@pytest.fixture(scope='session',
                params=[500])
def generate_data_pliv_partialX(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p
    theta = 1.

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_pliv_CHS2015(N, alpha=theta, dim_z=5)
        datasets.append(data)

    return datasets
