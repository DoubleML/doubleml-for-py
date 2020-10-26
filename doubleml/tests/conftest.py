import numpy as np
import pandas as pd

import pytest
from scipy.linalg import toeplitz

from sklearn.datasets import make_spd_matrix
from sklearn.datasets import make_regression

from doubleml.tests.helper_general import get_n_datasets
from doubleml.datasets import make_plr_turrell2018, make_irm_data, make_iivm_data, make_pliv_CHS2015


def g(x):
    return np.power(np.sin(x), 2)


def m(x,nu=0., gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


def m2(x):
    return np.power(x, 2)


def m3(x, nu=0., gamma=1.):
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
        data = make_plr_turrell2018(N, p, theta, return_type=pd.DataFrame)
        datasets.append(data)
    
    return datasets


@pytest.fixture(scope='session',
                params=[(500, 20)])
def generate_data2(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_plr_turrell2018(N, p, theta)
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
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_pliv_CHS2015(n_obs=N, dim_x=p, alpha=theta, dim_z=1, return_type=pd.DataFrame)
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
        data = make_irm_data(N, p, theta, return_type='array')
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
        data = make_iivm_data(N, p, theta, gamma_z, return_type=pd.DataFrame)
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

@pytest.fixture(scope='session',
                params=[500])
def generate_data_pliv_partialZ(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p
    theta = 1.

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_data_pliv_partialZ(N, alpha=theta, dim_x=5)
        datasets.append(data)

    return datasets


def make_data_pliv_partialZ(n_obs, alpha=1., dim_x=5, dim_z=150):
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.6], [0.6, 1.]]),
                                       size=[n_obs, ])
    epsilon = xx[:,0]
    u = xx[:,1]

    sigma = toeplitz([np.power(0.5, k) for k in range(1, dim_x + 1)])
    X = np.random.multivariate_normal(np.zeros(dim_x),
                                      sigma,
                                      size=[n_obs, ])

    I_z = np.eye(dim_z)
    xi = np.random.multivariate_normal(np.zeros(dim_z),
                                       0.25*I_z,
                                       size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    gamma = beta
    delta = [1 / (k**2) for k in range(1, dim_z + 1)]

    I_x = np.eye(dim_x)
    Pi = np.hstack((I_x, np.zeros((dim_x, dim_z-dim_x))))
    Z = np.dot(X, Pi) + xi

    D = np.dot(X, gamma) + np.dot(Z, delta) + u
    Y = alpha * D + np.dot(X, beta) + epsilon

    x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
    z_cols = [f'Z{i + 1}' for i in np.arange(dim_z)]
    data = pd.DataFrame(np.column_stack((X, Y, D, Z)),
                        columns=x_cols + ['y', 'd'] + z_cols)

    return data


@pytest.fixture(scope='session',
                params=[(253, 10), (501, 52)])
def generate_data_cv_predict(request):
    np.random.seed(3141)
    # setting parameters
    n_p = request.param
    n = n_p[0]
    p = n_p[1]

    # generating data
    datasets = []
    for i in range(n_datasets):
        x, y = make_regression(n_samples=n, n_features=p)
        datasets.append((x, y))

    return datasets
