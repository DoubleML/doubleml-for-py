import numpy as np
import pytest
import math
import scipy

from sklearn.datasets import make_spd_matrix

from dml.tests.helper_general import get_n_datasets


def g(x):
    return np.power(np.sin(x),2)

def m(x,nu=0.,gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope="module",
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
    b= [1/k for k in range(1,p+1)]
    sigma = make_spd_matrix(p)
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        X = np.random.multivariate_normal(np.ones(p),sigma,size=[N,])
        G = g(np.dot(X,b))
        M = m(np.dot(X,b))
        D = M+np.random.standard_normal(size=[N,])
        Y = np.dot(theta,D)+G+np.random.standard_normal(size=[N,])
        xx = {'X': X, 'y': Y, 'd': D}
        datasets.append(xx)
    
    return datasets
    
@pytest.fixture(scope="module",
                params = [(1000, 20)])
def generate_data_iv(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta=0.5
    gamma_z=0.4
    b= [1/k for k in range(1,p+1)]
    sigma = make_spd_matrix(p)
    
    # generating data
    datasets = []
    for i in range(n_datasets):
        X = np.random.multivariate_normal(np.ones(p),sigma,size=[N,])
        G = g(np.dot(X,b))
        # instrument 
        Z = m(np.dot(X,b)) + np.random.standard_normal(size=[N,])
        M = m(gamma_z * Z + np.dot(X,b))
        # treatment
        D = M + np.random.standard_normal(size=[N,])
        Y = np.dot(theta,D)+G+np.random.standard_normal(size=[N,])
        xx = {'X': X, 'y': Y, 'd': D, 'z': Z}
        datasets.append(xx)
    
    return datasets

