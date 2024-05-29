import numpy as np
import pandas as pd

import pytest
from scipy.linalg import toeplitz

from sklearn.datasets import make_spd_matrix
from doubleml.datasets import make_irm_data, make_iivm_data


def _g(x):
    return np.power(np.sin(x), 2)


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_irm(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    data = make_irm_data(n, p, theta, return_type='array')

    return data


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_irm_binary(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5
    b = [1 / k for k in range(1, p + 1)]
    sigma = make_spd_matrix(p)

    # generating data
    x = np.random.multivariate_normal(np.zeros(p), sigma, size=[n, ])
    G = _g(np.dot(x, b))
    pr = 1 / (1 + np.exp((-1) * (x[:, 0] * (-0.5) + x[:, 1] * 0.5 + np.random.standard_normal(size=[n, ]))))
    d = np.random.binomial(p=pr, n=1, size=[n, ])
    err = np.random.standard_normal(n)

    pry = 1 / (1 + np.exp((-1) * theta * d + G + err))
    y = np.random.binomial(p=pry, n=1, size=[n, ])

    return x, y, d


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20)])
def generate_data_irm_w_missings(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    (x, y, d) = make_irm_data(n, p, theta, return_type='array')

    # randomly set some entries to np.nan
    ind = np.random.choice(np.arange(x.size), replace=False,
                           size=int(x.size * 0.05))
    x[np.unravel_index(ind, x.shape)] = np.nan
    data = (x, y, d)

    return data


@pytest.fixture(scope='session',
                params=[(500, 11)])
def generate_data_iivm(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5
    gamma_z = 0.4

    # generating data
    data = make_iivm_data(n, p, theta, gamma_z, return_type=pd.DataFrame)

    return data


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_iivm_binary(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5
    b = [1 / k for k in range(1, p + 1)]
    sigma = make_spd_matrix(p)

    # generating data
    x = np.random.multivariate_normal(np.zeros(p), sigma, size=[n, ])
    G = _g(np.dot(x, b))

    prz = 1 / (1 + np.exp((-1) * (x[:, 0] * (-1) * b[4] + x[:, 1] * b[2] + np.random.standard_normal(size=[n, ]))))
    z = np.random.binomial(p=prz, n=1, size=[n, ])
    u = np.random.standard_normal(size=[n, ])
    pr = 1 / (1 + np.exp((-1) * (0.5 * z + x[:, 0] * (-0.5) + x[:, 1] * 0.25 - 0.5 * u
                                 + np.random.standard_normal(size=[n, ]))))
    d = np.random.binomial(p=pr, n=1, size=[n, ])
    err = np.random.standard_normal(n)

    pry = 1 / (1 + np.exp((-1) * theta * d + G + 4 * u + err))
    y = np.random.binomial(p=pry, n=1, size=[n, ])

    return x, y, d, z


@pytest.fixture(scope='session',
                params=[(500, 5),
                        (1000, 10)])
def generate_data_quantiles(request):
    n_p = request.param
    np.random.seed(1111)

    # setting parameters
    n = n_p[0]
    p = n_p[1]

    def f_loc(D, X):
        loc = 2 * D
        return loc

    def f_scale(D, X):
        scale = np.sqrt(0.5 * D + 1)
        return scale

    d = (np.random.normal(size=n) > 0) * 1.0
    x = np.random.uniform(0, 1, size=[n, p])
    epsilon = np.random.normal(size=n)

    y = f_loc(d, x) + f_scale(d, x) * epsilon
    data = (x, y, d)

    return data


@pytest.fixture(scope='session',
                params=[(5000, 5),
                        (10000, 10)])
def generate_data_local_quantiles(request):
    n_p = request.param
    np.random.seed(1111)

    # setting parameters
    n = n_p[0]
    p = n_p[1]

    def f_loc(D, X, X_conf):
        loc = 2 * D
        return loc

    def f_scale(D, X, X_conf):
        scale = np.sqrt(0.5 * D + 1)
        return scale

    def generate_treatment(Z, X, X_conf):
        eta = np.random.normal(size=len(Z))
        d = ((1.5 * Z + eta) > 0) * 1.0
        return d

    x = np.random.uniform(0, 1, size=[n, p])
    x_conf = np.random.uniform(-1, 1, size=[n, 4])
    z = np.random.binomial(1, p=0.5, size=n)
    d = generate_treatment(z, x, x_conf)
    epsilon = np.random.normal(size=n)

    y = f_loc(d, x, x_conf) + f_scale(d, x, x_conf)*epsilon
    data = (x, y, d, z)

    return data


@pytest.fixture(scope='session',
                params=[(8000, 2),
                        (16000, 5)])
def generate_data_selection_mar(request):
    params = request.param
    np.random.seed(1111)
    # setting parameters
    n_obs = params[0]
    dim_x = params[1]

    sigma = np.array([[1, 0], [0, 1]])
    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [0.4 / (k**2) for k in range(1, dim_x + 1)]

    d = np.where(np.dot(x, beta) + np.random.randn(n_obs) > 0, 1, 0)
    z = None
    s = np.where(np.dot(x, beta) + e[0] > 0, 1, 0)

    y = np.dot(x, beta) + 1 * d + e[1]
    y[s == 0] = 0

    data = (x, y, d, z, s)

    return data


@pytest.fixture(scope='session',
                params=[(8000, 2),
                        (16000, 5)])
def generate_data_selection_nonignorable(request):
    params = request.param
    np.random.seed(1111)
    # setting parameters
    n_obs = params[0]
    dim_x = params[1]

    sigma = np.array([[1, 0.5], [0.5, 1]])
    gamma = 1
    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [0.4 / (k**2) for k in range(1, dim_x + 1)]

    d = np.where(np.dot(x, beta) + np.random.randn(n_obs) > 0, 1, 0)
    z = np.random.randn(n_obs)
    s = np.where(np.dot(x, beta) + 0.25 * d + gamma * z + e[0] > 0, 1, 0)

    y = np.dot(x, beta) + 1 * d + e[1]
    y[s == 0] = 0

    data = (x, y, d, z, s)

    return data
