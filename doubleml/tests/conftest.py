import numpy as np
import pandas as pd

import pytest
from scipy.linalg import toeplitz

from sklearn.datasets import make_spd_matrix
from sklearn.datasets import make_regression, make_classification

from doubleml.datasets import make_plr_turrell2018, make_irm_data, make_iivm_data, make_pliv_CHS2015


def _g(x):
    return np.power(np.sin(x), 2)


def _m(x, nu=0., gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


def _m2(x):
    return np.power(x, 2)


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data1(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    data = make_plr_turrell2018(n, p, theta, return_type=pd.DataFrame)

    return data


@pytest.fixture(scope='session',
                params=[(500, 20)])
def generate_data2(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    data = make_plr_turrell2018(n, p, theta)

    return data


@pytest.fixture(scope='session',
                params=[(1000, 20)])
def generate_data_bivariate(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = np.array([0.5, 0.9])
    b = [1/k for k in range(1, p+1)]
    sigma = make_spd_matrix(p)

    # generating data
    x = np.random.multivariate_normal(np.zeros(p), sigma, size=[n, ])
    G = _g(np.dot(x, b))
    M0 = _m(np.dot(x, b))
    M1 = _m2(np.dot(x, b))
    D0 = M0 + np.random.standard_normal(size=[n, ])
    D1 = M1 + np.random.standard_normal(size=[n, ])
    y = theta[0] * D0 + theta[1] * D1 + G + np.random.standard_normal(size=[n, ])
    d = np.column_stack((D0, D1))
    column_names = [f'X{i+1}' for i in np.arange(p)] + ['y'] + \
                   [f'd{i+1}' for i in np.arange(2)]
    data = pd.DataFrame(np.column_stack((x, y, d)),
                        columns=column_names)

    return data


@pytest.fixture(scope='session',
                params=[(1000, 20)])
def generate_data_toeplitz(request, betamax=4, decay=0.99, threshold=0, noisevar=10):
    n_p = request.param
    np.random.seed(3141)
    # setting parameters
    n = n_p[0]
    p = n_p[1]

    beta = np.array([betamax * np.power(j+1, -decay) for j in range(p)])
    beta[beta < threshold] = 0

    cols_treatment = [0, 4, 9]

    sigma = toeplitz([np.power(0.9, k) for k in range(p)])
    mu = np.zeros(p)

    # generating data
    x = np.random.multivariate_normal(mu, sigma, size=[n, ])
    y = np.dot(x, beta) + np.random.normal(loc=0.0, scale=np.sqrt(noisevar), size=[n, ])
    d = x[:, cols_treatment]
    x = np.delete(x, cols_treatment, axis=1)
    column_names = [f'X{i+1}' for i in np.arange(x.shape[1])] + \
                   ['y'] + [f'd{i+1}' for i in np.arange(len(cols_treatment))]
    data = pd.DataFrame(np.column_stack((x, y, d)),
                        columns=column_names)

    return data


@pytest.fixture(scope='session',
                params=[(1000, 20)])
def generate_data_iv(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    data = make_pliv_CHS2015(n_obs=n, dim_x=p, alpha=theta, dim_z=1, return_type=pd.DataFrame)

    return data


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
                params=[500])
def generate_data_pliv_partialXZ(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p
    theta = 1.

    # generating data
    data = make_pliv_CHS2015(n, alpha=theta)

    return data


@pytest.fixture(scope='session',
                params=[500])
def generate_data_pliv_partialX(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p
    theta = 1.

    # generating data
    data = make_pliv_CHS2015(n, alpha=theta, dim_z=5)

    return data


@pytest.fixture(scope='session',
                params=[500])
def generate_data_pliv_partialZ(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p
    theta = 1.

    # generating data
    data = make_data_pliv_partialZ(n, alpha=theta, dim_x=5)

    return data


def make_data_pliv_partialZ(n_obs, alpha=1., dim_x=5, dim_z=150):
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.6], [0.6, 1.]]),
                                       size=[n_obs, ])
    epsilon = xx[:, 0]
    u = xx[:, 1]

    sigma = toeplitz([np.power(0.5, k) for k in range(1, dim_x + 1)])
    x = np.random.multivariate_normal(np.zeros(dim_x),
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
    z = np.dot(x, Pi) + xi

    d = np.dot(x, gamma) + np.dot(z, delta) + u
    y = alpha * d + np.dot(x, beta) + epsilon

    x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
    z_cols = [f'Z{i + 1}' for i in np.arange(dim_z)]
    data = pd.DataFrame(np.column_stack((x, y, d, z)),
                        columns=x_cols + ['y', 'd'] + z_cols)

    return data


@pytest.fixture(scope='session',
                params=[(253, 10, False), (501, 52, False),
                        (253, 10, True), (501, 52, True)])
def generate_data_cv_predict(request):
    np.random.seed(3141)
    # setting parameters
    n_p_c = request.param
    n = n_p_c[0]
    p = n_p_c[1]
    classifier = n_p_c[2]

    # generating data
    if classifier:
        x, y = make_classification(n_samples=n, n_features=p)
    else:
        x, y = make_regression(n_samples=n, n_features=p)
    data = (x, y, classifier)

    return data


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
