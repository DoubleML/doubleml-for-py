import numpy as np
import pandas as pd

import pytest

from sklearn.datasets import make_spd_matrix
from sklearn.datasets import make_regression, make_classification

from doubleml.datasets import make_plr_turrell2018, make_irm_data, \
    make_pliv_CHS2015


def _g(x):
    return np.power(np.sin(x), 2)


def _m(x, nu=0., gamma=1.):
    return 0.5 / np.pi * (np.sinh(gamma)) / (np.cosh(gamma) - np.cos(x - nu))


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
                params=[(1000, 20)])
def generate_data_bivariate(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = np.array([0.5, 0.9])
    b = [1 / k for k in range(1, p + 1)]
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
    column_names = [f'X{i + 1}' for i in np.arange(p)] + ['y'] + \
                   [f'd{i + 1}' for i in np.arange(2)]
    data = pd.DataFrame(np.column_stack((x, y, d)),
                        columns=column_names)

    return data
