import pandas as pd
import numpy as np

from sklearn.datasets import make_spd_matrix


def fetch_401K():
    url = 'https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta'
    data = pd.read_stata(url)
    return data


def fetch_bonus():
    url = 'https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat'
    data = pd.read_stata(url)
    return data


def g(x):
    return np.power(np.sin(x), 2)


def m(x, nu=0., gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


def m2(x):
    return np.power(x, 2)


def m3(x, nu=0., gamma=1.):
    return 1./np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


def make_plr_data(n_samples=100, n_features=20, theta=0.5):
    b = [1 / k for k in range(1, n_features + 1)]
    sigma = make_spd_matrix(n_features)

    X = np.random.multivariate_normal(np.zeros(n_features), sigma, size=[n_samples, ])
    G = g(np.dot(X, b))
    M = m(np.dot(X, b))
    D = M + np.random.standard_normal(size=[n_samples, ])
    Y = np.dot(theta, D) + G + np.random.standard_normal(size=[n_samples, ])

    data = pd.DataFrame(np.column_stack((X, Y, D)),
                        columns=[f'X{i + 1}' for i in np.arange(n_features)] + ['y', 'd'])

    return data


def make_pliv_data(n_samples=100, n_features=20, theta=0.5, gamma_z=0.4):
    b = [1/k for k in range(1, n_features+1)]
    sigma = make_spd_matrix(n_features)

    X = np.random.multivariate_normal(np.zeros(n_features), sigma, size=[n_samples, ])
    G = g(np.dot(X, b))
    # instrument
    Z = m(np.dot(X, b)) + np.random.standard_normal(size=[n_samples, ])
    # treatment
    M = m(gamma_z * Z + np.dot(X, b))
    D = M + np.random.standard_normal(size=[n_samples, ])
    Y = np.dot(theta, D) + G + np.random.standard_normal(size=[n_samples, ])

    data = pd.DataFrame(np.column_stack((X, Y, D, Z)),
                        columns=[f'X{i + 1}' for i in np.arange(n_features)] + ['y', 'd', 'z'])

    return data


def make_irm_data(n_samples=100, n_features=20, theta=0.5):
    b = [1/k for k in range(1,n_features+1)]
    sigma = make_spd_matrix(n_features)

    X = np.random.multivariate_normal(np.zeros(n_features), sigma, size=[n_samples, ])
    G = g(np.dot(X, b))
    M = m3(np.dot(X, b))
    MM = M + np.random.standard_normal(size=[n_samples, ])
    MMM = np.maximum(np.minimum(MM, 0.99), 0.01)
    D = np.random.binomial(p=MMM, n=1)
    Y = np.dot(theta, D) + G + np.random.standard_normal(size=[n_samples, ])
    data = pd.DataFrame(np.column_stack((X, Y, D)),
                        columns=[f'X{i + 1}' for i in np.arange(n_features)] + ['y', 'd'])

    return data


def make_iivm_data(n_samples=100, n_features=20, theta=0.5, gamma_z=0.4):
    b = [1/k for k in range(1, n_features+1)]
    sigma = make_spd_matrix(n_features)

    X = np.random.multivariate_normal(np.zeros(n_features), sigma, size=[n_samples, ])
    G = g(np.dot(X, b))
    # instrument
    M1 = m3(np.dot(X, b))
    MM = M1 + np.random.standard_normal(size=[n_samples, ])
    MMM = np.maximum(np.minimum(MM, 0.99), 0.01)
    Z = np.random.binomial(p=MMM, n=1)
    # treatment
    M = m3(gamma_z * Z + np.dot(X, b))
    MM = M + np.random.standard_normal(size=[n_samples, ])
    MMM = np.maximum(np.minimum(MM, 0.99), 0.01)
    D = np.random.binomial(p=MMM, n=1)
    Y = np.dot(theta, D) + G + np.random.standard_normal(size=[n_samples, ])
    data = pd.DataFrame(np.column_stack((X, Y, D, Z)),
                        columns=[f'X{i + 1}' for i in np.arange(n_features)] + ['y', 'd', 'z'])

    return data

