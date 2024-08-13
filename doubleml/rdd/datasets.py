import numpy as np


def make_simple_rdd_data(n_obs=5000, **kwargs):
    dim_x = kwargs.get('dim_x', 5)
    score = np.random.uniform(size=n_obs, low=-1, high=1)
    D = (score >= 0)
    # Make it fuzzy
    D[:300] = np.random.binomial(n=1, size=300, p=0.5)
    D = D.astype("int")

    # independent covariates
    X = np.random.uniform(size=(n_obs, dim_x), low=-1, high=1)
    g0 = X[:, 0] + score*X[:, 1]**2 + 3*np.sin(X[:, 2])
    g1 = g0

    epsilon = np.random.normal(size=n_obs)
    # potential outcomes
    Y0 = - score**2 + g0 + epsilon
    Y1 = 1 + score**2 + g1 + epsilon

    Y = Y0 * (1 - D) + Y1 * D

    oracle_values = {
        'Y0': Y0,
        'Y1': Y1,
    }
    res_dict = {
        'score': score,
        'Y': Y,
        'D': D,
        'X': X,
        'oracle_values': oracle_values
    }
    return res_dict
