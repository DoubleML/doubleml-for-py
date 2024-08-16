import numpy as np


def make_simple_rdd_data(n_obs=5000, fuzzy=True, **kwargs):
    dim_x = kwargs.get('dim_x', 4)
    score = np.random.uniform(size=n_obs, low=-1, high=1)
    D = (score >= 0)
    if fuzzy:
        p = 0.2 + 0.2 * score + 0.5 * D
        D = np.random.binomial(n=1, p=p, size=n_obs)
    D = D.astype("int")

    # independent covariates
    X = np.random.uniform(size=(n_obs, dim_x), low=-1, high=1)
    g0 = X[:, 0] + 5*X[:, 1]**2 + 3*np.sin(X[:, 2])
    g1 = g0

    eps_scale = 0.5
    # potential outcomes with independent errors
    Y0 = - score**2 + g0 + np.random.normal(size=n_obs, scale=eps_scale)
    Y1 = 1 + score**2 + g1 + np.random.normal(size=n_obs, scale=eps_scale)

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
