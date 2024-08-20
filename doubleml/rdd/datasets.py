import numpy as np
from numpy.polynomial.polynomial import Polynomial


def make_simple_rdd_data(n_obs=5000, p=4, fuzzy=True, **kwargs):
    cutoff = kwargs.get('cutoff', 0.0)
    dim_x = kwargs.get('dim_x', 3)

    score = np.random.normal(size=n_obs)
    intended_treatment = (score >= cutoff).astype(int)
    # independent covariates
    X = np.random.uniform(size=(n_obs, dim_x), low=-1, high=1)

    # Create polynomials of covariates
    if p == 0:
        covs = np.zeros((n_obs, 1))
    else:
        covs = np.column_stack([Polynomial(np.arange(p + 1))(X[:, i]) for i in range(X.shape[1])])
    g_cov = np.sum(covs, axis=1)

    g0 = 0.1 * score**2
    g1 = intended_treatment + 0.1 * score**2 - 0.5 * score**2 * intended_treatment

    eps_scale = 0.2
    # potential outcomes with independent errors
    Y0 = g0 + g_cov + np.random.normal(size=n_obs, scale=eps_scale)
    Y1 = g1 + g_cov + np.random.normal(size=n_obs, scale=eps_scale)

    D = (score >= 0)
    if fuzzy:
        prob = 0.3 + 0.4 * intended_treatment + 0.01 * score**2 - 0.02 * score**2 * intended_treatment + 0.2 * g_cov
        prob = np.clip(prob, 0.0, 1.0)
        D = np.random.binomial(n=1, p=prob, size=n_obs)
    D = D.astype(int)

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
