import numpy as np
from numpy.polynomial.polynomial import Polynomial


def make_simple_rdd_data(n_obs=5000, p=4, fuzzy=True, binary_outcome=False, **kwargs):
    """
    Generates synthetic data for a regression discontinuity design (RDD) analysis.
    The data generating process is defined as

    .. math::
        Y_0 &= g_0 + g_{cov} + \\epsilon_0,

        Y_1 &= g_1 + g_{cov} + \\epsilon_1,

        g_0 &= 0.1 \\cdot \\text{score}^2,

        g_1 &= \\tau + 0.1 \\cdot score^2 - 0.5 \\cdot score^2 + a
        \\sum_{i=1}^{\\text{dim}_x} X_i \\cdot score,

        g_{cov} &= \\sum_{i=1}^{\\text{dim}_x} \\text{Polynomial}(X_i),

    with random noise :math:`\\epsilon_0, \\epsilon_1 \\sim \\mathcal{N}(0, 0.2^2)` and :math:`X_i`
    being drawn independently from a uniform distribution.

    Parameters
    ----------
    n_obs : int
        Number of observations to generate. Default is 5000.

    p : int
        Degree of the polynomial for covariates. Default is 4. If zero, no covariate effect is considered.

    fuzzy : bool
        If True, generates data for a fuzzy RDD. Default is True.

    binary_outcome : bool
        If True, generates binary outcomes based on a logistic transformation. Default is False.

    **kwargs : Additional keyword arguments.
        cutoff : float
            The cutoff value for the score. Default is 0.0.
        dim_x : int
            The number of independent covariates. Default is 3.
        a : float
            Factor to control interaction of score and covariates in the outcome equation. Default is 0.0.
        tau : float
            Parameter to control the true effect in the generated data at the given cutoff. Default is 1.0.

    Returns
    -------
    res_dict : dictionary
        Dictionary with entries ``score``, ``X``, ``Y``, ``D``, and ``oracle_values``.
        The oracle values contain the potential outcomes.
    """

    cutoff = kwargs.get('cutoff', 0.0)
    dim_x = kwargs.get('dim_x', 3)
    a = kwargs.get('a', 0.0)
    tau = kwargs.get('tau', 1.0)

    score = np.random.normal(size=n_obs)
    # independent covariates
    X = np.random.uniform(size=(n_obs, dim_x), low=-1, high=1)

    # Create polynomials of covariates
    if p == 0:
        covs = np.zeros((n_obs, 1))
    else:
        covs = np.column_stack([Polynomial(np.arange(p + 1))(X[:, i]) for i in range(X.shape[1])])
    g_cov = np.sum(covs, axis=1)

    g0 = 0.1 * score**2
    g1 = tau + 0.1 * score**2 - 0.5 * score**2 + a * np.sum(X, axis=1) * score

    eps_scale = 0.2
    # potential outcomes with independent errors
    if not binary_outcome:
        Y0 = g0 + g_cov + np.random.normal(size=n_obs, scale=eps_scale)
        Y1 = g1 + g_cov + np.random.normal(size=n_obs, scale=eps_scale)
    else:
        p_Y0 = 1 / (1 + np.exp(-1.0 * (g0 + g_cov)))
        p_Y1 = 1 / (1 + np.exp(-1.0 * (g1 + g_cov)))
        Y0 = np.random.binomial(n=1, p=p_Y0, size=n_obs)
        Y1 = np.random.binomial(n=1, p=p_Y1, size=n_obs)

    intended_treatment = (score >= cutoff).astype(int)
    if fuzzy:
        prob = 0.3 + 0.4 * intended_treatment + 0.01 * score**2 - 0.02 * score**2 * intended_treatment + 0.2 * g_cov
        prob = np.clip(prob, 0.0, 1.0)
        D = np.random.binomial(n=1, p=prob, size=n_obs)
    else:
        D = intended_treatment

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
