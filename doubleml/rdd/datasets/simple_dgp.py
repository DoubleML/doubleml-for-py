import numpy as np
from numpy.polynomial.polynomial import Polynomial


def make_simple_rdd_data(n_obs=5000, p=4, fuzzy=True, binary_outcome=False, **kwargs):
    """
    Generates synthetic data for a regression discontinuity design (RDD) analysis.

    .. math::
        Y_0 &= g_0 + g_{cov} + \\epsilon_0 \\
        Y_1 &= g_1 + g_{cov} + \\epsilon_1 \\
        g_0 &= 0.1 \\cdot \\text{score}^2 \\
        g_1 &= 1 + 0.1 \\cdot \\text{score}^2 - 0.5 \\cdot \\text{score}^2 \\
        g_{cov} &= \\sum_{i=1}^{\text{dim\\_x}} \text{Polynomial}(X_i) \\
        \\epsilon_0, \\epsilon_1 &\\sim \\mathcal{N}(0, 0.2^2)

    Parameters
    ----------
    n_obs : int
        Number of observations to generate. Default is 5000.

    p : int
        Degree of the polynomial for covariates. Default is 4.

    fuzzy : bool
        If True, generates data for a fuzzy RDD. Default is True.

    binary_outcome : bool
        If True, generates binary outcomes. Default is False.

    **kwargs : Additional keyword arguments.
        cutoff : float
            The cutoff value for the score. Default is 0.0.
        dim_x : int
            The number of independent covariates. Default is 3.
        a : float
            Factor to control interaction of score and covariates to the outcome equation.

    Returns
    -------
    dict: A dictionary containing the generated data with keys:
        'score' (np.ndarray): The running variable.
        'X' (np.ndarray): The independent covariates.
        'Y0' (np.ndarray): The potential outcomes without treatment.
        'Y1' (np.ndarray): The potential outcomes with treatment.
        'intended_treatment' (np.ndarray): The intended treatment assignment.
    """

    cutoff = kwargs.get('cutoff', 0.0)
    dim_x = kwargs.get('dim_x', 3)
    a = kwargs.get('a', 0)

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
    g1 = 1 + 0.1 * score**2 - 0.5 * score**2 + a * np.sum(X, axis=1) * score

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
