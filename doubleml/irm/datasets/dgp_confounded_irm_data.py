import warnings

import numpy as np
from scipy.linalg import toeplitz


def make_confounded_irm_data(n_obs=500, theta=0.0, gamma_a=0.127, beta_a=0.58, linear=False, **kwargs):
    """
    Generates counfounded data from an interactive regression model.

    The data generating process is defined as follows (inspired by the Monte Carlo simulation used
    in Sant'Anna and Zhao (2020)).

    Let :math:`X= (X_1, X_2, X_3, X_4, X_5)^T \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` corresponds
    to the identity matrix.
    Further, define :math:`Z_j = (\\tilde{Z_j} - \\mathbb{E}[\\tilde{Z}_j]) / \\sqrt{\\text{Var}(\\tilde{Z}_j)}`,
    where

    .. math::

        \\tilde{Z}_1 &= \\exp(0.5 \\cdot X_1)

        \\tilde{Z}_2 &= 10 + X_2/(1 + \\exp(X_1))

        \\tilde{Z}_3 &= (0.6 + X_1 \\cdot X_3 / 25)^3

        \\tilde{Z}_4 &= (20 + X_2 + X_4)^2

        \\tilde{Z}_5 &= X_5.

    Additionally, generate a confounder :math:`A \\sim \\mathcal{U}[-1, 1]`.
    At first, define the propensity score as

    .. math::

        m(X, A) = P(D=1|X,A) = p(Z) + \\gamma_A \\cdot A

    where

    .. math::

        p(Z) &= \\frac{\\exp(f_{ps}(Z))}{1 + \\exp(f_{ps}(Z))},

        f_{ps}(Z) &= 0.75 \\cdot (-Z_1 + 0.1 \\cdot Z_2 -0.25 \\cdot Z_3 - 0.1 \\cdot Z_4).

    and generate the treatment :math:`D = 1\\{m(X, A) \\ge U\\}` with :math:`U \\sim \\mathcal{U}[0, 1]`.
    Since :math:`A` is independent of :math:`X`, the short form of the propensity score is given as

    .. math::

        P(D=1|X) = p(Z).

    Further, generate the outcome of interest :math:`Y` as

    .. math::

        Y &= \\theta \\cdot D (Z_5 + 1) + g(Z) + \\beta_A \\cdot A + \\varepsilon

        g(Z) &= 2.5 + 0.74 \\cdot Z_1 + 0.25 \\cdot Z_2 + 0.137 \\cdot (Z_3 + Z_4)

    where :math:`\\varepsilon \\sim \\mathcal{N}(0,5)`.
    This implies an average treatment effect of :math:`\\theta`. Additionally, the long and short forms of
    the conditional expectation take the following forms

    .. math::

        \\mathbb{E}[Y|D, X, A] &= \\theta \\cdot D (Z_5 + 1) + g(Z) + \\beta_A \\cdot A

        \\mathbb{E}[Y|D, X] &= (\\theta + \\beta_A \\frac{\\mathrm{Cov}(A, D(Z_5 + 1))}{\\mathrm{Var}(D(Z_5 + 1))})
            \\cdot D (Z_5 + 1) + g(Z).

    Consequently, the strength of confounding is determined via :math:`\\gamma_A` and :math:`\\beta_A`, which can be
    set via the parameters ``gamma_a`` and ``beta_a``.

    The observed data is given as :math:`W = (Y, D, Z)`.
    Further, orcale values of the confounder :math:`A`, the transformed covariated :math:`Z`,
    the potential outcomes of :math:`Y`, the long and short forms of the main regression and the propensity score and
    in sample versions of the confounding parameters :math:`cf_d` and :math:`cf_y` (for ATE and ATTE)
    are returned in a dictionary.

    Parameters
    ----------
    n_obs : int
        The number of observations to simulate.
        Default is ``500``.
    theta : float or int
        Average treatment effect.
        Default is ``0.0``.
    gamma_a : float
        Coefficient of the unobserved confounder in the propensity score.
        Default is ``0.127``.
    beta_a : float
        Coefficient of the unobserved confounder in the outcome regression.
        Default is ``0.58``.
    linear : bool
        If ``True``, the Z will be set to X, such that the underlying (short) models are linear/logistic.
        Default is ``False``.

    Returns
    -------
    res_dict : dictionary
       Dictionary with entries ``x``, ``y``, ``d`` and ``oracle_values``.

    References
    ----------
    Sant'Anna, P. H. and Zhao, J. (2020),
    Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
    doi:`10.1016/j.jeconom.2020.06.003 <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.
    """
    c = 0.0  # the confounding strength is only valid for c=0
    xi = 0.75
    dim_x = kwargs.get("dim_x", 5)
    trimming_threshold = kwargs.get("trimming_threshold", 0.01)
    var_eps_y = kwargs.get("var_eps_y", 1.0)

    # Specification of main regression function
    def f_reg(w):
        res = 2.5 + 0.74 * w[:, 0] + 0.25 * w[:, 1] + 0.137 * (w[:, 2] + w[:, 3])
        return res

    # Specification of prop score function
    def f_ps(w, xi):
        res = xi * (-w[:, 0] + 0.1 * w[:, 1] - 0.25 * w[:, 2] - 0.1 * w[:, 3])
        return res

    # observed covariates
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )
    z_tilde_1 = np.exp(0.5 * x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0] * x[:, 2] / 25) ** 3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3]) ** 2
    z_tilde_5 = x[:, 4]
    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4, z_tilde_5))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)
    # error terms and unobserved confounder
    eps_y = np.random.normal(loc=0, scale=np.sqrt(var_eps_y), size=n_obs)
    # unobserved confounder
    a_bounds = (-1, 1)
    a = np.random.uniform(low=a_bounds[0], high=a_bounds[1], size=n_obs)
    var_a = np.square(a_bounds[1] - a_bounds[0]) / 12

    # Choose the features used in the models
    if linear:
        features_ps = x
        features_reg = x
    else:
        features_ps = z
        features_reg = z

    p = np.exp(f_ps(features_ps, xi)) / (1 + np.exp(f_ps(features_ps, xi)))
    # compute short and long form of propensity score
    m_long = p + gamma_a * a
    m_short = p
    # check propensity score bounds
    if np.any(m_long < trimming_threshold) or np.any(m_long > 1.0 - trimming_threshold):
        m_long = np.clip(m_long, trimming_threshold, 1.0 - trimming_threshold)
        m_short = np.clip(m_short, trimming_threshold, 1.0 - trimming_threshold)
        warnings.warn(
            f"Propensity score is close to 0 or 1. "
            f"Trimming is at {trimming_threshold} and {1.0 - trimming_threshold} is applied"
        )
    # generate treatment based on long form
    u = np.random.uniform(low=0, high=1, size=n_obs)
    d = 1.0 * (m_long >= u)
    # add treatment heterogeneity
    d1x = z[:, 4] + 1
    var_dx = np.var(d * (d1x))
    cov_adx = gamma_a * var_a
    # Outcome regression
    g_partial_reg = f_reg(features_reg)
    # short model
    g_short_d0 = g_partial_reg
    g_short_d1 = (theta + beta_a * cov_adx / var_dx) * d1x + g_partial_reg
    g_short = d * g_short_d1 + (1.0 - d) * g_short_d0
    # long model
    g_long_d0 = g_partial_reg + beta_a * a
    g_long_d1 = theta * d1x + g_partial_reg + beta_a * a
    g_long = d * g_long_d1 + (1.0 - d) * g_long_d0
    # Potential outcomes
    y_0 = g_long_d0 + eps_y
    y_1 = g_long_d1 + eps_y
    # Realized outcome
    y = d * y_1 + (1.0 - d) * y_0
    # In-sample values for confounding strength
    explained_residual_variance = np.square(g_long - g_short)
    residual_variance = np.square(y - g_short)
    cf_y = np.mean(explained_residual_variance) / np.mean(residual_variance)
    # compute the Riesz representation
    treated_weight = d / np.mean(d)
    untreated_weight = (1.0 - d) / np.mean(d)
    # Odds ratios
    propensity_ratio_long = m_long / (1.0 - m_long)
    rr_long_ate = d / m_long - (1.0 - d) / (1.0 - m_long)
    rr_long_atte = treated_weight - np.multiply(untreated_weight, propensity_ratio_long)
    propensity_ratio_short = m_short / (1.0 - m_short)
    rr_short_ate = d / m_short - (1.0 - d) / (1.0 - m_short)
    rr_short_atte = treated_weight - np.multiply(untreated_weight, propensity_ratio_short)
    cf_d_ate = (np.mean(1 / (m_long * (1 - m_long))) - np.mean(1 / (m_short * (1 - m_short)))) / np.mean(
        1 / (m_long * (1 - m_long))
    )
    cf_d_atte = (np.mean(propensity_ratio_long) - np.mean(propensity_ratio_short)) / np.mean(propensity_ratio_long)
    if (beta_a == 0) | (gamma_a == 0):
        rho_ate = 0.0
        rho_atte = 0.0
    else:
        rho_ate = np.corrcoef((g_long - g_short), (rr_long_ate - rr_short_ate))[0, 1]
        rho_atte = np.corrcoef((g_long - g_short), (rr_long_atte - rr_short_atte))[0, 1]
    oracle_values = {
        "g_long": g_long,
        "g_short": g_short,
        "m_long": m_long,
        "m_short": m_short,
        "gamma_a": gamma_a,
        "beta_a": beta_a,
        "a": a,
        "y_0": y_0,
        "y_1": y_1,
        "z": z,
        "cf_y": cf_y,
        "cf_d_ate": cf_d_ate,
        "cf_d_atte": cf_d_atte,
        "rho_ate": rho_ate,
        "rho_atte": rho_atte,
    }
    res_dict = {"x": x, "y": y, "d": d, "oracle_values": oracle_values}
    return res_dict
