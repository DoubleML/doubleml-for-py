import numpy as np
import pandas as pd


def make_plpr_CP2025(num_id=250, num_t=10, dim_x=30, theta=0.5, dgp_type="dgp1"):
    """
    Generates synthetic data for a partially linear panel regression model, based on Clarke and Polselli (2025).
    The data generating process is defined as

        .. math::

        Y_{it} &= D_{it} \\theta + l_0(X_{it}) + \\alpha_i + U_{it}, & &U_{it} \\sim \\mathcal{N}(0,1),

        D_{it} &= m_0(X_{it}) + c_i + V_{it}, & &V_{it} \\sim \\mathcal{N}(0,1),

        \\alpha_i &= 0.25 \\left(\\frac{1}{T} \\sum_{t=1}^{T} D_{it} - \\bar{D} \\right)
        + 0.25 \\frac{1}{T} \\sum_{t=1}^{T} \\sum_{k \\in \\mathcal{K}} X_{it,k} + a_i


    with :math:`a_i \\sim \\mathcal{N}(0,0.95)`, :math:`X_{it,p} \\sim \\mathcal{N}(0,5)`, :math:`c_i \\sim \\mathcal{N}(0,1)`.
    Where :math:`k \\in \\mathcal{K} = \\{1,3\\}` is the number of relevant (non-zero) confounding variables, and :math:`p` is
    the number of total confounding variables.

    Clarke and Polselli (2025) consider three functional forms of the confounders to model the nuisance functions :math:`l_0`
    and :math:`m_0` with varying levels of non-linearity and non-smoothness:

    Design 1. (dgp1): Linear in the nuisance parameters

        .. math::

        l_0(X_{it}) &= a X_{it,1} + X_{it,3}

        m_0(X_{it}) &= a X_{it,1} + X_{it,3}

    Design 2. (dgp2): Non-linear and smooth in the nuisance parameters

        .. math::

        l_0(X_{it}) &= \\frac{\\exp(X_{it,1})}{1 + \\exp(X_{it,1})} + a \\cos(X_{it,3})

        m_0(X_{it}) &= \\cos(X_{it,1}) + a \\frac{\\exp(X_{it,3})}{1 + \\exp(X_{it,3})}

    Design 3. (dgp3): Non-linear and discontinuous in the nuisance parameters

        .. math::

        l_0(X_{it}) &= b (X_{it,1} \\cdot X_{it,3}) + a (X_{it,3} \\cdot 1\\{X_{it,3} > 0\\})

        m_0(X_{it}) &= a (X_{it,1} \\cdot 1\\{X_{it,1} > 0\\}) + b (X_{it,1} \\cdot X_{it,3}),

    where :math:`a = 0.25`, :math:`b = 0.5`.

    Parameters
    ----------
    num_id :
        The number of units in the panel.
    num_t :
        The number of time periods in the panel.
    num_x :
        The number of confounding variables.
    theta :
        The value of the causal parameter.
    dgp_type :
        The type of DGP design to be used. Default is ``'dgp1'``, other options are ``'dgp2'`` and ``'dgp3'``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the simulated static panel data.

    References
    ----------
    Clarke, P. S. and Polselli, A. (2025),
    Double machine learning for static panel models with fixed effects. The Econometrics Journal, utaf011,
    doi:`10.1093/ectj/utaf011 <https://doi.org/10.1093/ectj/utaf011>`_.
    """

    # parameters
    a = 0.25
    b = 0.5
    sigma2_a = 0.95
    sigma2_x = 5

    # id and time vectors
    id_var = np.repeat(np.arange(1, num_id + 1), num_t)
    time = np.tile(np.arange(1, num_t + 1), num_id)

    # individual fixed effects
    a_i = np.repeat(np.random.normal(0, np.sqrt(sigma2_a), num_id), num_t)
    c_i = np.repeat(np.random.standard_normal(num_id), num_t)

    # covariates and errors
    x_mean = 0
    x_it = np.random.normal(loc=x_mean, scale=np.sqrt(sigma2_x), size=(num_id * num_t, dim_x))
    u_it = np.random.standard_normal(num_id * num_t)
    v_it = np.random.standard_normal(num_id * num_t)

    # functional forms in nuisance functions
    if dgp_type == "dgp1":
        l_0 = a * x_it[:, 0] + x_it[:, 2]
        m_0 = a * x_it[:, 0] + x_it[:, 2]
    elif dgp_type == "dgp2":
        l_0 = np.divide(np.exp(x_it[:, 0]), 1 + np.exp(x_it[:, 0])) + a * np.cos(x_it[:, 2])
        m_0 = np.cos(x_it[:, 0]) + a * np.divide(np.exp(x_it[:, 2]), 1 + np.exp(x_it[:, 2]))
    elif dgp_type == "dgp3":
        l_0 = b * (x_it[:, 0] * x_it[:, 2]) + a * (x_it[:, 2] * np.where(x_it[:, 2] > 0, 1, 0))
        m_0 = a * (x_it[:, 0] * np.where(x_it[:, 0] > 0, 1, 0)) + b * (x_it[:, 0] * x_it[:, 2])
    else:
        raise ValueError("Invalid dgp type.")

    # treatment
    d_it = m_0 + c_i + v_it

    def alpha_i(x_it, d_it, a_i, num_n, num_t):
        d_i = np.array_split(d_it, num_n)
        d_i_term = np.repeat(np.mean(d_i, axis=1), num_t) - np.mean(d_it)

        x_i = np.array_split(np.sum(x_it[:, [0, 2]], axis=1), num_n)
        x_i_mean = np.mean(x_i, axis=1)
        x_i_term = np.repeat(x_i_mean, num_t)

        alpha_term = 0.25 * d_i_term + 0.25 * x_i_term + a_i
        return alpha_term

    # outcome
    y_it = d_it * theta + l_0 + alpha_i(x_it, d_it, a_i, num_id, num_t) + u_it

    x_cols = [f"x{i + 1}" for i in np.arange(dim_x)]

    data = pd.DataFrame(np.column_stack((id_var, time, y_it, d_it, x_it)), columns=["id", "time", "y", "d"] + x_cols).astype(
        {"id": "int64", "time": "int64"}
    )

    return data
