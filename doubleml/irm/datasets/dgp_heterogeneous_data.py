import numpy as np
import pandas as pd


def make_heterogeneous_data(n_obs=200, p=30, support_size=5, n_x=1, binary_treatment=False):
    """
    Creates a simple synthetic example for heterogeneous treatment effects.
    The data generating process is based on the Monte Carlo simulation from Oprescu et al. (2019).

    The data is generated as

    .. math::

        Y_i & = \\theta_0(X_i)D_i + \\langle X_i,\\gamma_0\\rangle + \\epsilon_i

        D_i & = \\langle X_i,\\beta_0\\rangle + \\eta_i,

    where :math:`X_i\\sim\\mathcal{U}[0,1]^{p}` and :math:`\\epsilon_i,\\eta_i
    \\sim\\mathcal{U}[-1,1]`.
    If the treatment is set to be binary, the treatment is generated as

    .. math::
        D_i = 1\\{\\langle X_i,\\beta_0\\rangle \\ge \\eta_i\\}.

    The coefficient vectors :math:`\\gamma_0` and :math:`\\beta_0` both have small random (identical) support
    which values are drawn independently from :math:`\\mathcal{U}[0,1]` and :math:`\\mathcal{U}[0,0.3]`.
    Further, :math:`\\theta_0(x)` defines the conditional treatment effect, which is defined differently depending
    on the dimension of :math:`x`.

    If the heterogeneity is univariate the conditional treatment effect takes the following form

    .. math::
            \\theta_0(x) = \\exp(2x_0) + 3\\sin(4x_0),

    whereas for the two-dimensional case the conditional treatment effect is defined as

    .. math::
        \\theta_0(x) = \\exp(2x_0) + 3\\sin(4x_1).

    Parameters
    ----------
    n_obs : int
        Number of observations to simulate.
        Default is ``200``.

    p : int
        Dimension of covariates.
        Default is ``30``.

    support_size : int
        Number of relevant (confounding) covariates.
        Default is ``5``.

    n_x : int
        Dimension of the heterogeneity. Can be either ``1`` or ``2``.
        Default is ``1``.

    binary_treatment : bool
        Indicates whether the treatment is binary.
        Default is ``False``.

    Returns
    -------
    res_dict : dictionary
       Dictionary with entries ``data``, ``effects``, ``treatment_effect``.

    """
    # simple input checks
    assert n_x in [1, 2], "n_x must be either 1 or 2."
    assert support_size <= p, "support_size must be smaller than p."
    assert isinstance(binary_treatment, bool), "binary_treatment must be a boolean."

    # define treatment effects
    if n_x == 1:

        def treatment_effect(x):
            return np.exp(2 * x[:, 0]) + 3 * np.sin(4 * x[:, 0])

    else:
        assert n_x == 2

        # redefine treatment effect
        def treatment_effect(x):
            return np.exp(2 * x[:, 0]) + 3 * np.sin(4 * x[:, 1])

    # Outcome support and coefficients
    support_y = np.random.choice(np.arange(p), size=support_size, replace=False)
    coefs_y = np.random.uniform(0, 1, size=support_size)
    # treatment support and coefficients
    support_d = support_y
    coefs_d = np.random.uniform(0, 0.3, size=support_size)

    # noise
    epsilon = np.random.uniform(-1, 1, size=n_obs)
    eta = np.random.uniform(-1, 1, size=n_obs)

    # Generate controls, covariates, treatments and outcomes
    x = np.random.uniform(0, 1, size=(n_obs, p))
    # Heterogeneous treatment effects
    te = treatment_effect(x)
    if binary_treatment:
        d = 1.0 * (np.dot(x[:, support_d], coefs_d) >= eta)
    else:
        d = np.dot(x[:, support_d], coefs_d) + eta
    y = te * d + np.dot(x[:, support_y], coefs_y) + epsilon

    # Now we build the dataset
    y_df = pd.DataFrame({"y": y})
    d_df = pd.DataFrame({"d": d})
    x_df = pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=[f"X_{i}" for i in range(x.shape[1])])

    data = pd.concat([y_df, d_df, x_df], axis=1)
    res_dict = {"data": data, "effects": te, "treatment_effect": treatment_effect}
    return res_dict
