import numpy as np
from scipy.linalg import toeplitz


def make_irm_data_discrete_treatments(n_obs=200, n_levels=3, linear=False, random_state=None, **kwargs):
    """
    Generates data from a interactive regression (IRM) model with multiple treatment levels (based on an
    underlying continous treatment).

    The data generating process is defined as follows (similar to the Monte Carlo simulation used
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

    A continuous treatment :math:`D_{\\text{cont}}` is generated as

    .. math::

        D_{\\text{cont}} = \\xi (-Z_1 + 0.5 Z_2 - 0.25 Z_3 - 0.1 Z_4) + \\varepsilon_D,

    where :math:`\\varepsilon_D \\sim \\mathcal{N}(0,1)` and :math:`\\xi=0.3`. The corresponding treatment
    effect is defined as

    .. math::

        \\theta (d) = 0.1 \\exp(d) + 10 \\sin(0.7 d) + 2 d - 0.2 d^2.

    Based on the continous treatment, a discrete treatment :math:`D` is generated as with a baseline level of
    :math:`D=0` and additional levels based on the quantiles of :math:`D_{\\text{cont}}`. The number of levels
    is defined by :math:`n_{\\text{levels}}`. Each level is chosen to have the same probability of being selected.

    The potential outcomes are defined as

    .. math::

            Y(0) &= 210 + 27.4 Z_1 + 13.7 (Z_2 + Z_3 + Z_4) + \\varepsilon_Y

            Y(1) &= \\theta (D_{\\text{cont}}) 1\\{D_{\\text{cont}} > 0\\} + Y(0),

    where :math:`\\varepsilon_Y \\sim \\mathcal{N}(0,5)`. Further, the observed outcome is defined as

    .. math::

        Y = Y(1) 1\\{D > 0\\} + Y(0) 1\\{D = 0\\}.

    The data is returned as a dictionary with the entries ``x``, ``y``, ``d`` and ``oracle_values``.

    Parameters
    ----------
    n_obs : int
        The number of observations to simulate.
        Default is ``200``.

    n_levels : int
        The number of treatment levels.
        Default is ``3``.

    linear : bool
        Indicates whether the true underlying regression is linear.
        Default is ``False``.

    random_state : int
        Random seed for reproducibility.
        Default is ``42``.

    Returns
    -------
    res_dict : dictionary
       Dictionary with entries ``x``, ``y``, ``d`` and ``oracle_values``.
       The oracle values contain the continuous treatment, the level bounds, the potential level, ITE
       and the potential outcome without treatment.

    """
    if random_state is not None:
        np.random.seed(random_state)
    xi = kwargs.get("xi", 0.3)
    c = kwargs.get("c", 0.0)
    dim_x = kwargs.get("dim_x", 5)

    if not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer.")
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2.")

    # observed covariates
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    def f_reg(w):
        res = 210 + 27.4 * w[:, 0] + 13.7 * (w[:, 1] + w[:, 2] + w[:, 3])
        return res

    def f_treatment(w, xi):
        res = xi * (-w[:, 0] + 0.5 * w[:, 1] - 0.25 * w[:, 2] - 0.1 * w[:, 3])
        return res

    def treatment_effect(d, scale=15):
        return scale * (1 / (1 + np.exp(-d - 1.2 * np.cos(d)))) - 2

    z_tilde_1 = np.exp(0.5 * x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0] * x[:, 2] / 25) ** 3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3]) ** 2

    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4, x[:, 4:]))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)

    # error terms
    var_eps_y = 5
    eps_y = np.random.normal(loc=0, scale=np.sqrt(var_eps_y), size=n_obs)
    var_eps_d = 1
    eps_d = np.random.normal(loc=0, scale=np.sqrt(var_eps_d), size=n_obs)

    if linear:
        g = f_reg(x)
        m = f_treatment(x, xi)
    else:
        assert not linear
        g = f_reg(z)
        m = f_treatment(z, xi)

    cont_d = m + eps_d
    level_bounds = np.quantile(cont_d, q=np.linspace(0, 1, n_levels + 1))
    potential_level = sum([1.0 * (cont_d >= bound) for bound in level_bounds[1:-1]]) + 1
    eta = np.random.uniform(0, 1, size=n_obs)
    d = 1.0 * (eta >= 1 / n_levels) * potential_level

    ite = treatment_effect(cont_d)
    y0 = g + eps_y
    # only treated for d > 0 compared to the baseline
    y = ite * (d > 0) + y0

    oracle_values = {
        "cont_d": cont_d,
        "level_bounds": level_bounds,
        "potential_level": potential_level,
        "ite": ite,
        "y0": y0,
    }

    resul_dict = {"x": x, "y": y, "d": d, "oracle_values": oracle_values}

    return resul_dict
