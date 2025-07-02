import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def make_irm_data(n_obs=500, dim_x=20, theta=0, R2_d=0.5, R2_y=0.5, return_type="DoubleMLData"):
    """
    Generates data from a interactive regression (IRM) model.
    The data generating process is defined as

    .. math::

        d_i &= 1\\left\\lbrace \\frac{\\exp(c_d x_i' \\beta)}{1+\\exp(c_d x_i' \\beta)} > v_i \\right\\rbrace, & &v_i
        \\sim \\mathcal{U}(0,1),

        y_i &= \\theta d_i + c_y x_i' \\beta d_i + \\zeta_i, & &\\zeta_i \\sim \\mathcal{N}(0,1),

    with covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}`.
    :math:`\\beta` is a `dim_x`-vector with entries :math:`\\beta_j=\\frac{1}{j^2}` and the constants :math:`c_y` and
    :math:`c_d` are given by

    .. math::

        c_y = \\sqrt{\\frac{R_y^2}{(1-R_y^2) \\beta' \\Sigma \\beta}}, \\qquad c_d =
        \\sqrt{\\frac{(\\pi^2 /3) R_d^2}{(1-R_d^2) \\beta' \\Sigma \\beta}}.

    The data generating process is inspired by a process used in the simulation experiment (see Appendix P) of Belloni
    et al. (2017).

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    R2_d :
        The value of the parameter :math:`R_d^2`.
    R2_y :
        The value of the parameter :math:`R_y^2`.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.

    References
    ----------
    Belloni, A., Chernozhukov, V., Fernández‐Val, I. and Hansen, C. (2017). Program Evaluation and Causal Inference With
    High‐Dimensional Data. Econometrica, 85: 233-298.
    """
    # inspired by https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12723, see suplement
    v = np.random.uniform(
        size=[
            n_obs,
        ]
    )
    zeta = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y / ((1 - R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3.0 * R2_d / ((1 - R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    d = 1.0 * ((xx / (1 + xx)) > v)

    y = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)), columns=x_cols + ["y", "d"])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, "y", "d", x_cols)
    else:
        raise ValueError("Invalid return_type.")
