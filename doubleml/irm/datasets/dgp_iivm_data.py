import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def make_iivm_data(n_obs=500, dim_x=20, theta=1.0, alpha_x=0.2, return_type="DoubleMLData"):
    """
    Generates data from a interactive IV regression (IIVM) model.
    The data generating process is defined as

    .. math::

        d_i &= 1\\left\\lbrace \\alpha_x Z + v_i > 0 \\right\\rbrace,

        y_i &= \\theta d_i + x_i' \\beta + u_i,

    with :math:`Z \\sim \\text{Bernoulli}(0.5)` and

    .. math::

        \\left(\\begin{matrix} u_i \\\\ v_i \\end{matrix} \\right) \\sim
        \\mathcal{N}\\left(0, \\left(\\begin{matrix} 1 & 0.3 \\\\ 0.3 & 1 \\end{matrix} \\right) \\right).

    The covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}` and :math:`\\beta` is a `dim_x`-vector with entries
    :math:`\\beta_j=\\frac{1}{j^2}`.

    The data generating process is inspired by a process used in the simulation experiment of Farbmacher, Gruber and
    Klaassen (2020).

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    alpha_x :
        The value of the parameter :math:`\\alpha_x`.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d, z)``.

    References
    ----------
    Farbmacher, H., Guber, R. and Klaaßen, S. (2020). Instrument Validity Tests with Causal Forests. MEA Discussion
    Paper No. 13-2020. Available at SSRN: http://dx.doi.org/10.2139/ssrn.3619201.
    """
    # inspired by https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3619201
    xx = np.random.multivariate_normal(
        np.zeros(2),
        np.array([[1.0, 0.3], [0.3, 1.0]]),
        size=[
            n_obs,
        ],
    )
    u = xx[:, 0]
    v = xx[:, 1]

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]

    z = np.random.binomial(
        p=0.5,
        n=1,
        size=[
            n_obs,
        ],
    )
    d = 1.0 * (alpha_x * z + v > 0)
    y = d * theta + np.dot(x, beta) + u

    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)), columns=x_cols + ["y", "d", "z"])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, "y", "d", x_cols, "z")
    else:
        raise ValueError("Invalid return_type.")
