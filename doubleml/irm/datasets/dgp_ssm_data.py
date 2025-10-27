import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLSSMData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_ssm_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_ssm_data_alias = _get_dml_ssm_data_alias()


def make_ssm_data(n_obs=8000, dim_x=100, theta=1, mar=True, return_type="DoubleMLSSMData"):
    """
    Generates data from a sample selection model (SSM).
    The data generating process is defined as

    .. math::

        y_i &= \\theta d_i + x_i' \\beta d_i + u_i,

        s_i &= 1\\left\\lbrace d_i + \\gamma z_i + x_i' \\beta + v_i > 0 \\right\\rbrace,

        d_i &= 1\\left\\lbrace x_i' \\beta + w_i > 0 \\right\\rbrace,

    with Y being observed if :math:`s_i = 1` and covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma^2_x)`, where
    :math:`\\Sigma^2_x` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}`.
    :math:`\\beta` is a `dim_x`-vector with entries :math:`\\beta_j=\\frac{0.4}{j^2}`
    :math:`z_i \\sim \\mathcal{N}(0, 1)`,
    :math:`(u_i,v_i) \\sim \\mathcal{N}(0, \\Sigma^2_{u,v})`,
    :math:`w_i \\sim \\mathcal{N}(0, 1)`.


    The data generating process is inspired by a process used in the simulation study (see Appendix E) of Bia,
    Huber and Lafférs (2023).

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    mar:
        Boolean. Indicates whether missingness at random holds.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d, z, s)``.

    References
    ----------
    Michela Bia, Martin Huber & Lukáš Lafférs (2023) Double Machine Learning for Sample Selection Models,
    Journal of Business & Economic Statistics, DOI: 10.1080/07350015.2023.2271071
    """
    if mar:
        sigma = np.array([[1, 0], [0, 1]])
        gamma = 0
    else:
        sigma = np.array([[1, 0.8], [0.8, 1]])
        gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    beta = [0.4 / (k**2) for k in range(1, dim_x + 1)]

    d = np.where(np.dot(x, beta) + np.random.randn(n_obs) > 0, 1, 0)
    z = np.random.randn(n_obs)
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    y = np.dot(x, beta) + theta * d + e[1]
    y[s == 0] = 0

    if return_type in _array_alias:
        return x, y, d, z, s
    elif return_type in _data_frame_alias + _dml_ssm_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        if mar:
            data = pd.DataFrame(np.column_stack((x, y, d, s)), columns=x_cols + ["y", "d", "s"])
        else:
            data = pd.DataFrame(np.column_stack((x, y, d, z, s)), columns=x_cols + ["y", "d", "z", "s"])
        if return_type in _data_frame_alias:
            return data
        else:
            if mar:
                return DoubleMLSSMData(data, "y", "d", x_cols, z_cols=None, s_col="s")
            return DoubleMLSSMData(data, "y", "d", x_cols, z_cols="z", s_col="s")
    else:
        raise ValueError("Invalid return_type.")
