import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def _g(x):
    return np.power(np.sin(x), 2)


def _m(x, nu=0.0, gamma=1.0):
    return 0.5 / np.pi * (np.sinh(gamma)) / (np.cosh(gamma) - np.cos(x - nu))


def make_plr_turrell2018(n_obs=100, dim_x=20, theta=0.5, return_type="DoubleMLData", **kwargs):
    """
    Generates data from a partially linear regression model used in a blog article by Turrell (2018).
    The data generating process is defined as

    .. math::

        d_i &= m_0(x_i' b) + v_i, & &v_i \\sim \\mathcal{N}(0,1),

        y_i &= \\theta d_i + g_0(x_i' b) + u_i, & &u_i \\sim \\mathcal{N}(0,1),


    with covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a random symmetric,
    positive-definite matrix generated with :py:meth:`sklearn.datasets.make_spd_matrix`.
    :math:`b` is a vector with entries :math:`b_j=\\frac{1}{j}` and the nuisance functions are given by

    .. math::

        m_0(x_i) &= \\frac{1}{2 \\pi} \\frac{\\sinh(\\gamma)}{\\cosh(\\gamma) - \\cos(x_i-\\nu)},

        g_0(x_i) &= \\sin(x_i)^2.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameters
        :math:`\\nu=0`, or :math:`\\gamma=1`.

    References
    ----------
    Turrell, A. (2018), Econometrics in Python part I - Double machine learning, Markov Wanderer: A blog on economics,
    science, coding and data. `https://aeturrell.com/blog/posts/econometrics-in-python-parti-ml/
    <https://aeturrell.com/blog/posts/econometrics-in-python-parti-ml/>`_.
    """
    nu = kwargs.get("nu", 0.0)
    gamma = kwargs.get("gamma", 1.0)

    b = [1 / k for k in range(1, dim_x + 1)]
    sigma = make_spd_matrix(dim_x)

    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        sigma,
        size=[
            n_obs,
        ],
    )
    G = _g(np.dot(x, b))
    M = _m(np.dot(x, b), nu=nu, gamma=gamma)
    d = M + np.random.standard_normal(
        size=[
            n_obs,
        ]
    )
    y = (
        np.dot(theta, d)
        + G
        + np.random.standard_normal(
            size=[
                n_obs,
            ]
        )
    )

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
