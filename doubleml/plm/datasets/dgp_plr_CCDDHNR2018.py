import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, return_type="DoubleMLData", **kwargs):
    """
    Generates data from a partially linear regression model used in Chernozhukov et al. (2018) for Figure 1.
    The data generating process is defined as

    .. math::

        d_i &= m_0(x_i) + s_1 v_i, & &v_i \\sim \\mathcal{N}(0,1),

        y_i &= \\alpha d_i + g_0(x_i) + s_2 \\zeta_i, & &\\zeta_i \\sim \\mathcal{N}(0,1),


    with covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.7^{|j-k|}`.
    The nuisance functions are given by

    .. math::

        m_0(x_i) &= a_0 x_{i,1} + a_1 \\frac{\\exp(x_{i,3})}{1+\\exp(x_{i,3})},

        g_0(x_i) &= b_0 \\frac{\\exp(x_{i,1})}{1+\\exp(x_{i,1})} + b_1 x_{i,3}.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    alpha :
        The value of the causal parameter.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameters
        :math:`a_0=1`, :math:`a_1=0.25`, :math:`s_1=1`, :math:`b_0=1`, :math:`b_1=0.25` or :math:`s_2=1`.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    a_0 = kwargs.get("a_0", 1.0)
    a_1 = kwargs.get("a_1", 0.25)
    s_1 = kwargs.get("s_1", 1.0)

    b_0 = kwargs.get("b_0", 1.0)
    b_1 = kwargs.get("b_1", 0.25)
    s_2 = kwargs.get("s_2", 1.0)

    cov_mat = toeplitz([np.power(0.7, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    d = (
        a_0 * x[:, 0]
        + a_1 * np.divide(np.exp(x[:, 2]), 1 + np.exp(x[:, 2]))
        + s_1
        * np.random.standard_normal(
            size=[
                n_obs,
            ]
        )
    )
    y = (
        alpha * d
        + b_0 * np.divide(np.exp(x[:, 0]), 1 + np.exp(x[:, 0]))
        + b_1 * x[:, 2]
        + s_2
        * np.random.standard_normal(
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
