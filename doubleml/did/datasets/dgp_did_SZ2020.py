import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from ...data.base_data import DoubleMLData
from ...data.panel_data import DoubleMLPanelData
from ...utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def _generate_features(n_obs, c, dim_x=4):
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=n_obs)

    z_tilde_1 = np.exp(0.5 * x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0] * x[:, 2] / 25) ** 3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3]) ** 2

    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)

    return x, z


def _select_features(dgp_type, x, z):
    if dgp_type == 1:
        features_ps = z
        features_reg = z
    elif dgp_type == 2:
        features_ps = x
        features_reg = z
    elif dgp_type == 3:
        features_ps = z
        features_reg = x
    elif dgp_type == 4:
        features_ps = x
        features_reg = x
    elif dgp_type == 5:
        features_ps = None
        features_reg = z
    elif dgp_type == 6:
        features_ps = None
        features_reg = x
    else:
        raise ValueError("The dgp_type is not valid.")
    return features_ps, features_reg


def _f_reg(w):
    res = 210 + 27.4 * w[:, 0] + 13.7 * (w[:, 1] + w[:, 2] + w[:, 3])
    return res


def _f_ps(w, xi):
    res = xi * (-w[:, 0] + 0.5 * w[:, 1] - 0.25 * w[:, 2] - 0.1 * w[:, 3])
    return res


def make_did_SZ2020(n_obs=500, dgp_type=1, cross_sectional_data=False, return_type="DoubleMLData", **kwargs):
    """
    Generates data from a difference-in-differences model used in Sant'Anna and Zhao (2020).
    The data generating process is defined as follows. For a generic :math:`W=(W_1, W_2, W_3, W_4)^T`, let

    .. math::

        f_{reg}(W) &= 210 + 27.4 \\cdot W_1 +13.7 \\cdot (W_2 + W_3 + W_4),

        f_{ps}(W) &= 0.75 \\cdot (-W_1 + 0.5 \\cdot W_2 -0.25 \\cdot W_3 - 0.1 \\cdot W_4).


    Let :math:`X= (X_1, X_2, X_3, X_4)^T \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = c^{|j-k|}`. The default value is  :math:`c = 0`, corresponding to the identity matrix.
    Further, define :math:`Z_j = (\\tilde{Z_j} - \\mathbb{E}[\\tilde{Z}_j]) / \\sqrt{\\text{Var}(\\tilde{Z}_j)}`,
    where :math:`\\tilde{Z}_1 = \\exp(0.5 \\cdot X_1)`, :math:`\\tilde{Z}_2 = 10 + X_2/(1 + \\exp(X_1))`,
    :math:`\\tilde{Z}_3 = (0.6 + X_1 \\cdot X_3 / 25)^3` and :math:`\\tilde{Z}_4 = (20 + X_2 + X_4)^2`.
    At first define

    .. math::

        Y_0(0) &= f_{reg}(W_{reg}) + \\nu(W_{reg}, D) + \\varepsilon_0,

        Y_1(d) &= 2 \\cdot f_{reg}(W_{reg}) + \\nu(W_{reg}, D) + \\varepsilon_1(d),

        p(W_{ps}) &= \\frac{\\exp(f_{ps}(W_{ps}))}{1 + \\exp(f_{ps}(W_{ps}))},

        D &= 1\\{p(W_{ps}) \\ge U\\},

    where :math:`\\varepsilon_0, \\varepsilon_1(d), d=0, 1` are independent standard normal random variables,
    :math:`U \\sim \\mathcal{U}[0, 1]` is a independent standard uniform
    and :math:`\\nu(W_{reg}, D)\\sim \\mathcal{N}(D \\cdot f_{reg}(W_{reg}),1)`.
    The different data generating processes are defined via

    .. math::

        DGP1:\\quad W_{reg} &= Z \\quad W_{ps} = Z

        DGP2:\\quad W_{reg} &= Z \\quad W_{ps} = X

        DGP3:\\quad W_{reg} &= X \\quad W_{ps} = Z

        DGP4:\\quad W_{reg} &= X \\quad W_{ps} = X

        DGP5:\\quad W_{reg} &= Z \\quad W_{ps} = 0

        DGP6:\\quad W_{reg} &= X \\quad W_{ps} = 0,

    such that the last two settings correspond to an experimental setting with treatment probability
    of :math:`P(D=1) = \\frac{1}{2}.`
    For the panel data the outcome is already defined as the difference :math:`Y = Y_1(D) - Y_0(0)`.
    For cross-sectional data the flag ``cross_sectional_data`` has to be set to ``True``.
    Then the outcome will be defined to be

    .. math::

        Y = T \\cdot Y_1(D) + (1-T) \\cdot Y_0(0),

    where :math:`T = 1\\{U_T\\le \\lambda_T \\}` with :math:`U_T\\sim \\mathcal{U}[0, 1]` and :math:`\\lambda_T=0.5`.
    The true average treatment effect on the treated is zero for all data generating processes.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dgp_type :
        The DGP to be used. Default value is ``1`` (integer).
    cross_sectional_data :
        Indicates whether the setting is uses cross-sectional or panel data. Default value is ``False``.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``
        or ``(x, y, d, t)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameter
        :math:`xi=0.75`, :math:`c=0.0` and :math:`\\lambda_T=0.5`.

    References
    ----------
    Sant’Anna, P. H. and Zhao, J. (2020),
    Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
    doi:`10.1016/j.jeconom.2020.06.003 <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.
    """
    xi = kwargs.get("xi", 0.75)
    c = kwargs.get("c", 0.0)
    lambda_t = kwargs.get("lambda_t", 0.5)

    dim_x = 4
    x, z = _generate_features(n_obs, c, dim_x=dim_x)

    # error terms
    epsilon_0 = np.random.normal(loc=0, scale=1, size=n_obs)
    epsilon_1 = np.random.normal(loc=0, scale=1, size=[n_obs, 2])

    features_ps, features_reg = _select_features(dgp_type, x, z)

    # treatment and propensities
    is_experimental = (dgp_type == 5) or (dgp_type == 6)
    if is_experimental:
        # Set D to be experimental
        p = 0.5 * np.ones(n_obs)
    else:
        p = np.exp(_f_ps(features_ps, xi)) / (1 + np.exp(_f_ps(features_ps, xi)))
    u = np.random.uniform(low=0, high=1, size=n_obs)
    d = 1.0 * (p >= u)

    # potential outcomes
    nu = np.random.normal(loc=d * _f_reg(features_reg), scale=1, size=n_obs)
    y0 = _f_reg(features_reg) + nu + epsilon_0
    y1_d0 = 2 * _f_reg(features_reg) + nu + epsilon_1[:, 0]
    y1_d1 = 2 * _f_reg(features_reg) + nu + epsilon_1[:, 1]
    y1 = d * y1_d1 + (1 - d) * y1_d0

    if not cross_sectional_data:
        y = y1 - y0

        if return_type in _array_alias:
            return z, y, d, None
        elif return_type in _data_frame_alias + _dml_data_alias:
            z_cols = [f"Z{i + 1}" for i in np.arange(dim_x)]
            data = pd.DataFrame(np.column_stack((z, y, d)), columns=z_cols + ["y", "d"])
            if return_type in _data_frame_alias:
                return data
            else:
                return DoubleMLData(data, "y", "d", z_cols)
        elif return_type == "DoubleMLPanelData":
            z_cols = [f"Z{i + 1}" for i in np.arange(dim_x)]
            df0 = (
                pd.DataFrame(
                    {
                        "y": y0,
                        "d": d.astype(np.int32),
                        "t": np.zeros_like(y0, dtype=np.int32),
                        **{col: z[:, i] for i, col in enumerate(z_cols)},
                    }
                )
                .reset_index()
                .rename(columns={"index": "id"})
            )
            df1 = (
                pd.DataFrame(
                    {
                        "y": y1,
                        "d": d.astype(np.int32),
                        "t": np.ones_like(y0, dtype=np.int32),
                        **{col: z[:, i] for i, col in enumerate(z_cols)},
                    }
                )
                .reset_index()
                .rename(columns={"index": "id"})
            )
            df = pd.concat([df0, df1], axis=0)

            return DoubleMLPanelData(df, "y", "d", t_col="t", id_col="id", x_cols=z_cols)
        else:
            raise ValueError("Invalid return_type.")

    else:
        u_t = np.random.uniform(low=0, high=1, size=n_obs)
        t = 1.0 * (u_t <= lambda_t)
        y = t * y1 + (1 - t) * y0

        if return_type in _array_alias:
            return z, y, d, t
        elif return_type in _data_frame_alias + _dml_data_alias:
            z_cols = [f"Z{i + 1}" for i in np.arange(dim_x)]
            data = pd.DataFrame(np.column_stack((z, y, d, t)), columns=z_cols + ["y", "d", "t"])
            if return_type in _data_frame_alias:
                return data
            else:
                return DoubleMLData(data, "y", "d", z_cols, t_col="t")
        else:
            raise ValueError("Invalid return_type.")
