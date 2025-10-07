import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _array_alias, _data_frame_alias, _dml_data_alias


def make_pliv_multiway_cluster_CKMS2021(N=25, M=25, dim_X=100, theta=1.0, return_type="DoubleMLData", **kwargs):
    """
    Generates data from a partially linear IV regression model with multiway cluster sample used in Chiang et al.
    (2021). The data generating process is defined as

    .. math::

        Z_{ij} &= X_{ij}' \\xi_0 + V_{ij},

        D_{ij} &= Z_{ij}' \\pi_{10} + X_{ij}' \\pi_{20} + v_{ij},

        Y_{ij} &= D_{ij} \\theta + X_{ij}' \\zeta_0 + \\varepsilon_{ij},

    with

    .. math::

        X_{ij} &= (1 - \\omega_1^X - \\omega_2^X) \\alpha_{ij}^X
        + \\omega_1^X \\alpha_{i}^X + \\omega_2^X \\alpha_{j}^X,

        \\varepsilon_{ij} &= (1 - \\omega_1^\\varepsilon - \\omega_2^\\varepsilon) \\alpha_{ij}^\\varepsilon
        + \\omega_1^\\varepsilon \\alpha_{i}^\\varepsilon + \\omega_2^\\varepsilon \\alpha_{j}^\\varepsilon,

        v_{ij} &= (1 - \\omega_1^v - \\omega_2^v) \\alpha_{ij}^v
        + \\omega_1^v \\alpha_{i}^v + \\omega_2^v \\alpha_{j}^v,

        V_{ij} &= (1 - \\omega_1^V - \\omega_2^V) \\alpha_{ij}^V
        + \\omega_1^V \\alpha_{i}^V + \\omega_2^V \\alpha_{j}^V,

    and :math:`\\alpha_{ij}^X, \\alpha_{i}^X, \\alpha_{j}^X \\sim \\mathcal{N}(0, \\Sigma)`
    where  :math:`\\Sigma` is a :math:`p_x \\times p_x` matrix with entries
    :math:`\\Sigma_{kj} = s_X^{|j-k|}`.
    Further

    .. math::

        \\left(\\begin{matrix} \\alpha_{ij}^\\varepsilon \\\\ \\alpha_{ij}^v \\end{matrix}\\right),
        \\left(\\begin{matrix} \\alpha_{i}^\\varepsilon \\\\ \\alpha_{i}^v \\end{matrix}\\right),
        \\left(\\begin{matrix} \\alpha_{j}^\\varepsilon \\\\ \\alpha_{j}^v \\end{matrix}\\right)
        \\sim \\mathcal{N}\\left(0, \\left(\\begin{matrix} 1 & s_{\\varepsilon v} \\\\
        s_{\\varepsilon v} & 1 \\end{matrix} \\right) \\right)


    and :math:`\\alpha_{ij}^V, \\alpha_{i}^V, \\alpha_{j}^V \\sim \\mathcal{N}(0, 1)`.

    Parameters
    ----------
    N :
        The number of observations (first dimension).
    M :
        The number of observations (second dimension).
    dim_X :
        The number of covariates.
    theta :
        The value of the causal parameter.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object where
        ``DoubleMLData.data`` is a ``pd.DataFrame``.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s
        ``(x, y, d, cluster_vars, z)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameters
        :math:`\\pi_{10}=1.0`, :math:`\\omega_X = \\omega_{\\varepsilon} = \\omega_V = \\omega_v = (0.25, 0.25)`,
        :math:`s_X = s_{\\varepsilon v} = 0.25`,
        or the :math:`p_x`-vectors :math:`\\zeta_0 = \\pi_{20} = \\xi_0` with default entries
        :math:`(\\zeta_{0})_j = 0.5^j`.

    References
    ----------
    Chiang, H. D., Kato K., Ma, Y. and Sasaki, Y. (2021), Multiway Cluster Robust Double/Debiased Machine Learning,
    Journal of Business & Economic Statistics,
    doi: `10.1080/07350015.2021.1895815 <https://doi.org/10.1080/07350015.2021.1895815>`_,
    arXiv:`1909.03489 <https://arxiv.org/abs/1909.03489>`_.
    """
    # additional parameters specifiable via kwargs
    pi_10 = kwargs.get("pi_10", 1.0)

    xx = np.arange(1, dim_X + 1)
    zeta_0 = kwargs.get("zeta_0", np.power(0.5, xx))
    pi_20 = kwargs.get("pi_20", np.power(0.5, xx))
    xi_0 = kwargs.get("xi_0", np.power(0.5, xx))

    omega_X = kwargs.get("omega_X", np.array([0.25, 0.25]))
    omega_epsilon = kwargs.get("omega_epsilon", np.array([0.25, 0.25]))
    omega_v = kwargs.get("omega_v", np.array([0.25, 0.25]))
    omega_V = kwargs.get("omega_V", np.array([0.25, 0.25]))

    s_X = kwargs.get("s_X", 0.25)
    s_epsilon_v = kwargs.get("s_epsilon_v", 0.25)

    # use np.tile() and np.repeat() for repeating vectors in different styles, i.e.,
    # np.tile([v1, v2, v3], 2) [v1, v2, v3, v1, v2, v3]
    # np.repeat([v1, v2, v3], 2) [v1, v1, v2, v2, v3, v3]

    alpha_V = np.random.normal(size=(N * M))
    alpha_V_i = np.repeat(np.random.normal(size=N), M)
    alpha_V_j = np.tile(np.random.normal(size=M), N)

    cov_mat = np.array([[1, s_epsilon_v], [s_epsilon_v, 1]])
    alpha_eps_v = np.random.multivariate_normal(
        np.zeros(2),
        cov_mat,
        size=[
            N * M,
        ],
    )
    alpha_eps = alpha_eps_v[:, 0]
    alpha_v = alpha_eps_v[:, 1]

    alpha_eps_v_i = np.random.multivariate_normal(
        np.zeros(2),
        cov_mat,
        size=[
            N,
        ],
    )
    alpha_eps_i = np.repeat(alpha_eps_v_i[:, 0], M)
    alpha_v_i = np.repeat(alpha_eps_v_i[:, 1], M)

    alpha_eps_v_j = np.random.multivariate_normal(
        np.zeros(2),
        cov_mat,
        size=[
            M,
        ],
    )
    alpha_eps_j = np.tile(alpha_eps_v_j[:, 0], N)
    alpha_v_j = np.tile(alpha_eps_v_j[:, 1], N)

    cov_mat = toeplitz([np.power(s_X, k) for k in range(dim_X)])
    alpha_X = np.random.multivariate_normal(
        np.zeros(dim_X),
        cov_mat,
        size=[
            N * M,
        ],
    )
    alpha_X_i = np.repeat(
        np.random.multivariate_normal(
            np.zeros(dim_X),
            cov_mat,
            size=[
                N,
            ],
        ),
        M,
        axis=0,
    )
    alpha_X_j = np.tile(
        np.random.multivariate_normal(
            np.zeros(dim_X),
            cov_mat,
            size=[
                M,
            ],
        ),
        (N, 1),
    )

    # generate variables
    x = (1 - omega_X[0] - omega_X[1]) * alpha_X + omega_X[0] * alpha_X_i + omega_X[1] * alpha_X_j

    eps = (
        (1 - omega_epsilon[0] - omega_epsilon[1]) * alpha_eps + omega_epsilon[0] * alpha_eps_i + omega_epsilon[1] * alpha_eps_j
    )

    v = (1 - omega_v[0] - omega_v[1]) * alpha_v + omega_v[0] * alpha_v_i + omega_v[1] * alpha_v_j

    V = (1 - omega_V[0] - omega_V[1]) * alpha_V + omega_V[0] * alpha_V_i + omega_V[1] * alpha_V_j

    z = np.matmul(x, xi_0) + V
    d = z * pi_10 + np.matmul(x, pi_20) + v
    y = d * theta + np.matmul(x, zeta_0) + eps

    cluster_cols = ["cluster_var_i", "cluster_var_j"]
    cluster_vars = pd.MultiIndex.from_product([range(N), range(M)]).to_frame(name=cluster_cols).reset_index(drop=True)

    if return_type in _array_alias:
        return x, y, d, cluster_vars.values, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_X)]
        data = pd.concat((cluster_vars, pd.DataFrame(np.column_stack((x, y, d, z)), columns=x_cols + ["Y", "D", "Z"])), axis=1)
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, y_col="Y", d_cols="D", cluster_cols=cluster_cols, x_cols=x_cols, z_cols="Z")
    else:
        raise ValueError("Invalid return_type.")
