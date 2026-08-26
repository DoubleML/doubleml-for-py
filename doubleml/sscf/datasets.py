"""Data generating processes for the sample selection control function model."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from doubleml.data.base_data import DoubleMLData

_ARRAY_ALIAS = ["array", "np.ndarray", "np.array", np.ndarray]
_DATA_FRAME_ALIAS = ["DataFrame", "pd.DataFrame", pd.DataFrame]
_DML_DATA_ALIAS = ["DoubleMLData", DoubleMLData]
_DICT_ALIAS = ["dict", dict]


def decile_grid(n_deciles):
    r"""Representative points of :math:`L` characteristic-sorted portfolios.

    For a standardized characteristic the representative point of the :math:`\ell`-th
    portfolio is the midpoint of the corresponding quantile bin,
    :math:`P_\ell = \Phi^{-1}\big((\ell - 0.5)/L\big)`, :math:`\ell = 1, \dots, L`.
    """
    ell = np.arange(1, n_deciles + 1)
    return norm.ppf((ell - 0.5) / n_deciles)


def gaussian_kernel_features(x, grid, bandwidth=0.5):
    r"""Gaussian kernel features :math:`k(X_{ij}, P_{j,\ell})` of a characteristic matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Characteristics of dimension ``(n_obs, dim_x)``.

    grid : numpy.ndarray
        Representative portfolio points :math:`P_{j,\ell}` of dimension ``(n_deciles,)``
        (shared across characteristics) or ``(dim_x, n_deciles)``.

    bandwidth : float
        Bandwidth :math:`\sigma_k` of the Gaussian kernel
        :math:`k(u, v) = \exp\{-(u - v)^2 / (2\sigma_k^2)\}`. Default is ``0.5``.

    Returns
    -------
    k : numpy.ndarray
        Kernel design matrix of dimension ``(n_obs, dim_x * n_deciles)``, with the
        rows of the ``(dim_x, n_deciles)`` kernel matrix stacked on top of each other,
        i.e. columns are ordered as
        :math:`(j, \ell) = (1,1), \dots, (1,L), (2,1), \dots, (J,L)`.

    groups : numpy.ndarray
        Group index of length ``dim_x * n_deciles`` mapping each column to its
        characteristic :math:`j`.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n_obs, dim_x = x.shape
    if grid.ndim == 1:
        grid = np.tile(grid, (dim_x, 1))
    n_deciles = grid.shape[1]

    k = np.empty((n_obs, dim_x * n_deciles))
    for j in range(dim_x):
        diff = x[:, j][:, None] - grid[j][None, :]
        k[:, j * n_deciles : (j + 1) * n_deciles] = np.exp(-0.5 * (diff / bandwidth) ** 2)
    groups = np.repeat(np.arange(dim_x), n_deciles)
    return k, groups


def kernel_gram_matrices(grid, bandwidth=0.5, dim_x=None):
    r"""Kernel matrices :math:`[\mathsf{K}_j]_{\ell, \ell'} = k(P_{j,\ell}, P_{j,\ell'})/L`.

    Returns a list of ``dim_x`` matrices of dimension ``(n_deciles, n_deciles)`` which
    define the group norms :math:`\|b_j\|_{\mathsf{K}_j} = (b_j' \mathsf{K}_j b_j)^{1/2}`
    of the kernel group lasso penalty.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim == 1:
        if dim_x is None:
            raise ValueError("dim_x has to be provided if a single grid is passed.")
        grid = np.tile(grid, (dim_x, 1))
    n_deciles = grid.shape[1]
    mats = []
    for j in range(grid.shape[0]):
        diff = grid[j][:, None] - grid[j][None, :]
        mats.append(np.exp(-0.5 * (diff / bandwidth) ** 2) / n_deciles)
    return mats


def _toeplitz_cov(dim, rho=0.5):
    idx = np.arange(dim)
    return np.power(rho, np.abs(idx[:, None] - idx[None, :]))


def make_green_silence_data(
    n_obs=4000,
    dim_x=50,
    dim_z=10,
    n_deciles=10,
    theta=-0.5,
    sigma_epsilon=1.0,
    n_active_outcome=3,
    n_active_selection_x=2,
    n_active_selection_z=5,
    beta_x=0.4,
    beta_z=0.8,
    selection_rate=0.3,
    bandwidth=0.5,
    x_corr=0.5,
    disjoint_active_sets=True,
    nonlinear_index=False,
    return_type="DoubleMLData",
):
    r"""Generate data from the high-dimensional sample selection ("green silence") DGP.

    The DGP mimics the empirical setting of corporate carbon disclosure: emissions are
    observed only for firms which choose to disclose, and disclosure is driven by
    unobservables that are correlated with emissions.

    Outcome (variable selection) equation, additive in Gaussian kernel features of the
    characteristic-sorted portfolios,

    .. math::

        Y_i = \sum_{j=1}^{J}\sum_{\ell=1}^{L} b_{0,j\ell}\, k(X_{ij}, P_{j\ell}) + \varepsilon_i
            \equiv \mathbf{k}_i b_0 + \varepsilon_i,

    selection (disclosure) equation with :math:`Z_i = (X_i, U_i)` and index function
    :math:`f_0`,

    .. math::

        D_i = \mathbb{1}\{f_0(Z_i) + c + v_i > 0\},
        \qquad
        f_0(Z_i) = \begin{cases}
        Z_i'\beta_0 & \text{if } \texttt{nonlinear\_index=False},\\
        0.6\, Z_i'\beta_0 + \text{non-linear terms} & \text{otherwise.}
        \end{cases}

    and jointly normal errors

    .. math::

        \begin{pmatrix}\varepsilon_i \\ v_i\end{pmatrix} \sim
        \mathcal{N}\left(0, \begin{pmatrix} \sigma_\varepsilon^2 & \theta_0 \\
        \theta_0 & 1\end{pmatrix}\right),

    so that :math:`\theta_0 = \sigma_{\varepsilon v}` is the sample selection bias
    coefficient and :math:`\mathbb{E}[\varepsilon_i \mid Z_i, D_i = 1] = \theta_0 h_i`
    with :math:`h_i = \varphi(Z_i'\beta_0 + c)/\Phi(Z_i'\beta_0 + c)`. The outcome is
    observed only for :math:`D_i = 1`. The intercept :math:`c` is calibrated such that
    :math:`\mathbb{E}[D_i] =` ``selection_rate``.

    The active set of the outcome equation and the active set of the selection equation
    are (by default) disjoint, in line with the high-dimensional exclusion restriction:
    the variables :math:`U_i` and a few characteristics shift participation but do not
    enter the outcome equation.

    Parameters
    ----------
    n_obs : int
        Number of observations (firms) :math:`N`. Default is ``4000``.

    dim_x : int
        Number of characteristics :math:`J`. Default is ``50``.

    dim_z : int
        Number of additional selection variables :math:`U` (candidate exclusion
        restrictions). Default is ``10``.

    n_deciles : int
        Number of characteristic-sorted portfolios :math:`L`. Default is ``10``.

    theta : float
        True sample selection bias coefficient :math:`\theta_0 = \sigma_{\varepsilon v}`.
        Negative values correspond to unobservables which increase disclosure and lower
        emissions. Default is ``-0.5``.

    sigma_epsilon : float
        Standard deviation of the outcome error. Has to satisfy
        ``abs(theta) < sigma_epsilon`` for a valid covariance matrix. Default is ``1.0``.

    n_active_outcome : int
        Number of characteristics with non-zero coefficient group :math:`b_{0,j}`.
        Default is ``3``.

    n_active_selection_x : int
        Number of characteristics entering the selection equation. Default is ``2``.

    n_active_selection_z : int
        Number of variables :math:`U` entering the selection equation. Default is ``5``.

    beta_x, beta_z : float
        Selection coefficients of the active characteristics and of the active
        :math:`U` variables. Defaults are ``0.4`` and ``0.8``.

    selection_rate : float
        Target unconditional participation (disclosure) rate. Default is ``0.3``.

    bandwidth : float
        Bandwidth of the Gaussian kernel. Default is ``0.5``.

    x_corr : float
        Autocorrelation of the Toeplitz covariance matrix of the characteristics,
        :math:`\Sigma_{jk} = \rho^{|j-k|}`. Default is ``0.5``.

    disjoint_active_sets : bool
        If ``True`` the characteristics entering the selection equation are disjoint
        from those entering the outcome equation. Default is ``True``.

    nonlinear_index : bool
        If ``True`` the selection index :math:`f_0` is a non-linear function of
        :math:`Z`, combining the linear index with sine, quadratic, interaction and
        absolute-value terms in the active variables. A correctly specified probit or
        linear logistic learner is then misspecified, so this option is useful to
        demonstrate that the control function correction does not rely on a linear
        index. Default is ``False``.

    return_type : str
        ``'DoubleMLData'`` (default), ``'DataFrame'``, ``'array'`` or ``'dict'``. The
        option ``'dict'`` additionally returns the true parameters, which is needed to
        evaluate variable selection.

    Returns
    -------
    data : :class:`doubleml.DoubleMLData`, :class:`pandas.DataFrame`, tuple or dict
        The generated data. For ``return_type='dict'`` the returned dictionary contains
        the ``DoubleMLData`` object under key ``'dml_data'`` together with the true
        parameters (``'b'``, ``'beta'``, ``'theta'``, ``'active_outcome'``,
        ``'active_selection'``), the kernel design matrix ``'k'``, the group index
        ``'groups'``, the kernel matrices ``'K_mats'``, the true inverse Mills ratios
        ``'imr'`` and the latent outcomes ``'y_latent'``.
    """
    if abs(theta) >= sigma_epsilon:
        raise ValueError("The covariance matrix requires abs(theta) < sigma_epsilon.")
    if n_active_outcome > dim_x:
        raise ValueError("n_active_outcome must not exceed dim_x.")
    if n_active_selection_z > dim_z:
        raise ValueError("n_active_selection_z must not exceed dim_z.")

    # ------------------------------------------------------------------ covariates
    cov_x = _toeplitz_cov(dim_x, x_corr)
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_x, size=n_obs)
    u = np.random.normal(size=(n_obs, dim_z))

    grid = decile_grid(n_deciles)
    k, groups = gaussian_kernel_features(x, grid, bandwidth=bandwidth)
    k_mats = kernel_gram_matrices(grid, bandwidth=bandwidth, dim_x=dim_x)

    # ------------------------------------------------------- outcome coefficients b
    active_outcome = np.arange(n_active_outcome)
    # group specific loading profiles across the L sorted portfolios (monotone,
    # u-shaped and hump-shaped), scaled to unit group norm times a group weight
    profiles = np.vstack(
        [
            np.linspace(-1.0, 1.0, n_deciles),
            np.linspace(-1.0, 1.0, n_deciles) ** 2 - 0.5,
            np.sin(np.linspace(0.0, np.pi, n_deciles)),
        ]
    )
    group_weights = np.array([1.5, 1.2, 1.0])

    b = np.zeros(dim_x * n_deciles)
    for i_active, j in enumerate(active_outcome):
        profile = profiles[i_active % profiles.shape[0]]
        profile = profile / np.linalg.norm(profile)
        b[j * n_deciles : (j + 1) * n_deciles] = group_weights[i_active % group_weights.shape[0]] * profile

    # ----------------------------------------------------- selection coefficients beta
    if disjoint_active_sets:
        candidates = np.setdiff1d(np.arange(dim_x), active_outcome)
    else:
        candidates = np.arange(dim_x)
    if n_active_selection_x > candidates.shape[0]:
        raise ValueError("Not enough characteristics left for disjoint active sets.")
    active_selection_x = candidates[:n_active_selection_x]
    active_selection_z = np.arange(n_active_selection_z)

    beta = np.zeros(dim_x + dim_z)
    beta[active_selection_x] = beta_x
    beta[dim_x + active_selection_z] = beta_z

    # --------------------------------------------------------------------- errors
    cov_e = np.array([[sigma_epsilon**2, theta], [theta, 1.0]])
    errors = np.random.multivariate_normal(np.zeros(2), cov_e, size=n_obs)
    epsilon, v = errors[:, 0], errors[:, 1]

    # ------------------------------------------------- selection with calibrated rate
    z_full = np.column_stack((x, u))
    index = z_full @ beta
    if nonlinear_index:
        # a genuinely non-linear f_0(Z): the linear index is attenuated and combined with
        # smooth, quadratic and interaction terms in the active selection variables
        j0, j1 = active_selection_x[0], active_selection_x[-1]
        l0 = active_selection_z[0]
        l1 = active_selection_z[1 % active_selection_z.shape[0]]
        l2 = active_selection_z[2 % active_selection_z.shape[0]]
        l3 = active_selection_z[3 % active_selection_z.shape[0]]
        index_linear = index
        index = (
            0.6 * index_linear
            + 1.1 * np.sin(1.5 * u[:, l0])
            + 0.9 * (u[:, l1] ** 2 - 1.0)
            + 1.0 * u[:, l2] * u[:, l3]
            + 0.8 * np.abs(x[:, j0])
            + 0.5 * x[:, j1] ** 2
        )
        # rescale so that the dispersion of the index - and hence the degree of overlap -
        # is comparable to the linear design; otherwise the non-linear terms fatten the
        # tails of the propensity score and trimming, rather than misspecification,
        # would drive any difference between the two designs
        index = index * (index_linear.std() / index.std())

    def _rate(const):
        return np.mean(norm.cdf(index + const)) - selection_rate

    const = brentq(_rate, -20.0, 20.0)
    index = index + const
    d = (index + v > 0).astype(int)

    imr = norm.pdf(index) / np.clip(norm.cdf(index), 1e-12, None)

    # -------------------------------------------------------------------- outcomes
    y_latent = k @ b + epsilon
    y = np.where(d == 1, y_latent, 0.0)

    x_cols = [f"X{i + 1}" for i in range(dim_x)]
    k_cols = [f"k{j + 1}_{ell + 1}" for j in range(dim_x) for ell in range(n_deciles)]
    z_cols = [f"U{i + 1}" for i in range(dim_z)]

    if return_type in _ARRAY_ALIAS:
        return np.column_stack((x, k)), y, d, u

    data = pd.DataFrame(np.column_stack((x, k, u, y, d)), columns=x_cols + k_cols + z_cols + ["y", "d"])
    data["d"] = data["d"].astype(int)

    if return_type in _DATA_FRAME_ALIAS:
        return data

    dml_data = DoubleMLData(data, y_col="y", d_cols="d", x_cols=x_cols + k_cols, z_cols=z_cols)

    if return_type in _DML_DATA_ALIAS:
        return dml_data

    if return_type in _DICT_ALIAS:
        return {
            "dml_data": dml_data,
            "data": data,
            "x": x,
            "u": u,
            "k": k,
            "groups": groups,
            "K_mats": k_mats,
            "grid": grid,
            "b": b,
            "beta": beta,
            "theta": theta,
            "const": const,
            "nonlinear_index": nonlinear_index,
            "active_outcome": active_outcome,
            "active_selection_x": active_selection_x,
            "active_selection_z": active_selection_z,
            "imr": imr,
            "index": index,
            "y_latent": y_latent,
            "y": y,
            "d": d,
            "x_cols": x_cols,
            "k_cols": k_cols,
            "z_cols": z_cols,
        }

    raise ValueError("Invalid return_type.")
