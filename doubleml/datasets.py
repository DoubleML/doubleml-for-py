import pandas as pd
import numpy as np

from scipy.linalg import toeplitz
from scipy.optimize import minimize_scalar

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.datasets import make_spd_matrix

from .double_ml_data import DoubleMLData, DoubleMLClusterData

_array_alias = ['array', 'np.ndarray', 'np.array', np.ndarray]
_data_frame_alias = ['DataFrame', 'pd.DataFrame', pd.DataFrame]
_dml_data_alias = ['DoubleMLData', DoubleMLData]
_dml_cluster_data_alias = ['DoubleMLClusterData', DoubleMLClusterData]


def fetch_401K(return_type='DoubleMLData', polynomial_features=False):
    """
    Data set on financial wealth and 401(k) plan participation.

    Parameters
    ----------
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.
    polynomial_features :
        If ``True`` polynomial features are added (see replication files of Chernozhukov et al. (2018)).

    References
    ----------
    Abadie, A. (2003), Semiparametric instrumental variable estimation of treatment response models. Journal of
    Econometrics, 113(2): 231-263.

    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    url = 'https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta'
    raw_data = pd.read_stata(url)

    y_col = 'net_tfa'
    d_cols = ['e401']
    x_cols = ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']

    data = raw_data.copy()

    if polynomial_features:
        raise NotImplementedError('polynomial_features os not implemented yet for fetch_401K.')

    if return_type in _data_frame_alias + _dml_data_alias:
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, y_col, d_cols, x_cols)
    else:
        raise ValueError('Invalid return_type.')


def fetch_bonus(return_type='DoubleMLData', polynomial_features=False):
    """
    Data set on the Pennsylvania Reemployment Bonus experiment.

    Parameters
    ----------
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.
    polynomial_features :
        If ``True`` polynomial features are added (see replication files of Chernozhukov et al. (2018)).

    References
    ----------
    Bilias Y. (2000), Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment.
    Journal of Applied Econometrics, 15(6): 575-594.

    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    url = 'https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat'
    raw_data = pd.read_csv(url, delim_whitespace=True)

    ind = (raw_data['tg'] == 0) | (raw_data['tg'] == 4)
    data = raw_data.copy()[ind]
    data.reset_index(inplace=True)
    data['tg'].replace(4, 1, inplace=True)
    data['inuidur1'] = np.log(data['inuidur1'])

    # variable dep as factor (dummy encoding)
    dummy_enc = OneHotEncoder(drop='first', categories='auto').fit(data.loc[:, ['dep']])
    xx = dummy_enc.transform(data.loc[:, ['dep']]).toarray()
    data['dep1'] = xx[:, 0]
    data['dep2'] = xx[:, 1]

    y_col = 'inuidur1'
    d_cols = ['tg']
    x_cols = ['female', 'black', 'othrace',
              'dep1', 'dep2',
              'q2', 'q3', 'q4', 'q5', 'q6',
              'agelt35', 'agegt54', 'durable', 'lusd', 'husd']

    if polynomial_features:
        poly = PolynomialFeatures(2, include_bias=False)
        data_transf = poly.fit_transform(data[x_cols])
        x_cols = list(poly.get_feature_names_out(x_cols))

        data_transf = pd.DataFrame(data_transf, columns=x_cols)
        data = pd.concat((data[[y_col] + d_cols], data_transf),
                         axis=1, sort=False)

    if return_type in _data_frame_alias + _dml_data_alias:
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, y_col, d_cols, x_cols)
    else:
        raise ValueError('Invalid return_type.')


def _g(x):
    return np.power(np.sin(x), 2)


def _m(x, nu=0., gamma=1.):
    return 0.5/np.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))


def make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, return_type='DoubleMLData', **kwargs):
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
    a_0 = kwargs.get('a_0', 1.)
    a_1 = kwargs.get('a_1', 0.25)
    s_1 = kwargs.get('s_1', 1.)

    b_0 = kwargs.get('b_0', 1.)
    b_1 = kwargs.get('b_1', 0.25)
    s_2 = kwargs.get('s_2', 1.)

    cov_mat = toeplitz([np.power(0.7, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    d = a_0 * x[:, 0] + a_1 * np.divide(np.exp(x[:, 2]), 1 + np.exp(x[:, 2])) \
        + s_1 * np.random.standard_normal(size=[n_obs, ])
    y = alpha * d + b_0 * np.divide(np.exp(x[:, 0]), 1 + np.exp(x[:, 0])) \
        + b_1 * x[:, 2] + s_2 * np.random.standard_normal(size=[n_obs, ])

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')


def make_plr_turrell2018(n_obs=100, dim_x=20, theta=0.5, return_type='DoubleMLData', **kwargs):
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
    science, coding and data. `http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/
    <http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/>`_.
    """
    nu = kwargs.get('nu', 0.)
    gamma = kwargs.get('gamma', 1.)

    b = [1 / k for k in range(1, dim_x + 1)]
    sigma = make_spd_matrix(dim_x)

    x = np.random.multivariate_normal(np.zeros(dim_x), sigma, size=[n_obs, ])
    G = _g(np.dot(x, b))
    M = _m(np.dot(x, b), nu=nu, gamma=gamma)
    d = M + np.random.standard_normal(size=[n_obs, ])
    y = np.dot(theta, d) + G + np.random.standard_normal(size=[n_obs, ])

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')


def make_irm_data(n_obs=500, dim_x=20, theta=0, R2_d=0.5, R2_y=0.5, return_type='DoubleMLData'):
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
    v = np.random.uniform(size=[n_obs, ])
    zeta = np.random.standard_normal(size=[n_obs, ])

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y/((1-R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3. * R2_d/((1-R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    d = 1. * ((xx/(1+xx)) > v)

    y = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')


def make_iivm_data(n_obs=500, dim_x=20, theta=1., alpha_x=0.2, return_type='DoubleMLData'):
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
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.3], [0.3, 1.]]),
                                       size=[n_obs, ])
    u = xx[:, 0]
    v = xx[:, 1]

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]

    z = np.random.binomial(p=0.5, n=1, size=[n_obs, ])
    d = 1. * (alpha_x * z + v > 0)

    y = d * theta + np.dot(x, beta) + u

    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)),
                            columns=x_cols + ['y', 'd', 'z'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols, 'z')
    else:
        raise ValueError('Invalid return_type.')


def _make_pliv_data(n_obs=100, dim_x=20, theta=0.5, gamma_z=0.4, return_type='DoubleMLData'):
    b = [1/k for k in range(1, dim_x+1)]
    sigma = make_spd_matrix(dim_x)

    x = np.random.multivariate_normal(np.zeros(dim_x), sigma, size=[n_obs, ])
    G = _g(np.dot(x, b))
    # instrument
    z = _m(np.dot(x, b)) + np.random.standard_normal(size=[n_obs, ])
    # treatment
    M = _m(gamma_z * z + np.dot(x, b))
    d = M + np.random.standard_normal(size=[n_obs, ])
    y = np.dot(theta, d) + G + np.random.standard_normal(size=[n_obs, ])

    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)),
                            columns=x_cols + ['y', 'd', 'z'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols, 'z')
    else:
        raise ValueError('Invalid return_type.')


def make_pliv_CHS2015(n_obs, alpha=1., dim_x=200, dim_z=150, return_type='DoubleMLData'):
    """
    Generates data from a partially linear IV regression model used in Chernozhukov, Hansen and Spindler (2015).
    The data generating process is defined as

    .. math::

        z_i &= \\Pi x_i + \\zeta_i,

        d_i &= x_i' \\gamma + z_i' \\delta + u_i,

        y_i &= \\alpha d_i + x_i' \\beta + \\varepsilon_i,

    with

    .. math::

        \\left(\\begin{matrix} \\varepsilon_i \\\\ u_i \\\\ \\zeta_i \\\\ x_i \\end{matrix} \\right) \\sim
        \\mathcal{N}\\left(0, \\left(\\begin{matrix} 1 & 0.6 & 0 & 0 \\\\ 0.6 & 1 & 0 & 0 \\\\
        0 & 0 & 0.25 I_{p_n^z} & 0 \\\\ 0 & 0 & 0 & \\Sigma \\end{matrix} \\right) \\right)

    where  :math:`\\Sigma` is a :math:`p_n^x \\times p_n^x` matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}` and :math:`I_{p_n^z}` is the :math:`p_n^z \\times p_n^z` identity matrix.
    :math:`\\beta = \\gamma` is a :math:`p_n^x`-vector with entries :math:`\\beta_j=\\frac{1}{j^2}`,
    :math:`\\delta` is a :math:`p_n^z`-vector with entries :math:`\\delta_j=\\frac{1}{j^2}`
    and :math:`\\Pi = (I_{p_n^z}, 0_{p_n^z \\times (p_n^x - p_n^z)})`.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    alpha :
        The value of the causal parameter.
    dim_x :
        The number of covariates.
    dim_z :
        The number of instruments.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d, z)``.

    References
    ----------
    Chernozhukov, V., Hansen, C. and Spindler, M. (2015), Post-Selection and Post-Regularization Inference in Linear
    Models with Many Controls and Instruments. American Economic Review: Papers and Proceedings, 105 (5): 486-90.
    """
    assert dim_x >= dim_z
    # see https://assets.aeaweb.org/asset-server/articles-attachments/aer/app/10505/P2015_1022_app.pdf
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.6], [0.6, 1.]]),
                                       size=[n_obs, ])
    epsilon = xx[:, 0]
    u = xx[:, 1]

    sigma = toeplitz([np.power(0.5, k) for k in range(0, dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x),
                                      sigma,
                                      size=[n_obs, ])

    I_z = np.eye(dim_z)
    xi = np.random.multivariate_normal(np.zeros(dim_z),
                                       0.25*I_z,
                                       size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    gamma = beta
    delta = [1 / (k**2) for k in range(1, dim_z + 1)]
    Pi = np.hstack((I_z, np.zeros((dim_z, dim_x-dim_z))))

    z = np.dot(x, np.transpose(Pi)) + xi
    d = np.dot(x, gamma) + np.dot(z, delta) + u
    y = alpha * d + np.dot(x, beta) + epsilon

    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        z_cols = [f'Z{i + 1}' for i in np.arange(dim_z)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)),
                            columns=x_cols + ['y', 'd'] + z_cols)
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols, z_cols)
    else:
        raise ValueError('Invalid return_type.')


def make_pliv_multiway_cluster_CKMS2021(N=25, M=25, dim_X=100, theta=1., return_type='DoubleMLClusterData', **kwargs):
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
        If ``'DoubleMLClusterData'`` or ``DoubleMLClusterData``, returns a ``DoubleMLClusterData`` object where
        ``DoubleMLClusterData.data`` is a ``pd.DataFrame``.

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
    pi_10 = kwargs.get('pi_10', 1.0)

    xx = np.arange(1, dim_X + 1)
    zeta_0 = kwargs.get('zeta_0', np.power(0.5, xx))
    pi_20 = kwargs.get('pi_20', np.power(0.5, xx))
    xi_0 = kwargs.get('xi_0', np.power(0.5, xx))

    omega_X = kwargs.get('omega_X', np.array([0.25, 0.25]))
    omega_epsilon = kwargs.get('omega_epsilon', np.array([0.25, 0.25]))
    omega_v = kwargs.get('omega_v', np.array([0.25, 0.25]))
    omega_V = kwargs.get('omega_V', np.array([0.25, 0.25]))

    s_X = kwargs.get('s_X', 0.25)
    s_epsilon_v = kwargs.get('s_epsilon_v', 0.25)

    # use np.tile() and np.repeat() for repeating vectors in different styles, i.e.,
    # np.tile([v1, v2, v3], 2) [v1, v2, v3, v1, v2, v3]
    # np.repeat([v1, v2, v3], 2) [v1, v1, v2, v2, v3, v3]

    alpha_V = np.random.normal(size=(N * M))
    alpha_V_i = np.repeat(np.random.normal(size=N), M)
    alpha_V_j = np.tile(np.random.normal(size=M), N)

    cov_mat = np.array([[1, s_epsilon_v], [s_epsilon_v, 1]])
    alpha_eps_v = np.random.multivariate_normal(np.zeros(2), cov_mat, size=[N * M, ])
    alpha_eps = alpha_eps_v[:, 0]
    alpha_v = alpha_eps_v[:, 1]

    alpha_eps_v_i = np.random.multivariate_normal(np.zeros(2), cov_mat, size=[N, ])
    alpha_eps_i = np.repeat(alpha_eps_v_i[:, 0], M)
    alpha_v_i = np.repeat(alpha_eps_v_i[:, 1], M)

    alpha_eps_v_j = np.random.multivariate_normal(np.zeros(2), cov_mat, size=[M, ])
    alpha_eps_j = np.tile(alpha_eps_v_j[:, 0], N)
    alpha_v_j = np.tile(alpha_eps_v_j[:, 1], N)

    cov_mat = toeplitz([np.power(s_X, k) for k in range(dim_X)])
    alpha_X = np.random.multivariate_normal(np.zeros(dim_X), cov_mat, size=[N * M, ])
    alpha_X_i = np.repeat(np.random.multivariate_normal(np.zeros(dim_X), cov_mat, size=[N, ]),
                          M, axis=0)
    alpha_X_j = np.tile(np.random.multivariate_normal(np.zeros(dim_X), cov_mat, size=[M, ]),
                        (N, 1))

    # generate variables
    x = (1 - omega_X[0] - omega_X[1]) * alpha_X \
        + omega_X[0] * alpha_X_i + omega_X[1] * alpha_X_j

    eps = (1 - omega_epsilon[0] - omega_epsilon[1]) * alpha_eps \
        + omega_epsilon[0] * alpha_eps_i + omega_epsilon[1] * alpha_eps_j

    v = (1 - omega_v[0] - omega_v[1]) * alpha_v \
        + omega_v[0] * alpha_v_i + omega_v[1] * alpha_v_j

    V = (1 - omega_V[0] - omega_V[1]) * alpha_V \
        + omega_V[0] * alpha_V_i + omega_V[1] * alpha_V_j

    z = np.matmul(x, xi_0) + V
    d = z * pi_10 + np.matmul(x, pi_20) + v
    y = d * theta + np.matmul(x, zeta_0) + eps

    cluster_cols = ['cluster_var_i', 'cluster_var_j']
    cluster_vars = pd.MultiIndex.from_product([range(N), range(M)]).to_frame(name=cluster_cols).reset_index(drop=True)

    if return_type in _array_alias:
        return x, y, d, cluster_vars.values, z
    elif return_type in _data_frame_alias + _dml_cluster_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_X)]
        data = pd.concat((cluster_vars,
                          pd.DataFrame(np.column_stack((x, y, d, z)), columns=x_cols + ['Y', 'D', 'Z'])),
                         axis=1)
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLClusterData(data, 'Y', 'D', cluster_cols, x_cols, 'Z')
    else:
        raise ValueError('Invalid return_type.')


def make_did_SZ2020(n_obs=500, dgp_type=1, cross_sectional_data=False, return_type='DoubleMLData', **kwargs):
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
    xi = kwargs.get('xi', 0.75)
    c = kwargs.get('c', 0.0)
    lambda_t = kwargs.get('lambda_t', 0.5)

    def f_reg(w):
        res = 210 + 27.4*w[:, 0] + 13.7*(w[:, 1] + w[:, 2] + w[:, 3])
        return res

    def f_ps(w, xi):
        res = xi*(-w[:, 0] + 0.5*w[:, 1] - 0.25*w[:, 2] - 0.1*w[:, 3])
        return res

    dim_x = 4
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    z_tilde_1 = np.exp(0.5*x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0]*x[:, 2]/25)**3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3])**2

    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)

    # error terms
    epsilon_0 = np.random.normal(loc=0, scale=1, size=n_obs)
    epsilon_1 = np.random.normal(loc=0, scale=1, size=[n_obs, 2])

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
        raise ValueError('The dgp_type is not valid.')

    # treatment and propensities
    is_experimental = (dgp_type == 5) or (dgp_type == 6)
    if is_experimental:
        # Set D to be experimental
        p = 0.5 * np.ones(n_obs)
    else:
        p = np.exp(f_ps(features_ps, xi)) / (1 + np.exp(f_ps(features_ps, xi)))
    u = np.random.uniform(low=0, high=1, size=n_obs)
    d = 1.0 * (p >= u)

    # potential outcomes
    nu = np.random.normal(loc=d*f_reg(features_reg), scale=1, size=n_obs)
    y0 = f_reg(features_reg) + nu + epsilon_0
    y1_d0 = 2 * f_reg(features_reg) + nu + epsilon_1[:, 0]
    y1_d1 = 2 * f_reg(features_reg) + nu + epsilon_1[:, 1]
    y1 = d * y1_d1 + (1-d) * y1_d0

    if not cross_sectional_data:
        y = y1 - y0

        if return_type in _array_alias:
            return z, y, d
        elif return_type in _data_frame_alias + _dml_data_alias:
            z_cols = [f'Z{i + 1}' for i in np.arange(dim_x)]
            data = pd.DataFrame(np.column_stack((z, y, d)),
                                columns=z_cols + ['y', 'd'])
            if return_type in _data_frame_alias:
                return data
            else:
                return DoubleMLData(data, 'y', 'd', z_cols)
        else:
            raise ValueError('Invalid return_type.')

    else:
        u_t = np.random.uniform(low=0, high=1, size=n_obs)
        t = 1.0 * (u_t <= lambda_t)
        y = t * y1 + (1-t)*y0

        if return_type in _array_alias:
            return z, y, d, t
        elif return_type in _data_frame_alias + _dml_data_alias:
            z_cols = [f'Z{i + 1}' for i in np.arange(dim_x)]
            data = pd.DataFrame(np.column_stack((z, y, d, t)),
                                columns=z_cols + ['y', 'd', 't'])
            if return_type in _data_frame_alias:
                return data
            else:
                return DoubleMLData(data, 'y', 'd', z_cols, t_col='t')
        else:
            raise ValueError('Invalid return_type.')


def make_confounded_irm_data(n_obs=500, theta=5.0, cf_y=0.04, cf_d=0.04):
    """
    Generates counfounded data from an interactive regression model.

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

    Additionally, generate a confounder :math:`A \\sim \\mathcal{U}[-1, 1]`.
    At first, define the propensity score as

    .. math::

        m(X, A) = P(D=1|X,A) = 0.5 + \\gamma_A \\cdot A

    and generate the treatment :math:`D = 1\\{m(X, A) \\ge U\\}` with :math:`U \\sim \\mathcal{U}[0, 1]`.
    Since :math:`A` is independent of :math:`X`, the short form of the propensity score is given as

    .. math::

        P(D=1|X) = 0.5.

    Further, generate the outcome of interest :math:`Y` as

    .. math::

        Y &= \\theta \\cdot D (Z_5 + 1) + g(Z) + \\beta_A \\cdot A + \\varepsilon

        g(Z) &= 210 + 27.4 \\cdot Z_1 +13.7 \\cdot (Z_2 + Z_3 + Z_4)

    where :math:`\\varepsilon \\sim \\mathcal{N}(0,5)`.
    This implies an average treatment effect of :math:`\\theta`. Additionally, the long and short forms of
    the conditional expectation take the following forms

    .. math::

        \\mathbb{E}[Y|D, X, A] &= \\theta \\cdot D (Z_5 + 1) + g(Z) + \\beta_A \\cdot A

        \\mathbb{E}[Y|D, X] &= (\\theta + \\beta_A \\frac{\\mathrm{Cov}(A, D(Z_5 + 1))}{\\mathrm{Var}(D(Z_5 + 1))})
            \\cdot D (Z_5 + 1) + g(Z).

    Consequently, the strength of confounding is determined via :math:`\\gamma_A` and :math:`\\beta_A`.
    Both are chosen to obtain the desired confounding of the outcome and Riesz Representer (in sample).

    The observed data is given as :math:`W = (Y, D, X)`.
    Further, orcale values of the confounder :math:`A`, the transformed covariated :math:`Z`,
    the potential outcomes of :math:`Y`, the coefficients :math:`\\gamma_a`, :math:`\\beta_a`, the
    long and short forms of the main regression and the propensity score
    are returned in a dictionary.

    Parameters
    ----------
    n_obs : int
        The number of observations to simulate.
        Default is ``500``.
    theta : float or int
        Average treatment effect.
        Default is ``5.0``.
    cf_y : float
        Percentage of the residual variation of the outcome explained by latent/confounding variable.
        Default is ``0.04``.
    cf_d : float
        Percentage gains in the variation of the Riesz Representer generated by latent/confounding variable.
        Default is ``0.04``.

    Returns
    -------
    res_dict : dictionary
       Dictionary with entries ``x``, ``y``, ``d`` and ``oracle_values``.

    References
    ----------
    Sant’Anna, P. H. and Zhao, J. (2020),
    Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
    doi:`10.1016/j.jeconom.2020.06.003 <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.
    """
    c = 0.0  # the confounding strength is only valid for c=0
    dim_x = 5

    # observed covariates
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    z_tilde_1 = np.exp(0.5*x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0] * x[:, 2]/25)**3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3])**2

    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4, x[:, 4:]))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)

    # error terms and unobserved confounder
    var_eps_y = 5
    eps_y = np.random.normal(loc=0, scale=np.sqrt(var_eps_y), size=n_obs)

    # unobserved confounder
    a_bounds = (-1, 1)
    a = np.random.uniform(low=a_bounds[0], high=a_bounds[1], size=n_obs)

    # get the required impact of the confounder on the propensity score
    possible_coefs = np.arange(0.001, 0.4999, 0.001)
    gamma_a = possible_coefs[(np.arctanh(2*possible_coefs) / (2*possible_coefs)) - 1 - cf_d/(1 - cf_d) >= 0][0]

    # compute short and long form of riesz representer
    m_long = 0.5 + gamma_a*a
    m_short = 0.5 * np.ones_like(m_long)

    u = np.random.uniform(low=0, high=1, size=n_obs)
    d = 1.0 * (m_long >= u)

    # short and long version of g
    g_partial_reg = 210 + 27.4*z[:, 0] + 13.7*(z[:, 1] + z[:, 2] + z[:, 3])

    dx = d * (x[:, 4] + 1)
    d1x = x[:, 4] + 1
    var_dx = np.var(dx)
    cov_adx = np.cov(a, dx)[0, 1]

    def f_g(beta_a):
        g_diff = beta_a * (a - cov_adx / var_dx)
        y_diff = eps_y + g_diff
        return np.square(np.mean(np.square(g_diff)) / np.mean(np.square(y_diff)) - cf_y)
    beta_a = minimize_scalar(f_g).x

    g_short_d0 = g_partial_reg
    g_short_d1 = (theta + beta_a * cov_adx / var_dx) * d1x + g_partial_reg
    g_short = d * g_short_d1 + (1.0-d) * g_short_d0

    g_long_d0 = g_partial_reg + beta_a * a
    g_long_d1 = theta * d1x + g_partial_reg + beta_a * a
    g_long = d * g_long_d1 + (1.0-d) * g_long_d0

    y0 = g_long_d0 + eps_y
    y1 = g_long_d1 + eps_y

    y = d * y1 + (1.0-d) * y0

    oracle_values = {'g_long': g_long,
                     'g_short': g_short,
                     'm_long': m_long,
                     'm_short': m_short,
                     'gamma_a': gamma_a,
                     'beta_a': beta_a,
                     'a': a,
                     'y0': y0,
                     'y1': y1,
                     'z': z}

    res_dict = {'x': x,
                'y': y,
                'd': d,
                'oracle_values': oracle_values}

    return res_dict


def make_confounded_plr_data(n_obs=500, theta=5.0, cf_y=0.04, cf_d=0.04, **kwargs):
    """
    Generates counfounded data from an partially linear regression model.

    The data generating process is defined as follows (similar to the Monte Carlo simulation used
    in Sant'Anna and Zhao (2020)). Let :math:`X= (X_1, X_2, X_3, X_4, X_5)^T \\sim \\mathcal{N}(0, \\Sigma)`,
    where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = c^{|j-k|}`. The default value is  :math:`c = 0`, corresponding to the identity matrix.
    Further, define :math:`Z_j = (\\tilde{Z_j} - \\mathbb{E}[\\tilde{Z}_j]) / \\sqrt{\\text{Var}(\\tilde{Z}_j)}`,
    where

    .. math::

        \\tilde{Z}_1 &= \\exp(0.5 \\cdot X_1)

        \\tilde{Z}_2 &= 10 + X_2/(1 + \\exp(X_1))

        \\tilde{Z}_3 &= (0.6 + X_1 \\cdot X_3 / 25)^3

        \\tilde{Z}_4 &= (20 + X_2 + X_4)^2.

    Additionally, generate a confounder :math:`A \\sim \\mathcal{U}[-1, 1]`.
    At first, define the treatment as

    .. math::

        D = -Z_1 + 0.5 \\cdot Z_2 - 0.25 \\cdot Z_3 - 0.1 \\cdot Z_4 + \\gamma_A \\cdot A + \\varepsilon_D

    and with :math:`\\varepsilon \\sim \\mathcal{N}(0,1)`.
    Since :math:`A` is independent of :math:`X`, the long and short form of the treatment regression are given as

    .. math::

        E[D|X,A] = -Z_1 + 0.5 \\cdot Z_2 - 0.25 \\cdot Z_3 - 0.1 \\cdot Z_4 + \\gamma_A \\cdot A

        E[D|X] = -Z_1 + 0.5 \\cdot Z_2 - 0.25 \\cdot Z_3 - 0.1 \\cdot Z_4.

    Further, generate the outcome of interest :math:`Y` as

    .. math::

        Y &= \\theta \\cdot D + g(Z) + \\beta_A \\cdot A + \\varepsilon

        g(Z) &= 210 + 27.4 \\cdot Z_1 +13.7 \\cdot (Z_2 + Z_3 + Z_4)

    where :math:`\\varepsilon \\sim \\mathcal{N}(0,5)`.
    This implies an average treatment effect of :math:`\\theta`. Additionally, the long and short forms of
    the conditional expectation take the following forms

    .. math::

        \\mathbb{E}[Y|D, X, A] &= \\theta \\cdot D + g(Z) + \\beta_A \\cdot A

        \\mathbb{E}[Y|D, X] &= (\\theta + \\gamma_A\\beta_A \\frac{\\mathrm{Var}(A)}{\\mathrm{Var}(D)}) \\cdot D + g(Z).

    Consequently, the strength of confounding is determined via :math:`\\gamma_A` and :math:`\\beta_A`.
    Both are chosen to obtain the desired confounding of the outcome and Riesz Representer (in sample).

    The observed data is given as :math:`W = (Y, D, X)`.
    Further, orcale values of the confounder :math:`A`, the transformed covariated :math:`Z`, the effect :math:`\\theta`,
    the coefficients :math:`\\gamma_a`, :math:`\\beta_a`, the long and short forms of the main regression and
    the propensity score are returned in a dictionary.

    Parameters
    ----------
    n_obs : int
        The number of observations to simulate.
        Default is ``500``.
    theta : float or int
        Average treatment effect.
        Default is ``5.0``.
    cf_y : float
        Percentage of the residual variation of the outcome explained by latent/confounding variable.
        Default is ``0.04``.
    cf_d : float
        Percentage gains in the variation of the Riesz Representer generated by latent/confounding variable.
        Default is ``0.04``.

    Returns
    -------
    res_dict : dictionary
       Dictionary with entries ``x``, ``y``, ``d`` and ``oracle_values``.

    References
    ----------
    Sant’Anna, P. H. and Zhao, J. (2020),
    Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
    doi:`10.1016/j.jeconom.2020.06.003 <https://doi.org/10.1016/j.jeconom.2020.06.003>`_.
    """
    c = kwargs.get('c', 0.0)
    dim_x = kwargs.get('dim_x', 4)

    # observed covariates
    cov_mat = toeplitz([np.power(c, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    z_tilde_1 = np.exp(0.5*x[:, 0])
    z_tilde_2 = 10 + x[:, 1] / (1 + np.exp(x[:, 0]))
    z_tilde_3 = (0.6 + x[:, 0] * x[:, 2]/25)**3
    z_tilde_4 = (20 + x[:, 1] + x[:, 3])**2

    z_tilde = np.column_stack((z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4, x[:, 4:]))
    z = (z_tilde - np.mean(z_tilde, axis=0)) / np.std(z_tilde, axis=0)

    # error terms
    var_eps_y = 5
    eps_y = np.random.normal(loc=0, scale=np.sqrt(var_eps_y), size=n_obs)
    var_eps_d = 1
    eps_d = np.random.normal(loc=0, scale=np.sqrt(var_eps_d), size=n_obs)

    # unobserved confounder
    a_bounds = (-1, 1)
    a = np.random.uniform(low=a_bounds[0], high=a_bounds[1], size=n_obs)
    var_a = np.square(a_bounds[1] - a_bounds[0]) / 12

    # get the required impact of the confounder on the propensity score
    m_short = -z[:, 0] + 0.5*z[:, 1] - 0.25*z[:, 2] - 0.1*z[:, 3]

    def f_m(gamma_a):
        rr_long = eps_d / var_eps_d
        rr_short = (gamma_a * a + eps_d) / (gamma_a**2 * var_a + var_eps_d)
        C2_D = (np.mean(np.square(rr_long)) - np.mean(np.square(rr_short))) / np.mean(np.square(rr_short))
        return np.square(C2_D / (1 + C2_D) - cf_d)

    gamma_a = minimize_scalar(f_m).x
    m_long = m_short + gamma_a*a
    d = m_long + eps_d

    # short and long version of g
    g_partial_reg = 210 + 27.4*z[:, 0] + 13.7*(z[:, 1] + z[:, 2] + z[:, 3])

    var_d = np.var(d)

    def f_g(beta_a):
        g_diff = beta_a * (a - gamma_a * (var_a/var_d) * d)
        y_diff = eps_y + g_diff
        return np.square(np.mean(np.square(g_diff)) / np.mean(np.square(y_diff)) - cf_y)

    beta_a = minimize_scalar(f_g).x

    g_long = theta*d + g_partial_reg + beta_a*a
    g_short = (theta + gamma_a*beta_a * var_a / var_d)*d + g_partial_reg

    y = g_long + eps_y

    oracle_values = {'g_long': g_long,
                     'g_short': g_short,
                     'm_long': m_long,
                     'm_short': m_short,
                     'theta': theta,
                     'gamma_a': gamma_a,
                     'beta_a': beta_a,
                     'a': a,
                     'z': z}

    res_dict = {'x': x,
                'y': y,
                'd': d,
                'oracle_values': oracle_values}

    return res_dict
