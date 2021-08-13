import pandas as pd
import numpy as np

from scipy.linalg import toeplitz

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
        x_cols = poly.get_feature_names(x_cols)

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
    Klaaßen (2020).

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
