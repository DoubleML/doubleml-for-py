import numpy as np
from sklearn.utils import check_X_y

import warnings
from scipy.stats import norm

from .double_ml import DoubleML
from .double_ml_data import DoubleMLPartialDependenceData
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions
from ._utils_copula import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula


class DoubleMLPartialCopula(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partial copulas

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLPartialDependenceData` object
        The :class:`DoubleMLPartialDependenceData` object providing the data and specifying the variables for the model.

    copula_family : str
        A str (``'Clayton'``, ``'Gaussian'``, ``'Frank'`` or ``'Gumbel'``) specifying the copula family.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'orthogonal'`` or ``'likelihood'``) specifying the score function.
        Default is ``'orthogonal'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_partial_copula_additive_approx_sparse
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(1234)
    >>> learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_g = clone(learner)
    >>> ml_m = clone(learner)
    >>> dml_data_pcop = make_partial_copula_additive_approx_sparse(copula_family='Gumbel', theta=3.)
    >>> dml_pcorr = dml.DoubleMLPartialCopula(dml_data_pcop, 'Gumbel', ml_g, ml_m)
    >>> dml_pcorr.fit().summary
               coef   std err          t         P>|t|     2.5 %    97.5 %
    theta  2.924531  0.138846  21.063127  1.733477e-98  2.652398  3.196664

    Notes
    -----
    :class:`DoubleMLPartialCopula` implements a double machine learning estimate for the partial copulas.
    For this, we consider the joint regression model

    .. math::

        Y &= g_0(X) + \\varepsilon, & &\\mathbb{E}(\\varepsilon | X) = 0,

        Z &= m_0(X) + \\xi, & &\\mathbb{E}(\\xi | X) = 0,

    where we assume a parametric copula (Gaussian, Clayton, Frank or Gumbel) model

    .. math::

        (u(W; \\lambda), v(W; \\lambda)) := ( F_{\\varepsilon}(\\varepsilon), F_{\\xi}(\\xi) )
        := \\bigg(\\Phi\\bigg(\\frac{Y - g_0(X)}{\\sigma_0}\\bigg),
        \\Phi\\bigg(\\frac{Z - m_0(X)}{\\nu_0}\\bigg)\\bigg) \\sim C(\\theta_0),

    with :math:`W := (Y,Z,X)` and nuisance parameters :math:`\\lambda := (g, \\sigma, m, \\nu)`.
    For the disturbances, we assume normal distributions :math:`\\varepsilon \\sim \\mathcal{N}(0,\\sigma_0^2)`
    and :math:`\\xi \\sim \\mathcal{N}(0,\\nu_0^2)`.
    The nuisance parameters are given by :math:`g_0(X) = \\mathbb{E}(Y|X)`, :math:`m_0(X) = \\mathbb{E}(Z|X)`
    and the variances :math:`\\sigma_0^2 = \\mathbb{E}(\\varepsilon^2)` and :math:`\\nu_0^2 = \\mathbb{E}(\\xi^2)`.
    The true parameter :math:`\\theta_0` of the partial copula is the estimation target.

    The implemented score functions are: The classical ``score='likelihood'`` identifying the likelihood estimator given
    by (see Kurz and Kück, 2022)

    .. math::

        \\psi_5(W; \\theta, \\lambda) = \\partial_\\theta \\ell(u(W; \\lambda), v(W; \\lambda); \\theta),

    with nuisance parameters :math:`\\lambda := (g, \\sigma, m, \\nu)`, and the orthogonalized score function
    ``score='orthogonal'`` given by (see Kurz and Kück, 2022)

    .. math::

        \\psi_6(W; \\theta, \\eta) = \\partial_\\theta \\ell(u(W; \\lambda), v(W; \\lambda); \\theta)
        - \\mu \\cdot \\tilde{\\ell}_{\\lambda}(W; \\theta, \\lambda),

    with nuisance parameters :math:`\\eta := (g, \\sigma, m, \\nu, \\mu)`.

    References
    ----------
    Kurz, M. S. and Kück, J. (2022), Double machine learning for partial correlations and partial copulas, Unpublished
    Working Paper.
    """

    _par_initial = False
    _par_for_mu = None
    _mu = np.full(4, np.nan)

    def __init__(self,
                 obj_dml_data,
                 copula_family,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='orthogonal',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        (self.copula, self._coef_bounds, self._coef_start_val) = self._check_copula_family(copula_family)
        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict'}

        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    @property
    def _score_element_names(self):
        score_element_names = ['u', 'v', 'eps_y', 'eps_z', 'sigma_eps_y', 'sigma_eps_z']
        return score_element_names

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_copula_family(self, copula_family):
        valid_copula_families = ['Clayton', 'Gaussian', 'Frank', 'Gumbel']
        if (not isinstance(copula_family, str)) | (copula_family not in valid_copula_families):
            raise ValueError('Invalid copula family ' + str(copula_family) + '. ' +
                             'Valid copula families ' + ' or '.join(valid_copula_families) + '.')
        if copula_family == 'Gaussian':
            copula = GaussianCopula()
            coef_bounds = (-0.999, 0.999)
            coef_start_val = 0.5
        elif copula_family == 'Clayton':
            copula = ClaytonCopula()
            coef_bounds = (0.0001, 28)
            coef_start_val = 3.0
        elif copula_family == 'Frank':
            copula = FrankCopula()
            coef_bounds = (-40, 40)
            coef_start_val = 5.0
        else:
            assert copula_family == 'Gumbel'
            copula = GumbelCopula()
            coef_bounds = (1, 20)
            coef_start_val = 3.0

        return copula, coef_bounds, coef_start_val

    def _check_score(self, score):
        valid_score = ['likelihood', 'orthogonal']
        if (not isinstance(score, str)) | (score not in valid_score):
            raise ValueError('Invalid score ' + str(score) + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')
        if score == 'likelihood':
            warnings.warn(('The likelihood score function for the estimation of partial copulas is not'
                           'necessarily Neyman orthogonal. It might result in biased estimates and standard errors'
                           ' / confidence intervals might not be valid. It is therefore not recommended.'))
        return

    def _check_data(self, obj_dml_data):
        # check and pick up obj_dml_data
        if not isinstance(obj_dml_data, DoubleMLPartialDependenceData):
            raise TypeError('The data must be of DoubleMLPartialDependenceData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, z = check_X_y(x, self._dml_data.z)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'],
                                return_models=return_models)
        _check_finite_predictions(g_hat['preds'], self._learner['ml_g'], 'ml_g', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                return_models=return_models)
        _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        score_elements = self._score_elements(y, z, g_hat['preds'], m_hat['preds'], smpls)
        preds = {'predictions': {'ml_g': g_hat['preds'],
                                 'ml_m': m_hat['preds']},
                 'targets': {'ml_g': g_hat['targets'],
                             'ml_m': m_hat['targets']},
                 'models': {'ml_g': g_hat['models'],
                            'ml_m': m_hat['models']}
                 }

        return score_elements, preds

    def _score_elements(self, y, z, g_hat, m_hat, smpls):
        # compute residuals
        eps_y = y - g_hat
        eps_z = z - m_hat

        sigma_eps_y = np.full_like(y, np.nan, dtype='float64')
        sigma_eps_z = np.full_like(z, np.nan, dtype='float64')
        for _, test_index in smpls:
            sigma_eps_y[test_index] = np.sqrt(np.var(eps_y[test_index]))
            sigma_eps_z[test_index] = np.sqrt(np.var(eps_z[test_index]))

        eps_y_scaled = np.divide(eps_y, sigma_eps_y)
        eps_z_scaled = np.divide(eps_z, sigma_eps_z)
        u = norm.cdf(eps_y_scaled)
        v = norm.cdf(eps_z_scaled)

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0.):
            u[u < self.trimming_threshold] = self.trimming_threshold
            u[u > 1. - self.trimming_threshold] = 1. - self.trimming_threshold
            v[v < self.trimming_threshold] = self.trimming_threshold
            v[v > 1. - self.trimming_threshold] = 1. - self.trimming_threshold

        score_elements = {'u': u, 'eps_y': eps_y, 'sigma_eps_y': sigma_eps_y,
                          'v': v, 'eps_z': eps_z, 'sigma_eps_z': sigma_eps_z,
                          }

        par_like = self.copula.mle_est(u, v)
        self._coef_start_val = par_like

        return score_elements

    def _est_mu(self, par,
                u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z):
        mu, _, _ = _est_mu_components(self.copula, par,
                                      u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z)
        return mu

    def _compute_score(self, psi_elements, par):
        if self.score == 'orthogonal':
            u = psi_elements['u']
            v = psi_elements['v']
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if self._par_initial:
                if self._par_for_mu is None:
                    par_initial = self.copula.mle_est(u, v)  # ToDo: check whether this is ok with dml1 vs. dml2
                else:
                    par_initial = self._par_for_mu
                mu = self._est_mu(par_initial,
                                  u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z)
            else:
                mu = self._est_mu(par,
                                  u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z)

            self._mu = mu
            mu_g = np.full_like(u, mu[0])
            mu_sigma_g = np.full_like(u, mu[1])
            mu_m = np.full_like(v, mu[2])
            mu_sigma_m = np.full_like(v, mu[3])

            eps_y_scaled = np.divide(eps_y, sigma_eps_y)
            eps_z_scaled = np.divide(eps_z, sigma_eps_z)
            f_eps_y = norm.pdf(eps_y_scaled)
            f_eps_z = norm.pdf(eps_z_scaled)

            u_d_g = - np.divide(f_eps_y, sigma_eps_y)
            u_d_sigma_g = np.multiply(eps_y_scaled, u_d_g)

            v_d_m = - np.divide(f_eps_z, sigma_eps_z)
            v_d_sigma_m = np.multiply(eps_z_scaled, v_d_m)

            ll_d_par = self.copula.ll_deriv(par, u, v, 'd_par')
            ll_d_u = self.copula.ll_deriv(par, u, v, 'd_u')
            ll_d_v = self.copula.ll_deriv(par, u, v, 'd_v')

            ll_d_g = np.multiply(u_d_g, ll_d_u) + np.divide(eps_y, np.power(sigma_eps_y, 2))
            ll_d_sigma_g = np.multiply(u_d_sigma_g, ll_d_u) \
                + np.divide(np.power(eps_y, 2), np.power(sigma_eps_y, 3)) - np.divide(1, sigma_eps_y)

            ll_d_m = np.multiply(v_d_m, ll_d_v) + np.divide(eps_z, np.power(sigma_eps_z, 2))
            ll_d_sigma_m = np.multiply(v_d_sigma_m, ll_d_v) \
                + np.divide(np.power(eps_z, 2), np.power(sigma_eps_z, 3)) - np.divide(1, sigma_eps_z)

            res = ll_d_par - np.multiply(mu_g, ll_d_g) \
                - np.multiply(mu_sigma_g, ll_d_sigma_g) \
                - np.multiply(mu_m, ll_d_m) \
                - np.multiply(mu_sigma_m, ll_d_sigma_m)
        else:
            assert self.score == 'likelihood'
            u = psi_elements['u']
            v = psi_elements['v']
            res = self.copula.ll_deriv(par,
                                       u, v,
                                       'd_par')
        return res

    def _compute_score_deriv(self, psi_elements, par):
        if self.score == 'orthogonal':
            u = psi_elements['u']
            v = psi_elements['v']
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if self._par_initial:
                if self._par_for_mu is None:
                    par_initial = self.copula.mle_est(u, v)  # ToDo: check whether this is ok with dml1 vs. dml2
                else:
                    par_initial = self._par_for_mu
                mu = self._est_mu(par_initial,
                                  u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z)
            else:
                mu = self._est_mu(par,
                                  u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z)

            mu_g = np.full_like(u, mu[0])
            mu_sigma_g = np.full_like(u, mu[1])
            mu_m = np.full_like(v, mu[2])
            mu_sigma_m = np.full_like(v, mu[3])

            eps_y_scaled = np.divide(eps_y, sigma_eps_y)
            eps_z_scaled = np.divide(eps_z, sigma_eps_z)
            f_eps_y = norm.pdf(eps_y_scaled)
            f_eps_z = norm.pdf(eps_z_scaled)

            u_d_g = - np.divide(f_eps_y, sigma_eps_y)
            u_d_sigma_g = np.multiply(eps_y_scaled, u_d_g)

            v_d_m = - np.divide(f_eps_z, sigma_eps_z)
            v_d_sigma_m = np.multiply(eps_z_scaled, v_d_m)

            ll_d2_par_par = self.copula.ll_deriv(par, u, v, 'd2_par_par')
            ll_d2_par_u = self.copula.ll_deriv(par, u, v, 'd2_par_u')
            ll_d2_par_v = self.copula.ll_deriv(par, u, v, 'd2_par_v')

            ll_d2_par_g = np.multiply(u_d_g, ll_d2_par_u)
            ll_d2_par_sigma_g = np.multiply(u_d_sigma_g, ll_d2_par_u)

            ll_d2_par_m = np.multiply(v_d_m, ll_d2_par_v)
            ll_d2_par_sigma_m = np.multiply(v_d_sigma_m, ll_d2_par_v)

            res = ll_d2_par_par - np.multiply(mu_g, ll_d2_par_g) \
                - np.multiply(mu_sigma_g, ll_d2_par_sigma_g) \
                - np.multiply(mu_m, ll_d2_par_m) \
                - np.multiply(mu_sigma_m, ll_d2_par_sigma_m)
        else:
            assert self.score == 'likelihood'
            u = psi_elements['u']
            v = psi_elements['v']
            res = self.copula.ll_deriv(par,
                                       u, v,
                                       'd2_par_par')
        return res

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, z = check_X_y(x, self._dml_data.z)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        g_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(z, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res


def _est_mu_components(copula, par,
                       u, v, eps_y, eps_z, sigma_eps_y, sigma_eps_z):
    eps_y_scaled = np.divide(eps_y, sigma_eps_y)
    eps_z_scaled = np.divide(eps_z, sigma_eps_z)
    f_eps_y = norm.pdf(eps_y_scaled)
    f_eps_z = norm.pdf(eps_z_scaled)

    u_d_g = - np.divide(f_eps_y, sigma_eps_y)
    u_d_sigma_g = np.multiply(eps_y_scaled, u_d_g)
    u_d2_g_g = np.divide(u_d_sigma_g,
                         sigma_eps_y)
    u_d2_sigma_g_sigma_g = - np.multiply(u_d2_g_g,
                                         2 - np.power(eps_y_scaled, 2))
    u_d2_g_sigma_g = np.multiply(np.divide(f_eps_y, np.power(sigma_eps_y, 2)),
                                 1 - np.power(eps_y_scaled, 2))

    v_d_m = - np.divide(f_eps_z, sigma_eps_z)
    v_d_sigma_m = np.multiply(eps_z_scaled, v_d_m)
    v_d2_m_m = np.divide(v_d_sigma_m,
                         sigma_eps_z)
    v_d2_sigma_m_sigma_m = - np.multiply(v_d2_m_m,
                                         2 - np.power(eps_z_scaled, 2))
    v_d2_m_sigma_m = np.multiply(np.divide(f_eps_z, np.power(sigma_eps_z, 2)),
                                 1 - np.power(eps_z_scaled, 2))

    ll_d_u = copula.ll_deriv(par, u, v, 'd_u')
    ll_d_v = copula.ll_deriv(par, u, v, 'd_v')
    ll_d2_u_u = copula.ll_deriv(par, u, v, 'd2_u_u')
    ll_d2_v_v = copula.ll_deriv(par, u, v, 'd2_v_v')
    ll_d2_u_v = copula.ll_deriv(par, u, v, 'd2_u_v')
    ll_d2_par_u = copula.ll_deriv(par, u, v, 'd2_par_u')
    ll_d2_par_v = copula.ll_deriv(par, u, v, 'd2_par_v')

    ll_d2_par_g = np.multiply(u_d_g, ll_d2_par_u)
    ll_d2_par_sigma_g = np.multiply(u_d_sigma_g, ll_d2_par_u)

    ll_d2_par_m = np.multiply(v_d_m, ll_d2_par_v)
    ll_d2_par_sigma_m = np.multiply(v_d_sigma_m, ll_d2_par_v)

    ll_d2_g_g = np.multiply(u_d2_g_g, ll_d_u) \
        + np.multiply(np.power(u_d_g, 2), ll_d2_u_u) \
        - np.divide(1, np.power(sigma_eps_y, 2))
    ll_d2_sigma_g_sigma_g = np.multiply(u_d2_sigma_g_sigma_g, ll_d_u) \
        + np.multiply(np.power(u_d_sigma_g, 2), ll_d2_u_u) \
        + np.divide(1, np.power(sigma_eps_y, 2)) \
        - 3. * np.divide(np.power(eps_y, 2), np.power(sigma_eps_y, 4))
    ll_d2_g_sigma_g = np.multiply(u_d2_g_sigma_g, ll_d_u) \
        + np.multiply(np.multiply(u_d_g, u_d_sigma_g), ll_d2_u_u) \
        - 2. * np.divide(eps_y, np.power(sigma_eps_y, 3))

    ll_d2_m_m = np.multiply(v_d2_m_m, ll_d_v) \
        + np.multiply(np.power(v_d_m, 2), ll_d2_v_v) \
        - np.divide(1, np.power(sigma_eps_z, 2))
    ll_d2_sigma_m_sigma_m = np.multiply(v_d2_sigma_m_sigma_m, ll_d_v) \
        + np.multiply(np.power(v_d_sigma_m, 2), ll_d2_v_v) \
        + np.divide(1, np.power(sigma_eps_z, 2)) \
        - 3. * np.divide(np.power(eps_z, 2), np.power(sigma_eps_z, 4))
    ll_d2_m_sigma_m = np.multiply(v_d2_m_sigma_m, ll_d_v) \
        + np.multiply(np.multiply(v_d_m, v_d_sigma_m), ll_d2_v_v) \
        - 2. * np.divide(eps_z, np.power(sigma_eps_z, 3))

    ll_d2_g_m = np.multiply(np.multiply(u_d_g, v_d_m), ll_d2_u_v)
    ll_d2_g_sigma_m = np.multiply(np.multiply(u_d_g, v_d_sigma_m), ll_d2_u_v)
    ll_d2_sigma_g_m = np.multiply(np.multiply(u_d_sigma_g, v_d_m), ll_d2_u_v)
    ll_d2_sigma_g_sigma_m = np.multiply(np.multiply(u_d_sigma_g, v_d_sigma_m), ll_d2_u_v)

    mu_g_sigma_g_m_sigma_m = np.array([np.mean(ll_d2_par_g), np.mean(ll_d2_par_sigma_g),
                                       np.mean(ll_d2_par_m), np.mean(ll_d2_par_sigma_m)], ndmin=2)
    j_mat = np.array([[np.mean(ll_d2_g_g), np.mean(ll_d2_g_sigma_g),
                       np.mean(ll_d2_g_m), np.mean(ll_d2_g_sigma_m)],
                      [np.mean(ll_d2_g_sigma_g), np.mean(ll_d2_sigma_g_sigma_g),
                       np.mean(ll_d2_sigma_g_m), np.mean(ll_d2_sigma_g_sigma_m)],
                      [np.mean(ll_d2_g_m), np.mean(ll_d2_sigma_g_m),
                       np.mean(ll_d2_m_m), np.mean(ll_d2_m_sigma_m)],
                      [np.mean(ll_d2_g_sigma_m), np.mean(ll_d2_sigma_g_sigma_m),
                       np.mean(ll_d2_m_sigma_m), np.mean(ll_d2_sigma_m_sigma_m)]])
    mu = np.squeeze(np.matmul(mu_g_sigma_g_m_sigma_m, np.linalg.inv(j_mat)))

    return mu, j_mat, mu_g_sigma_g_m_sigma_m
