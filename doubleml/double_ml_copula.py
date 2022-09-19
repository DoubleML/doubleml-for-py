import numpy as np
from sklearn.utils import check_X_y

import warnings
from scipy.stats import norm

from .double_ml import DoubleML
from .double_ml_data import DoubleMLPartialDependenceData
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _dml_tune
from ._utils_copula import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula


class DoubleMLPartialCopula(NonLinearScoreMixin, DoubleML):

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
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        # TODO: One class for DoubleMLPartialCopula or separate for the different families, like DoubleMLGaussianCopula
        valid_copula_families = ['Clayton', 'Gaussian', 'Frank', 'Gumbel']
        # TODO that copula_family is a str
        if copula_family not in valid_copula_families:
            raise ValueError('Invalid copula family ' + copula_family + '. ' +
                             'Valid copula families ' + ' or '.join(valid_copula_families) + '.')
        if copula_family == 'Gaussian':
            self.copula = GaussianCopula()
            self._coef_bounds = (-0.999, 0.999)
            self._coef_start_val = 0.5
        elif copula_family == 'Clayton':
            self.copula = ClaytonCopula()
            self._coef_bounds = (0.0001, 28)
            self._coef_start_val = 3.0
        elif copula_family == 'Frank':
            self.copula = FrankCopula()
            self._coef_bounds = (-40, 40)
            self._coef_start_val = 5.0
        else:
            assert copula_family == 'Gumbel'
            self.copula = GumbelCopula()
            self._coef_bounds = (1, 20)
            self._coef_start_val = 3.0

        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict'}

        self._initialize_ml_nuisance_params()

    @property
    def _score_element_names(self):
        score_element_names = ['u', 'v', 'eps_y', 'eps_z', 'sigma_eps_y', 'sigma_eps_z']
        return score_element_names

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['likelihood', 'orthogonal']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
            if score == 'likelihood':
                warnings.warn(('The likelihood score function for the estimation of partial copulas is not'
                               'necessarily Neyman orthogonal. It might result in biased estimates and standard errors'
                               ' / confidence intervals might not be valid. It is therefore not recommended.'))
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):
        # check and pick up obj_dml_data
        if not isinstance(obj_dml_data, DoubleMLPartialDependenceData):
            raise TypeError('The data must be of DoubleMLPartialDependenceData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, z = check_X_y(x, self._dml_data.z)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'])

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])

        score_elements = self._score_elements(y, z, g_hat, m_hat, smpls)
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}

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

    def _compute_score(self, psi_elements, par, inds=None):
        if self.score == 'orthogonal':
            u = psi_elements['u']
            v = psi_elements['v']
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if inds is not None:
                u = u[inds]
                v = v[inds]
                eps_y = eps_y[inds]
                eps_z = eps_z[inds]
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

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
            if inds is not None:
                u = u[inds]
                v = v[inds]
            res = self.copula.ll_deriv(par,
                                       u, v,
                                       'd_par')
        return res

    def _compute_score_deriv(self, psi_elements, par, inds=None):
        if self.score == 'orthogonal':
            u = psi_elements['u']
            v = psi_elements['v']
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if inds is not None:
                u = u[inds]
                v = v[inds]
                eps_y = eps_y[inds]
                eps_z = eps_z[inds]
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

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
            if inds is not None:
                u = u[inds]
                v = v[inds]
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
