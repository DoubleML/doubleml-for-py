import numpy as np
from sklearn.utils import check_X_y

import warnings

from .double_ml import DoubleML
from .double_ml_data import DoubleMLPartialDependenceData
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _dml_tune


class DoubleMLPartialCorr(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partial correlations

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLPartialDependenceData` object
        The :class:`DoubleMLPartialDependenceData` object providing the data and specifying the variables for the model.

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
        A str (``'orthogonal'`` or ``'corr'``) specifying the score function.
        Default is ``'orthogonal'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

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
    >>> dml_data_pcop = make_partial_copula_additive_approx_sparse(copula_family='Gaussian', theta=0.6)
    >>> dml_pcorr = dml.DoubleMLPartialCorr(dml_data_pcop, ml_g, ml_m)
    >>> dml_pcorr.fit().summary
               coef   std err          t          P>|t|     2.5 %    97.5 %
    theta  0.622696  0.027874  22.340007  1.510205e-110  0.568065  0.677327

    Notes
    -----
    ToDo
    """
    _theta_initial = False
    _theta_for_mu = None
    _est_mu_type = 'standard_exp'  # 'standard', 'standard_exp'
    _mu = np.full(2, np.nan)

    def __init__(self,
                 obj_dml_data,
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

        self._coef_bounds = (-0.999, 0.999)
        self._coef_start_val = 0.5

        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict'}

        self._initialize_ml_nuisance_params()

    @property
    def _score_element_names(self):
        score_element_names = ['eps_y', 'eps_z', 'sigma_eps_y', 'sigma_eps_z']
        return score_element_names

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['corr', 'orthogonal']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
            if score == 'corr':
                warnings.warn(('The classical score function for the estimation of partial correlations is not'
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

        score_elements = {'eps_y': eps_y, 'sigma_eps_y': sigma_eps_y,
                          'eps_z': eps_z, 'sigma_eps_z': sigma_eps_z,
                          }

        pcorr = np.corrcoef(eps_y, eps_z)[0,1]
        self._coef_start_val = pcorr

        return score_elements

    def _est_mu(self, theta,
                eps_y, eps_z, sigma_eps_y, sigma_eps_z):

        if self._est_mu_type == 'standard':
            mu, _, _ = _est_mu_components(theta,
                                          eps_y, eps_z, sigma_eps_y, sigma_eps_z)
        else:
            assert self._est_mu_type == 'standard_exp'
            mu, _, _ = _est_mu_components_exp(theta,
                                              eps_y, eps_z, sigma_eps_y, sigma_eps_z)

        return mu

    def _compute_score(self, psi_elements, theta, inds=None):
        if self.score == 'orthogonal':
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if inds is not None:
                eps_y = eps_y[inds]
                eps_z = eps_z[inds]
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

            if self._theta_initial:
                if self._theta_for_mu is None:
                    theta_initial = np.corrcoef(eps_y, eps_z)[0, 1]  # ToDo: check whether this is ok with dml1 vs. dml2
                else:
                    theta_initial = self._theta_for_mu
                mu = self._est_mu(theta_initial,
                                  eps_y, eps_z, sigma_eps_y, sigma_eps_z)
            else:
                mu = self._est_mu(theta,
                                  eps_y, eps_z, sigma_eps_y, sigma_eps_z)

            self._mu = mu

            mu_sigma_g = np.full_like(eps_y, mu[0])
            mu_sigma_m = np.full_like(eps_z, mu[1])

            corr_score = np.multiply(eps_y, eps_z) - theta * np.multiply(sigma_eps_y, sigma_eps_z)

            c_d_theta = np.multiply(corr_score, np.multiply(sigma_eps_y, sigma_eps_z))

            c_d_sigma_g = np.multiply(corr_score, theta * sigma_eps_z) \
                + np.multiply(sigma_eps_y, np.power(eps_y, 2)) - np.power(sigma_eps_y, 3)

            c_d_sigma_m = np.multiply(corr_score, theta * sigma_eps_y) \
                + np.multiply(sigma_eps_z, np.power(eps_z, 2)) - np.power(sigma_eps_z, 3)

            res = c_d_theta \
                - np.multiply(mu_sigma_g, c_d_sigma_g) \
                - np.multiply(mu_sigma_m, c_d_sigma_m)
        else:
            assert self.score == 'corr'
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']
            if inds is not None:
                eps_y = eps_y[inds]
                eps_z = eps_z[inds]
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

            res = np.multiply(eps_y, eps_z) - theta * np.multiply(sigma_eps_y, sigma_eps_z)

        return res

    def _compute_score_deriv(self, psi_elements, theta, inds=None):
        if self.score == 'orthogonal':
            eps_y = psi_elements['eps_y']
            eps_z = psi_elements['eps_z']
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if inds is not None:
                eps_y = eps_y[inds]
                eps_z = eps_z[inds]
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

            if self._theta_initial:
                if self._theta_for_mu is None:
                    theta_initial = np.corrcoef(eps_y, eps_z)[0, 1]  # ToDo: check whether this is ok with dml1 vs. dml2
                else:
                    theta_initial = self._theta_for_mu
                mu = self._est_mu(theta_initial,
                                  eps_y, eps_z, sigma_eps_y, sigma_eps_z)
            else:
                mu = self._est_mu(theta,
                                  eps_y, eps_z, sigma_eps_y, sigma_eps_z)

            mu_sigma_g = np.full_like(eps_y, mu[0])
            mu_sigma_m = np.full_like(eps_z, mu[1])

            eps_y_eps_z = np.multiply(eps_y, eps_z)

            #  TODO: Long-term there will likely be only one _est_mu_type which then will also determine the following
            if self._est_mu_type == 'standard':
                c_d2_theta_theta = - np.power(np.multiply(sigma_eps_y, sigma_eps_z), 2)

                c_d2_theta_sigma_g = np.multiply(eps_y_eps_z, sigma_eps_z) \
                    - 2 * theta * np.multiply(sigma_eps_y, np.power(sigma_eps_z, 2))
                c_d2_theta_sigma_m = np.multiply(eps_y_eps_z, sigma_eps_y) \
                    - 2 * theta * np.multiply(np.power(sigma_eps_y, 2), sigma_eps_z)
            else:
                assert self._est_mu_type == 'standard_exp'
                c_d2_theta_theta = - np.power(np.multiply(sigma_eps_y, sigma_eps_z), 2)

                c_d2_theta_sigma_g = - theta * np.multiply(sigma_eps_y, np.power(sigma_eps_z, 2))
                c_d2_theta_sigma_m = - theta * np.multiply(np.power(sigma_eps_y, 2), sigma_eps_z)

            res = c_d2_theta_theta \
                - np.multiply(mu_sigma_g, c_d2_theta_sigma_g) \
                - np.multiply(mu_sigma_m, c_d2_theta_sigma_m)
        else:
            assert self.score == 'corr'
            sigma_eps_y = psi_elements['sigma_eps_y']
            sigma_eps_z = psi_elements['sigma_eps_z']

            if inds is not None:
                sigma_eps_y = sigma_eps_y[inds]
                sigma_eps_z = sigma_eps_z[inds]

            res = - np.multiply(sigma_eps_y, sigma_eps_z)
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


def _est_mu_components(theta,
                       eps_y, eps_z, sigma_eps_y, sigma_eps_z):

    eps_y_eps_z = np.multiply(eps_y, eps_z)
    eps_y_2 = np.power(eps_y, 2)
    eps_z_2 = np.power(eps_z, 2)

    c_d2_theta_sigma_g = np.multiply(eps_y_eps_z, sigma_eps_z) \
        - 2 * theta * np.multiply(sigma_eps_y, np.power(sigma_eps_z, 2))
    c_d2_theta_sigma_m = np.multiply(eps_y_eps_z, sigma_eps_y) \
        - 2 * theta * np.multiply(np.power(sigma_eps_y, 2), sigma_eps_z)

    c_d2_sigma_g_sigma_g = - theta**2 * np.power(sigma_eps_z, 2) \
        + eps_y_2 - 3 * np.power(sigma_eps_y, 2)
    c_d2_sigma_m_sigma_m = - theta**2 * np.power(sigma_eps_y, 2) \
        + eps_z_2 - 3 * np.power(sigma_eps_z, 2)
    c_d2_sigma_g_sigma_m = theta * eps_y_eps_z \
        - 2 * theta**2 * np.multiply(sigma_eps_y, sigma_eps_z)

    mu_sigma_g_sigma_m = np.array([np.mean(c_d2_theta_sigma_g), np.mean(c_d2_theta_sigma_m)], ndmin=2)
    j_mat = np.array([[np.mean(c_d2_sigma_g_sigma_g), np.mean(c_d2_sigma_g_sigma_m)],
                      [np.mean(c_d2_sigma_g_sigma_m), np.mean(c_d2_sigma_m_sigma_m)]])
    mu = np.squeeze(np.matmul(mu_sigma_g_sigma_m, np.linalg.inv(j_mat)))

    return mu, j_mat, mu_sigma_g_sigma_m


def _est_mu_components_exp(theta,
                           eps_y, eps_z, sigma_eps_y, sigma_eps_z):

    c_d2_theta_sigma_g = - theta * np.multiply(sigma_eps_y, np.power(sigma_eps_z, 2))
    c_d2_theta_sigma_m = - theta * np.multiply(np.power(sigma_eps_y, 2), sigma_eps_z)

    c_d2_sigma_g_sigma_g = - theta**2 * np.power(sigma_eps_z, 2) - 2 * np.power(sigma_eps_y, 2)
    c_d2_sigma_m_sigma_m = - theta**2 * np.power(sigma_eps_y, 2) - 2 * np.power(sigma_eps_z, 2)
    c_d2_sigma_g_sigma_m = - theta**2 * np.multiply(sigma_eps_y, sigma_eps_z)

    mu_sigma_g_sigma_m = np.array([np.mean(c_d2_theta_sigma_g), np.mean(c_d2_theta_sigma_m)], ndmin=2)
    j_mat = np.array([[np.mean(c_d2_sigma_g_sigma_g), np.mean(c_d2_sigma_g_sigma_m)],
                      [np.mean(c_d2_sigma_g_sigma_m), np.mean(c_d2_sigma_m_sigma_m)]])
    mu = np.squeeze(np.matmul(mu_sigma_g_sigma_m, np.linalg.inv(j_mat)))

    return mu, j_mat, mu_sigma_g_sigma_m
