import numpy as np
from scipy.optimize import root_scalar
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold, train_test_split

from .double_ml import DoubleML
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict
from .double_ml_data import DoubleMLData, DoubleMLClusterData


class DoubleMLPQ(NonLinearScoreMixin, DoubleML):
    """Double machine learning for potential quantiles

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function
         :math:`g_0(X) = E[Y <= \theta | X, D=d]` .

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=d|X]`.

    treatment : int
        Binary treatment indicator. Has to be either ``0`` or ``1``. Determines the potential outcome to be considered.
        Default is ``1``.

    quantile : float
        Quantile of the potential outcome. Has to be between ``0`` and ``1``.
        Default is ``0.5``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'PQ'`` is the only choice) specifying the score function
        for potential quantiles.
        Default is ``'PQ'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

    h : float or None
        The bandwidth to be used for the kernel density estimation of the score derivative.
        If ``None`` the bandwidth will be set to ``np.power(n_obs, -0.2)``, where ``n_obs`` is
        the number of observations in the sample.
        Default is ``1e-12``.

    normalize : bool
        Indicates whether to normalize weights in the estimation of the score derivative.
        Default is ``True``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied(``True`` is the only choice).
        Default is ``True``.
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment,
                 quantile=0.5,
                 n_folds=5,
                 n_rep=1,
                 score='PQ',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 h=None,
                 normalize=True,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._quantile = quantile
        self._treatment = treatment
        self._h = h
        if self.h is None:
            self._h = np.power(self._dml_data.n_obs, -0.2)
        self._normalize = normalize

        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            self._is_cluster_data = True
        if self._is_cluster_data:
            raise NotImplementedError('Estimation with clustering not implemented.')
        self._check_data(self._dml_data)
        self._check_score(self.score)
        self._check_quantile(self.quantile)
        self._check_treatment(self.treatment)
        self._check_bandwidth(self.h)
        if not isinstance(self.normalize, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize))} passed.')

        # initialize starting values and bounds
        self._coef_bounds = (self._dml_data.y.min(), self._dml_data.y.max())
        self._coef_start_val = np.quantile(self._dml_data.y, self.quantile)

        # initialize and check trimming
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold
        self._check_trimming()

        _ = self._check_learner(ml_g, 'ml_g', regressor=False, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m),
                         'ml_m_prelim': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba',
                                'ml_m_prelim': 'predict_proba'}

        self._initialize_ml_nuisance_params()

    @property
    def quantile(self):
        """
        Quantile for potential outcome.
        """
        return self._quantile

    @property
    def treatment(self):
        """
        Treatment indicator for potential outcome.
        """
        return self._treatment

    @property
    def h(self):
        """
        The bandwidth the kernel density estimation of the derivative.
        """
        return self._h

    @property
    def normalize(self):
        """
        Indicates of the weights in the derivative estimation should be normalized.
        """
        return self._normalize

    @property
    def _score_element_names(self):
        return ['ind_d', 'g', 'm', 'y']

    def _compute_ipw_score(self, theta, d, y, prop):
        score = (d == self._treatment) * (y <= theta) / prop - self.quantile
        return score

    def _compute_score(self, psi_elements, coef, inds=None):
        ind_d = psi_elements['ind_d']
        g = psi_elements['g']
        m = psi_elements['m']
        y = psi_elements['y']
        if inds is not None:
            ind_d = psi_elements['ind_d'][inds]
            g = psi_elements['g'][inds]
            m = psi_elements['m'][inds]
            y = psi_elements['y'][inds]

        score = ind_d * ((y <= coef) - g) / m + g - self.quantile
        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        ind_d = psi_elements['ind_d']
        m = psi_elements['m']
        y = psi_elements['y']
        if inds is not None:
            ind_d = psi_elements['ind_d'][inds]
            m = psi_elements['m'][inds]
            y = psi_elements['y'][inds]

        score_weights = ind_d / m
        normalization = score_weights.mean()
        if self._normalize:
            score_weights /= normalization

        u = (y - coef).reshape(-1, 1) / self._h
        kernel_est = np.exp(-1. * np.power(u, 2) / 2) / np.sqrt(2 * np.pi)
        deriv = np.multiply(score_weights, kernel_est.reshape(-1, )) / self._h

        return deriv

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in ['ml_g', 'ml_m', 'ml_m_prelim']}

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # initialize nuisance predictions
        g_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        m_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)

        # caculate nuisance functions over different folds
        for i_fold in range(self.n_folds):
            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5, random_state=42)
            smpls_prelim = [(train, test) for train, test in KFold(n_splits=self.n_folds).split(train_inds_1)]

            d_train_1 = d[train_inds_1]
            y_train_1 = y[train_inds_1]
            x_train_1 = x[train_inds_1, :]
            m_hat_prelim = _dml_cv_predict(self._learner['ml_m_prelim'], x_train_1, d_train_1,
                                           method='predict_proba', smpls=smpls_prelim)['preds']

            m_hat_prelim[m_hat_prelim < self.trimming_threshold] = self.trimming_threshold
            m_hat_prelim[m_hat_prelim > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

            if self._treatment == 0:
                m_hat_prelim = 1 - m_hat_prelim

            # preliminary ipw estimate
            def ipw_score(theta):
                res = np.mean(self._compute_ipw_score(theta, d_train_1, y_train_1, m_hat_prelim))
                return res

            def get_bracket_guess(coef_start, coef_bounds):
                max_bracket_length = coef_bounds[1] - coef_bounds[0]
                b_guess = coef_bounds
                delta = 0.1
                s_different = False
                while (not s_different) & (delta <= 1.0):
                    a = np.maximum(coef_start - delta * max_bracket_length / 2, coef_bounds[0])
                    b = np.minimum(coef_start + delta * max_bracket_length / 2, coef_bounds[1])
                    b_guess = (a, b)
                    f_a = ipw_score(b_guess[0])
                    f_b = ipw_score(b_guess[1])
                    s_different = (np.sign(f_a) != np.sign(f_b))
                    delta += 0.1
                return s_different, b_guess

            _, bracket_guess = get_bracket_guess(self._coef_start_val, self._coef_bounds)

            root_res = root_scalar(ipw_score,
                                   bracket=bracket_guess,
                                   method='brentq')
            ipw_est = root_res.root

            # readjust start value for minimization
            self._coef_start_val = ipw_est

            # use the preliminary estimates to fit the nuisance parameters on train_2
            d_train_2 = d[train_inds_2]
            y_train_2 = y[train_inds_2]
            x_train_2 = x[train_inds_2, :]

            dx_treat_train_2 = np.column_stack((d_train_2[d_train_2 == self._treatment],
                                                x_train_2[d_train_2 == self._treatment, :]))
            y_treat_train_2 = y_train_2[d_train_2 == self._treatment]
            self._learner['ml_g'].fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

            # predict nuisance values on the test data
            if self._treatment == 0:
                dx_test = np.column_stack((np.zeros_like(d[test_inds]), x[test_inds, :]))
            elif self._treatment == 1:
                dx_test = np.column_stack((np.ones_like(d[test_inds]), x[test_inds, :]))

            g_hat[test_inds] = self._learner['ml_g'].predict_proba(dx_test)[:, 1]

            # refit the propensity score on the whole training set
            self._learner['ml_m'].fit(x[train_inds, :], d[train_inds])
            m_hat[test_inds] = self._learner['ml_m'].predict_proba(x[test_inds, :])[:, self._treatment]

            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        psi_elements = {'ind_d': d == self._treatment,
                        'g': g_hat,
                        'm': m_hat,
                        'y': y}
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}
        return psi_elements, preds

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        raise NotImplementedError('Nuisance tuning not implemented for potential quantiles.')

    def _check_score(self, score):
        valid_score = ['PQ']
        if isinstance(score, str):
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            raise TypeError('Invalid score. ' +
                            'Valid score ' + ' or '.join(valid_score) + '.')
        return

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s).')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an PQ model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return

    def _check_quantile(self, quantile):
        if not isinstance(quantile, float):
            raise TypeError('Quantile has to be a float. ' +
                            f'Object of type {str(type(quantile))} passed.')

        if (quantile <= 0) | (quantile >= 1):
            raise ValueError('Quantile has be between 0 or 1. ' +
                             f'Quantile {str(quantile)} passed.')

    def _check_treatment(self, treatment):
        if not isinstance(treatment, int):
            raise TypeError('Treatment indicator has to be an integer. ' +
                            f'Object of type {str(type(treatment))} passed.')

        if (treatment != 0) & (treatment != 1):
            raise ValueError('Treatment indicator has be either 0 or 1. ' +
                             f'Treatment indicator {str(treatment)} passed.')

    def _check_bandwidth(self, bandwidth):
        if not isinstance(bandwidth, float):
            raise TypeError('Bandwidth has to be a float. ' +
                            f'Object of type {str(type(bandwidth))} passed.')

        if bandwidth <= 0:
            raise ValueError('Bandwidth has be positive. ' +
                             f'Bandwidth {str(bandwidth)} passed.')

    def _check_trimming(self):
        valid_trimming_rule = ['truncate']
        if self.trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + str(self.trimming_rule) + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        if not isinstance(self.trimming_threshold, float):
            raise TypeError('trimming_threshold has to be a float. ' +
                            f'Object of type {str(type(self.trimming_threshold))} passed.')
        if (self.trimming_threshold <= 0) | (self.trimming_threshold >= 0.5):
            raise ValueError('Invalid trimming_threshold ' + str(self.trimming_threshold) + '. ' +
                             'trimming_threshold has to be between 0 and 0.5.')
