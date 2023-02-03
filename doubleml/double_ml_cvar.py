import numpy as np
from scipy.optimize import root_scalar
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold, train_test_split

from .double_ml import DoubleML
from .double_ml_score_mixins import LinearScoreMixin
from ._utils import _dml_cv_predict, _trimm, _predict_zero_one_propensity, _check_contains_iv, \
    _check_zero_one_treatment, _check_quantile, _check_treatment, _check_trimming, _check_score, \
    _normalize_ipw
from .double_ml_data import DoubleMLData
from ._utils_resampling import DoubleMLResampling


class DoubleMLCVAR(LinearScoreMixin, DoubleML):
    """Double machine learning for conditional value at risk for potential outcomes

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
        A str (``'CVaR'`` is the only choice) specifying the score function
        for conditional value at risk for potential outcomes.
        Default is ``'CVaR'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied(``True`` is the only choice).
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_cvar_obj = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment=1, quantile=0.5)
    >>> dml_cvar_obj.fit().summary
           coef  std err          t         P>|t|     2.5 %    97.5 %
    d  3.143662  0.20791  15.120269  1.190551e-51  2.736165  3.551159
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment=1,
                 quantile=0.5,
                 n_folds=5,
                 n_rep=1,
                 score='CVaR',
                 dml_procedure='dml2',
                 normalize_ipw=True,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
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
        self._normalize_ipw = normalize_ipw

        if self._is_cluster_data:
            raise NotImplementedError('Estimation with clustering not implemented.')
        self._check_data(self._dml_data)
        valid_score = ['CVaR']
        _check_score(self.score, valid_score)
        _check_quantile(self.quantile)
        _check_treatment(self.treatment)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

        # initialize starting values and bounds
        self._coef_bounds = (self._dml_data.y.min(), self._dml_data.y.max())
        self._coef_start_val = np.quantile(self._dml_data.y, self.quantile)

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        _ = self._check_learner(ml_g, 'ml_g', regressor=False, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}

        self._initialize_ml_nuisance_params()

        if draw_sample_splitting:
            obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                    n_rep=self.n_rep,
                                                    n_obs=self._dml_data.n_obs,
                                                    apply_cross_fitting=self.apply_cross_fitting,
                                                    stratify=self._dml_data.d)
            self._smpls = obj_dml_resampling.split_samples()

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
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

    def _compute_ipw_score(self, theta, d, y, prop):
        score = (d == self.treatment) / prop * (y <= theta) - self.quantile
        return score

    def _score_elements(self, ipw_est, y, d, g_hat, m_hat):
        u1 = np.ones_like(y) * ipw_est
        u2 = (y - self.quantile * ipw_est) / (1 - self.quantile)
        u = np.max(np.column_stack((u1, u2)), 1)

        psi_b = (d == self.treatment) * (u - g_hat) / m_hat + g_hat
        psi_a = np.full_like(m_hat, -1.0)
        return psi_a, psi_b

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in ['ml_g', 'ml_m']}

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # initialize nuisance predictions, targets and models
        g_hat = {'models': None,
                 'targets': np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                 'preds': np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
                 }
        m_hat = {'models': None,
                 'targets': np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                 'preds': np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
                 }

        # initialize models
        fitted_models = {'ml_g': [clone(self._learner['ml_g']) for i_fold in range(self.n_folds)],
                         'ml_m': [clone(self._learner['ml_m']) for i_fold in range(self.n_folds)]
                         }
        # caculate nuisance functions over different folds
        for i_fold in range(self.n_folds):
            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                          random_state=42, stratify=d[train_inds])
            smpls_prelim = [(train, test) for train, test in
                            StratifiedKFold(n_splits=self.n_folds).split(X=train_inds_1, y=d[train_inds_1])]

            d_train_1 = d[train_inds_1]
            y_train_1 = y[train_inds_1]
            x_train_1 = x[train_inds_1, :]

            m_hat_prelim = _dml_cv_predict(self._learner['ml_m'], x_train_1, d_train_1,
                                           method='predict_proba', smpls=smpls_prelim)['preds']

            m_hat_prelim = _trimm(m_hat_prelim, self.trimming_rule, self.trimming_threshold)

            if self._normalize_ipw:
                m_hat_prelim = _normalize_ipw(m_hat_prelim, d_train_1)
            if self.treatment == 0:
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

            # use the preliminary estimates to fit the nuisance parameters on train_2
            d_train_2 = d[train_inds_2]
            y_train_2 = y[train_inds_2]
            x_train_2 = x[train_inds_2, :]

            x_test = x[test_inds, :]

            dx_treat_train_2 = x_train_2[d_train_2 == self.treatment, :]
            y_treat_train_2 = y_train_2[d_train_2 == self.treatment]
            fitted_models['ml_g'][i_fold].fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

            # predict nuisance values on the test data and the corresponding targets
            g_hat['preds'][test_inds] = _predict_zero_one_propensity(fitted_models['ml_g'][i_fold], x_test)
            g_hat['targets'][test_inds] = y[test_inds] <= ipw_est

            # refit the propensity score on the whole training set
            fitted_models['ml_m'][i_fold].fit(x[train_inds, :], d[train_inds])
            m_hat['preds'][test_inds] = _predict_zero_one_propensity(fitted_models['ml_m'][i_fold], x_test)

        # set target for propensity score
        m_hat['targets'] = d

        # set the target for g to be a float and only relevant values
        g_hat['targets'] = g_hat['targets'].astype(float)
        g_hat['targets'][d != self.treatment] = np.nan

        if return_models:
            g_hat['models'] = fitted_models['ml_g']
            m_hat['models'] = fitted_models['ml_m']

        # clip propensities and normalize ipw weights
        # this is not done in the score to be equivalent to PQ models
        m_hat_adj = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)
        if self._normalize_ipw:
            if self.dml_procedure == 'dml1':
                for _, test_index in smpls:
                    m_hat_adj[test_index] = _normalize_ipw(m_hat_adj[test_index], d[test_index])
            else:
                m_hat_adj = _normalize_ipw(m_hat_adj, d)

        if self.treatment == 0:
            m_hat_adj = 1 - m_hat_adj

        psi_a, psi_b = self._score_elements(ipw_est, y, d, g_hat['preds'], m_hat_adj)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_g': g_hat['preds'],
                                 'ml_m': m_hat['preds']},
                 'targets': {'ml_g': g_hat['targets'],
                             'ml_m': m_hat['targets']},
                 'models': {'ml_g': g_hat['models'],
                            'ml_m': m_hat['models']}
                 }
        return psi_elements, preds

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        raise NotImplementedError('Nuisance tuning not implemented for potential quantiles.')

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        _check_contains_iv(obj_dml_data)
        _check_zero_one_treatment(self)
        return
