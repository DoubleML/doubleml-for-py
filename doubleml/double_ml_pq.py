import numpy as np
import copy
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold, train_test_split

from .double_ml import DoubleML
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _trimm, _predict_zero_one_propensity, _check_contains_iv, \
    _check_zero_one_treatment, _check_quantile, _check_treatment, _check_trimming, _check_score, _get_bracket_guess, \
    _default_kde, _normalize_ipw, _dml_tune, _solve_ipw_score
from .double_ml_data import DoubleMLData
from ._utils_resampling import DoubleMLResampling


class DoubleMLPQ(NonLinearScoreMixin, DoubleML):
    """Double machine learning for potential quantiles

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function
        :math:`g_0(X) = E[Y <= \\theta | X, D=d]` .

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

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    kde : callable or None
        A callable object / function with signature ``deriv = kde(u, weights)`` for weighted kernel density estimation.
        Here ``deriv`` should evaluate the density in ``0``.
        Default is ``'None'``, which uses :py:class:`statsmodels.nonparametric.kde.KDEUnivariate` with a
        gaussian kernel and silverman for bandwidth determination.

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
    >>> ml_g = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_pq_obj = dml.DoubleMLPQ(obj_dml_data, ml_g, ml_m, treatment=1, quantile=0.5)
    >>> dml_pq_obj.fit().summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.553878  0.149858  3.696011  0.000219  0.260161  0.847595
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment=1,
                 quantile=0.5,
                 n_folds=5,
                 n_rep=1,
                 score='PQ',
                 dml_procedure='dml2',
                 normalize_ipw=True,
                 kde=None,
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
        if kde is None:
            self._kde = _default_kde
        else:
            if not callable(kde):
                raise TypeError('kde should be either a callable or None. '
                                '%r was passed.' % kde)
            self._kde = kde

        self._normalize_ipw = normalize_ipw
        self._check_data(self._dml_data)

        valid_score = ['PQ']
        _check_score(self.score, valid_score, allow_callable=False)
        _check_quantile(self.quantile)
        _check_treatment(self.treatment)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

        # initialize starting values and bounds
        self._coef_bounds = (self._dml_data.y.min(), self._dml_data.y.max())
        self._coef_start_val = np.quantile(self._dml_data.y[self._dml_data.d == self.treatment], self.quantile)

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        _ = self._check_learner(ml_g, 'ml_g', regressor=False, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
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
    def kde(self):
        """
        The kernel density estimation of the derivative.
        """
        return self._kde

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

    @property
    def _score_element_names(self):
        return ['ind_d', 'g', 'm', 'y']

    def _compute_ipw_score(self, theta, d, y, prop):
        score = (d == self.treatment) / prop * (y <= theta) - self.quantile
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

        u = (y - coef).reshape(-1, 1)
        deriv = self.kde(u, score_weights)

        return deriv

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
        m_hat = copy.deepcopy(g_hat)

        ipw_vec = np.full(shape=self.n_folds, fill_value=np.nan)
        # initialize models
        fitted_models = {}
        for learner in self.params_names:
            # set nuisance model parameters
            est_params = self._get_params(learner)
            if est_params is not None:
                fitted_models[learner] = [clone(self._learner[learner]).set_params(**est_params[i_fold])
                                          for i_fold in range(self.n_folds)]
            else:
                fitted_models[learner] = [clone(self._learner[learner]) for i_fold in range(self.n_folds)]

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

            # get a copy of ml_m as a preliminary learner
            ml_m_prelim = clone(fitted_models['ml_m'][i_fold])
            m_hat_prelim = _dml_cv_predict(ml_m_prelim, x_train_1, d_train_1,
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

            _, bracket_guess = _get_bracket_guess(ipw_score, self._coef_start_val, self._coef_bounds)
            ipw_est = _solve_ipw_score(ipw_score=ipw_score, bracket_guess=bracket_guess)
            ipw_vec[i_fold] = ipw_est

            # use the preliminary estimates to fit the nuisance parameters on train_2
            d_train_2 = d[train_inds_2]
            y_train_2 = y[train_inds_2]
            x_train_2 = x[train_inds_2, :]

            dx_treat_train_2 = x_train_2[d_train_2 == self.treatment, :]
            y_treat_train_2 = y_train_2[d_train_2 == self.treatment]

            fitted_models['ml_g'][i_fold].fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

            # predict nuisance values on the test data and the corresponding targets
            g_hat['preds'][test_inds] = _predict_zero_one_propensity(fitted_models['ml_g'][i_fold], x[test_inds, :])
            g_hat['targets'][test_inds] = y[test_inds] <= ipw_est

            # refit the propensity score on the whole training set
            fitted_models['ml_m'][i_fold].fit(x[train_inds, :], d[train_inds])
            m_hat['preds'][test_inds] = _predict_zero_one_propensity(fitted_models['ml_m'][i_fold], x[test_inds, :])

        # set target for propensity score
        m_hat['targets'] = d

        # set the target for g to be a float and only relevant values
        g_hat['targets'] = g_hat['targets'].astype(float)
        g_hat['targets'][d != self.treatment] = np.nan

        if return_models:
            g_hat['models'] = fitted_models['ml_g']
            m_hat['models'] = fitted_models['ml_m']

        # clip propensities and normalize ipw weights
        # this is not done in the score to save computation due to multiple score evaluations
        # to be able to evaluate the raw models the m_hat['preds'] are not changed
        m_hat_adj = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)
        if self._normalize_ipw:
            if self.dml_procedure == 'dml1':
                for _, test_index in smpls:
                    m_hat_adj[test_index] = _normalize_ipw(m_hat_adj[test_index], d[test_index])
            else:
                m_hat_adj = _normalize_ipw(m_hat_adj, d)

        if self.treatment == 0:
            m_hat_adj = 1 - m_hat_adj

        # readjust start value for minimization
        self._coef_start_val = np.mean(ipw_vec)

        psi_elements = {'ind_d': d == self.treatment, 'g': g_hat['preds'],
                        'm': m_hat_adj, 'y': y}

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
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_treat = [np.intersect1d(np.where(d == self.treatment)[0], train) for train, _ in smpls]

        # use self._coef_start_val as a very crude approximation of ipw_est
        approx_goal = y <= np.quantile(y[d == self.treatment], self.quantile)
        g_tune_res = _dml_tune(approx_goal, x, train_inds_treat,
                               self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = _dml_tune(d, x, train_inds,
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

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        _check_contains_iv(obj_dml_data)
        _check_zero_one_treatment(self)
        return
