from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import warnings

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData
from ..utils._estimation import (
    _trimm,
    _dml_cv_predict,
    _dml_tune,
    _get_cond_smpls_2d,
    _predict_zero_one_propensity)
from ..utils._checks import (
    _check_finite_predictions,
    _check_trimming,
    _check_score)
from ..double_ml_score_mixins import LinearScoreMixin


class DoubleMLSSM(LinearScoreMixin, DoubleML):
    """Double machine learning for sample selection models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g(S,D,X) = E[Y|S,D,X]`.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m(X) = Pr[D=1|X]`.

    ml_pi : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math: `\\pi(D,X) = Pr[S=1|D,X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'missing-at-random'`` or ``'nonignorable'``) specifying the score function.
        Default is ``'missing-at-random'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml import DoubleMLData
    >>> from sklearn.linear_model import LassoCV, LogisticRegressionCV()
    >>> from sklearn.base import clone
    >>> np.random.seed(3146)
    >>> n = 2000
    >>> p = 100
    >>> s = 2
    >>> sigma = np.array([[1, 0.5], [0.5, 1]])
    >>> e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n).T
    >>> X = np.random.randn(n, p)
    >>> beta = np.hstack((np.repeat(0.25, s), np.repeat(0, p - s)))
    >>> d = np.where(np.dot(X, beta) + np.random.randn(n) > 0, 1, 0)
    >>> z = np.random.randn(n)
    >>> s = np.where(np.dot(X, beta) + 0.25 * d + z + e[0] > 0, 1, 0)
    >>> y = np.dot(X, beta) + 0.5 * d + e[1]
    >>> y[s == 0] = 0
    >>> simul_data = DoubleMLData.from_arrays(X, y, d, z=None, t=s)
    >>> learner = LassoCV()
    >>> learner_class = LogisticRegressionCV()
    >>> ml_g_sim = clone(learner)
    >>> ml_pi_sim = clone(learner_class)
    >>> ml_m_sim = clone(learner_class)
    >>> obj_dml_sim = DoubleMLS(simul_data, ml_g_sim, ml_pi_sim, ml_m_sim)
    >>> obj_dml_sim.fit().summary
          coef   std err         t         P>|t|     2.5 %    97.5 %
    d  0.49135  0.070534  6.966097  3.258541e-12  0.353105  0.629595

    Notes
    -----
    Binary or multiple treatment effect evaluation with double machine learning under sample selection/outcome attrition.
    Potential outcomes Y(0) and Y(1) are estimated and ATE is returned as E[Y(1) - Y(0)].
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_pi,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='missing-at-random',
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        self._external_predictions_implemented = False
        self._sensitivity_implemented = False
        self._normalize_ipw = normalize_ipw

        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._check_data(self._dml_data)
        _check_score(self.score, ['missing-at-random', 'nonignorable'])

        # for both score function stratification by d and s is viable
        self._strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.s.reshape(-1, 1)
        if draw_sample_splitting:
            self.draw_sample_splitting()

        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_pi, 'ml_pi', regressor=False, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)

        self._learner = {'ml_g': clone(ml_g),
                         'ml_pi': clone(ml_pi),
                         'ml_m': clone(ml_m),
                         }
        self._predict_method = {'ml_g': 'predict',
                                'ml_pi': 'predict_proba',
                                'ml_m': 'predict_proba'
                                }
        if ml_g_is_classifier:
            if self._dml_data._check_binary_outcome():
                self._predict_method['ml_g'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome is not binary with values 0 and 1.')

        self._initialize_ml_nuisance_params()

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

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

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g_d0', 'ml_g_d1',
                         'ml_pi', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                        valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None and self._score == 'missing-at-random':
            warnings.warn(' and '.join(obj_dml_data.z_cols) +
                          ' have been set as instrumental variable(s). '
                          'You are estimating the effect under the assumption of data missing at random. \
                             Instrumental variables will not be used in estimation.')
        if obj_dml_data.z_cols is None and self._score == 'nonignorable':
            raise ValueError('Sample selection by nonignorable nonresponse was set but instrumental variable \
                             is None. To estimate treatment effect under nonignorable nonresponse, \
                             specify an instrument for the selection variable.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        x, s = check_X_y(x, self._dml_data.s, force_all_finite=False)

        if self._score == 'nonignorable':
            z, _ = check_X_y(self._dml_data.z, y, force_all_finite=False)
            dx = np.column_stack((x, d, z))
        else:
            dx = np.column_stack((x, d))

        _, smpls_d0_s1, _, smpls_d1_s1 = _get_cond_smpls_2d(smpls, d, s)

        if self._score == 'missing-at-random':
            pi_hat = _dml_cv_predict(self._learner['ml_pi'], dx, s, smpls=smpls, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_pi'), method=self._predict_method['ml_pi'],
                                     return_models=return_models)
            pi_hat['targets'] = pi_hat['targets'].astype(float)
            _check_finite_predictions(pi_hat['preds'], self._learner['ml_pi'], 'ml_pi', smpls)

            # propensity score m
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            m_hat['targets'] = m_hat['targets'].astype(float)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

            # conditional outcome
            g_hat_d1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1_s1, n_jobs=n_jobs_cv,
                                       est_params=self._get_params('ml_g_d1'), method=self._predict_method['ml_g'],
                                       return_models=return_models)
            g_hat_d1['targets'] = g_hat_d1['targets'].astype(float)
            _check_finite_predictions(g_hat_d1['preds'], self._learner['ml_g'], 'ml_g_d1', smpls)

            g_hat_d0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0_s1, n_jobs=n_jobs_cv,
                                       est_params=self._get_params('ml_g_d0'), method=self._predict_method['ml_g'],
                                       return_models=return_models)
            g_hat_d0['targets'] = g_hat_d0['targets'].astype(float)
            _check_finite_predictions(g_hat_d0['preds'], self._learner['ml_g'], 'ml_g_d0', smpls)

        else:
            assert self._score == 'nonignorable'
            # initialize nuisance predictions, targets and models
            g_hat_d1 = {'models': None,
                        'targets': np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                        'preds': np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
                        }
            g_hat_d0 = copy.deepcopy(g_hat_d1)
            pi_hat = copy.deepcopy(g_hat_d1)
            m_hat = copy.deepcopy(g_hat_d1)

            # pi_hat - used for preliminary estimation of propensity score pi, overwritten in each iteration
            pi_hat_prelim = {'models': None,
                             'targets': [],
                             'preds': []
                             }

            # initialize models
            fitted_models = {}
            for learner in self.params_names:
                # set nuisance model parameters
                est_params = self._get_params(learner)

                if learner == 'ml_g_d1' or learner == 'ml_g_d0':
                    nuisance = 'ml_g'
                else:
                    nuisance = learner

                if est_params is not None:
                    fitted_models[learner] = [
                        clone(self._learner[nuisance]).set_params(**est_params[i_fold]) for i_fold in range(self.n_folds)
                    ]
                else:
                    fitted_models[learner] = [clone(self._learner[nuisance]) for i_fold in range(self.n_folds)]

            # create strata for splitting
            strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.s.reshape(-1, 1)

            # calculate nuisance functions over different folds - nested cross-fitting
            for i_fold in range(self.n_folds):
                train_inds = smpls[i_fold][0]
                test_inds = smpls[i_fold][1]

                # start nested crossfitting - split training data into two equal parts
                train_inds_1, train_inds_2 = train_test_split(
                    train_inds, test_size=0.5, random_state=42, stratify=strata[train_inds]
                )

                s_train_1 = s[train_inds_1]
                dx_train_1 = dx[train_inds_1, :]

                # fit propensity score for selection on first part of training set
                fitted_models['ml_pi'][i_fold].fit(dx_train_1, s_train_1)
                pi_hat_prelim['preds'] = _predict_zero_one_propensity(fitted_models['ml_pi'][i_fold], dx)
                pi_hat_prelim['targets'] = s

                # predictions for small pi in denominator
                pi_hat['preds'][test_inds] = pi_hat_prelim['preds'][test_inds]
                pi_hat['targets'][test_inds] = s[test_inds]

                # add predicted selection propensity to covariates
                xpi = np.column_stack((x, pi_hat_prelim['preds']))

                # estimate propensity score m using the second training sample
                xpi_train_2 = xpi[train_inds_2, :]
                d_train_2 = d[train_inds_2]
                xpi_test = xpi[test_inds, :]

                fitted_models['ml_m'][i_fold].fit(xpi_train_2, d_train_2)

                m_hat['preds'][test_inds] = _predict_zero_one_propensity(fitted_models['ml_m'][i_fold], xpi_test)
                m_hat['targets'][test_inds] = d[test_inds]

                # estimate conditional outcome g on second training sample - treatment
                s1_d1_train_2_indices = np.intersect1d(np.where(d == 1)[0],
                                                       np.intersect1d(np.where(s == 1)[0], train_inds_2))
                xpi_s1_d1_train_2 = xpi[s1_d1_train_2_indices, :]
                y_s1_d1_train_2 = y[s1_d1_train_2_indices]

                fitted_models['ml_g_d1'][i_fold].fit(xpi_s1_d1_train_2, y_s1_d1_train_2)

                # predict conditional outcome
                g_hat_d1['preds'][test_inds] = fitted_models['ml_g_d1'][i_fold].predict(xpi_test)
                g_hat_d1['targets'][test_inds] = y[test_inds]

                # estimate conditional outcome on second training sample - control
                s1_d0_train_2_indices = np.intersect1d(np.where(d == 0)[0],
                                                       np.intersect1d(np.where(s == 1)[0], train_inds_2))
                xpi_s1_d0_train_2 = xpi[s1_d0_train_2_indices, :]
                y_s1_d0_train_2 = y[s1_d0_train_2_indices]

                fitted_models['ml_g_d0'][i_fold].fit(xpi_s1_d0_train_2, y_s1_d0_train_2)

                # predict conditional outcome
                g_hat_d0['preds'][test_inds] = fitted_models['ml_g_d0'][i_fold].predict(xpi_test)
                g_hat_d0['targets'][test_inds] = y[test_inds]

                if return_models:
                    g_hat_d1['models'] = fitted_models['ml_g_d1']
                    g_hat_d0['models'] = fitted_models['ml_g_d0']
                    pi_hat['models'] = fitted_models['ml_pi']
                    m_hat['models'] = fitted_models['ml_m']

        m_hat['preds'] = _trimm(m_hat['preds'], self._trimming_rule, self._trimming_threshold)

        # treatment indicator
        dtreat = (d == 1)
        dcontrol = (d == 0)

        psi_a, psi_b = self._score_elements(dtreat, dcontrol, g_hat_d1['preds'],
                                            g_hat_d0['preds'],
                                            pi_hat['preds'],
                                            m_hat['preds'],
                                            s, y)

        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}

        preds = {'predictions': {'ml_g_d0': g_hat_d0['preds'],
                                 'ml_g_d1': g_hat_d1['preds'],
                                 'ml_pi': pi_hat['preds'],
                                 'ml_m': m_hat['preds']},
                 'targets': {'ml_g_d0': g_hat_d0['targets'],
                             'ml_g_d1': g_hat_d1['targets'],
                             'ml_pi': pi_hat['targets'],
                             'ml_m': m_hat['targets']},
                 'models': {'ml_g_d0': g_hat_d0['models'],
                            'ml_g_d1': g_hat_d1['models'],
                            'ml_pi': pi_hat['models'],
                            'ml_m': m_hat['models']}
                 }

        return psi_elements, preds

    def _score_elements(self, dtreat, dcontrol, g_d1, g_d0,
                        pi, m, s, y):
        # psi_a
        psi_a = -1

        # psi_b
        if self._normalize_ipw:
            weight_treat = sum(dtreat) / sum((dtreat * s) / (pi * m))
            weight_control = sum(dcontrol) / sum((dcontrol * s) / (pi * (1 - m)))

            psi_b1 = weight_treat * ((dtreat * s * (y - g_d1)) / (m * pi)) + g_d1
            psi_b0 = weight_control * ((dcontrol * s * (y - g_d0)) / ((1 - m) * pi)) + g_d0

        else:
            psi_b1 = (dtreat * s * (y - g_d1)) / (m * pi) + g_d1
            psi_b0 = (dcontrol * s * (y - g_d0)) / ((1 - m) * pi) + g_d0

        psi_b = psi_b1 - psi_b0

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # time indicator is used for selection (selection not available in DoubleMLData yet)
        x, s = check_X_y(x, self._dml_data.s, force_all_finite=False)

        if self._score == 'nonignorable':
            z, _ = check_X_y(self._dml_data.z, y, force_all_finite=False)
            dx = np.column_stack((x, d, z))
        else:
            dx = np.column_stack((x, d))

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_pi': None,
                               'ml_m': None}

        # nuisance training sets conditional on d
        _, smpls_d0_s1, _, smpls_d1_s1 = _get_cond_smpls_2d(smpls, d, s)
        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0_s1 = [train_index for (train_index, _) in smpls_d0_s1]
        train_inds_d1_s1 = [train_index for (train_index, _) in smpls_d1_s1]

        # hyperparameter tuning for ML
        g_d0_tune_res = _dml_tune(y, x, train_inds_d0_s1,
                                  self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                  n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g_d1_tune_res = _dml_tune(y, x, train_inds_d1_s1,
                                  self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                  n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        pi_tune_res = _dml_tune(s, dx, train_inds,
                                self._learner['ml_pi'], param_grids['ml_pi'], scoring_methods['ml_pi'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_d0_best_params = [xx.best_params_ for xx in g_d0_tune_res]
        g_d1_best_params = [xx.best_params_ for xx in g_d1_tune_res]
        pi_best_params = [xx.best_params_ for xx in pi_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g_d0': g_d0_best_params,
                  'ml_g_d1': g_d1_best_params,
                  'ml_pi': pi_best_params,
                  'ml_m': m_best_params}

        tune_res = {'g_d0_tune': g_d0_tune_res,
                    'g_d1_tune': g_d1_tune_res,
                    'pi_tune': pi_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        pass
