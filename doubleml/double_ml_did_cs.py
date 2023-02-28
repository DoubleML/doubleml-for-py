import numpy as np
import pandas as pd
import warnings
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from .double_ml import DoubleML
from .double_ml_data import DoubleMLData
from .double_ml_score_mixins import LinearScoreMixin

from ._utils import _dml_cv_predict, _get_cond_smpls, _dml_tune, _check_finite_predictions, _check_is_propensity, \
    _trimm, _get_cond_smpls_2d


class DoubleMLDiDCS(LinearScoreMixin, DoubleML):
    """Double machine learning for difference-in-difference with repeated cross-sections.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(D,X) = E[Y|X,D]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'CS-1'`` to ``'CS-5'``) specifying the score function.
        Default is ``'CS-4'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

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
        Indicates whether cross-fitting should be applied.
        Default is ``True``.
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='CS-4',
                 dml_procedure='dml2',
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

        self._check_data(self._dml_data)
        self._check_score(self.score)

        # set stratication for resampling
        self._strata = self._dml_data.d.reshape(-1, 1) + \
            2 * self._dml_data.t.reshape(-1, 1)

        ml_g_is_classifier = self._check_learner(
            ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}

        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {
                    'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g_d0_t0', 'ml_g_d0_t1',
                         'ml_g_d1_t0', 'ml_g_d1_t1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['CS-4', 'CS*-4', 'CS-5', 'DR-1', 'DR-2', 'Chang']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        return

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('For repeated cross sections the data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'At the moment there are not DiD models with instruments implemented.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all(
            (np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an DiD model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')

        binary_time = (type_of_target(obj_dml_data.t) == 'binary')
        zero_one_time = np.all(
            (np.power(obj_dml_data.t, 2) - obj_dml_data.t) == 0)

        if not (binary_time & zero_one_time):
            raise ValueError('Incompatible data. '
                    'To fit an DiD model with DML '
                    'exactly one binary variable with values 0 and 1 '
                    'needs to be specified as time variable.')

        return

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        x, t = check_X_y(x, self._dml_data.t,
                         force_all_finite=False)

        # THIS DIFFERS FROM THE PAPER due to stratified splitting this should be the same for each fold
        # nuisance estimates of the uncond. treatment prob.
        p_hat = np.full_like(d, np.nan, dtype='float64')
        for train_index, test_index in smpls:
            p_hat[test_index] = np.mean(d[train_index])

        # nuisance estimates of the uncond. time prob.
        lambda_hat = np.full_like(t, np.nan, dtype='float64')
        for train_index, test_index in smpls:
            lambda_hat[test_index] = np.mean(t[train_index])

        # nuisance g
        smpls_d0_t0, smpls_d0_t1, smpls_d1_t0, smpls_d1_t1 = _get_cond_smpls_2d(
            smpls, d, t)

        g_hat_d0_t0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d0_t0, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_g_d0_t0'), method=self._predict_method['ml_g'],
                                      return_models=return_models)
        g_hat_d0_t0['targets'] = g_hat_d0_t0['targets'].astype(float)
        g_hat_d0_t0['targets'][np.invert((d == 0) & (t == 0))] = np.nan

        g_hat_d0_t1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d0_t1, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_g_d0_t1'), method=self._predict_method['ml_g'],
                                      return_models=return_models)
        g_hat_d0_t1['targets'] = g_hat_d0_t1['targets'].astype(float)
        g_hat_d0_t1['targets'][np.invert((d == 0) & (t == 1))] = np.nan

        g_hat_d1_t0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d1_t0, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_g_d1_t0'), method=self._predict_method['ml_g'],
                                      return_models=return_models)
        g_hat_d1_t0['targets'] = g_hat_d1_t0['targets'].astype(float)
        g_hat_d1_t0['targets'][np.invert((d == 1) & (t == 0))] = np.nan

        g_hat_d1_t1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d1_t1, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_g_d1_t1'), method=self._predict_method['ml_g'],
                                      return_models=return_models)
        g_hat_d1_t1['targets'] = g_hat_d1_t1['targets'].astype(float)
        g_hat_d1_t1['targets'][np.invert((d == 1) & (t == 1))] = np.nan

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                return_models=return_models)
        _check_finite_predictions(
            m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
        _check_is_propensity(
            m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        psi_a, psi_b = self._score_elements(y, d, t,
                                            g_hat_d0_t0['preds'], g_hat_d0_t1['preds'],
                                            g_hat_d1_t0['preds'], g_hat_d1_t1['preds'],
                                            m_hat['preds'], p_hat, lambda_hat)

        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_g_d0_t0': g_hat_d0_t0['preds'],
                                 'ml_g_d0_t1': g_hat_d0_t1['preds'],
                                 'ml_g_d1_t0': g_hat_d1_t0['preds'],
                                 'ml_g_d1_t1': g_hat_d1_t1['preds'],
                                 'ml_m': m_hat['preds']},
                 'targets': {'ml_g_d0_t0': g_hat_d0_t0['targets'],
                             'ml_g_d0_t1': g_hat_d0_t1['targets'],
                             'ml_g_d1_t0': g_hat_d1_t0['targets'],
                             'ml_g_d1_t1': g_hat_d1_t1['targets'],
                             'ml_m': m_hat['targets']},
                 'models': {'ml_g_d0_t0': g_hat_d0_t0['models'],
                            'ml_g_d0_t1': g_hat_d0_t1['models'],
                            'ml_g_d1_t0': g_hat_d1_t0['models'],
                            'ml_g_d1_t1': g_hat_d1_t1['models'],
                            'ml_m': m_hat['models']}
                 }

        return psi_elements, preds

    def _score_elements(self, y, d, t,
                        g_hat_d0_t0, g_hat_d0_t1,
                        g_hat_d1_t0, g_hat_d1_t1,
                        m_hat, p_hat, lambda_hat):

        # trimm propensities
        m_hat = _trimm(m_hat, self.trimming_rule, self.trimming_threshold)
        # calculate residuals
        resid_d0_t0 = y - g_hat_d0_t0
        resid_d0_t1 = y - g_hat_d0_t1
        resid_d1_t0 = y - g_hat_d1_t0
        resid_d1_t1 = y - g_hat_d1_t1

        # default term for chang and not efficient Zimmert to correct for the form of the score
        psi_b_3 = np.zeros_like(y)

        if self.score == 'CS-4':
            weight_psi_a = np.divide(d, p_hat)
            weight_g_d1_t1 = weight_psi_a
            weight_g_d1_t0 = -1.0 * weight_psi_a
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.divide(np.multiply(
                d, t), np.multiply(p_hat, lambda_hat))
            weight_resid_d1_t0 = -1.0 * \
                np.divide(np.multiply(d, 1.0-t),
                          np.multiply(p_hat, 1.0-lambda_hat))
            
            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            weight_resid_d0_t1 = -1.0 * np.multiply(np.divide(np.multiply(1.0-d, t), np.multiply(p_hat, lambda_hat)),
                                                    prop_weighting)
            weight_resid_d0_t0 = np.multiply(np.divide(np.multiply(1.0-d, 1.0-t), np.multiply(p_hat, 1.0-lambda_hat)),
                                             prop_weighting)
        elif self.score == 'CS*-4':
            # todo: improve this implementation
            weight_psi_a = np.divide(d, p_hat)
            weight_g_d1_t1 = np.zeros_like(weight_psi_a)
            weight_g_d1_t0 = np.zeros_like(weight_psi_a)
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.zeros_like(weight_psi_a)
            weight_resid_d1_t0 = np.zeros_like(weight_psi_a)

            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            weight_resid_d0_t1 = -1.0 * np.multiply(np.divide(np.multiply(1.0-d, t), np.multiply(p_hat, lambda_hat)),
                                                    prop_weighting)
            weight_resid_d0_t0 = np.multiply(np.divide(np.multiply(1.0-d, 1.0-t), np.multiply(p_hat, 1.0-lambda_hat)),
                                             prop_weighting)
            
            psi_b_3 = np.multiply(np.divide(np.multiply(d, t), np.multiply(p_hat, lambda_hat))
                                  - np.divide(np.multiply(d, 1.0-t), np.multiply(p_hat, 1.0-lambda_hat)), y)

        elif self.score == 'CS-5':
            # todo: improve this implementation
            weight_psi_a = np.ones_like(d)
            weight_g_d1_t1 = weight_psi_a
            weight_g_d1_t0 = -1.0 * weight_psi_a
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.divide(np.multiply(
                d, t), np.multiply(p_hat, lambda_hat))
            weight_resid_d1_t0 = -1.0 * \
                np.divide(np.multiply(d, 1.0-t),
                          np.multiply(p_hat, 1.0-lambda_hat))
            weight_resid_d0_t1 = -1.0 * \
                np.divide(np.multiply(1.0-d, t),
                          np.multiply(1.0-p_hat, lambda_hat))
            weight_resid_d0_t0 = np.divide(np.multiply(
                1.0-d, 1.0-t), np.multiply(1.0-p_hat, 1.0-lambda_hat))

        elif self.score == 'DR-1':
            weight_psi_a = np.ones_like(y)
            weight_g_d1_t1 = np.zeros_like(weight_psi_a)
            weight_g_d1_t0 = np.zeros_like(weight_psi_a)
            weight_g_d0_t1 = np.zeros_like(weight_psi_a)
            weight_g_d0_t0 = np.zeros_like(weight_psi_a)

            weight_resid_d1_t1 = np.zeros_like(weight_psi_a)
            weight_resid_d1_t0 = np.zeros_like(weight_psi_a)

            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            scaling_d0_t1 = np.mean(np.multiply(
                np.multiply(1.0-d, t), prop_weighting))
            weight_resid_d0_t1 = np.multiply(t, (np.divide(d, np.mean(np.multiply(d, t)))
                                                 - np.divide(np.multiply(1.0-d, prop_weighting),
                                                             scaling_d0_t1)))

            scaling_d0_t0 = np.mean(np.multiply(
                np.multiply(1.0-d, 1.0-t), prop_weighting))
            weight_resid_d0_t0 = -1.0 * np.multiply(1.0-t, (np.divide(d, np.mean(np.multiply(d, 1.0-t)))
                                                            - np.divide(np.multiply(1.0-d, prop_weighting),
                                                                        scaling_d0_t0)))

        elif self.score == 'DR-2':
            weight_psi_a = np.ones_like(y)
            weight_g_d1_t1 = np.divide(d, np.mean(d))
            weight_g_d1_t0 = -1.0 * np.divide(d, np.mean(d))
            weight_g_d0_t1 = weight_g_d1_t0
            weight_g_d0_t0 = weight_g_d1_t1

            weight_resid_d1_t1 = np.divide(
                np.multiply(d, t), np.mean(np.multiply(d, t)))
            weight_resid_d1_t0 = -1.0 * \
                np.divide(np.multiply(d, 1.0-t),
                          np.mean(np.multiply(d, 1.0-t)))

            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            scaling_d0_t1 = np.mean(np.multiply(
                np.multiply(1.0-d, t), prop_weighting))
            weight_resid_d0_t1 = -1.0 * \
                np.divide(np.multiply(np.multiply(1.0-d, t),
                          prop_weighting), scaling_d0_t1)

            scaling_d0_t0 = np.mean(np.multiply(
                np.multiply(1.0-d, 1.0-t), prop_weighting))
            weight_resid_d0_t0 = np.divide(np.multiply(
                np.multiply(1.0-d, 1.0-t), prop_weighting), scaling_d0_t0)

        elif self.score == 'Chang':
            weight_psi_a = np.divide(d, p_hat)
            weight_g_d1_t1 = np.zeros_like(weight_psi_a)
            weight_g_d1_t0 = np.zeros_like(weight_psi_a)
            weight_g_d0_t1 = np.zeros_like(weight_psi_a)
            weight_g_d0_t0 = np.zeros_like(weight_psi_a)

            weight_resid_d1_t1 = np.zeros_like(weight_psi_a)
            weight_resid_d1_t0 = np.zeros_like(weight_psi_a)

            # part of the weights for t==0 and t==1
            weight_d0 = d - np.multiply(1.0-d, np.divide(m_hat, 1.0-m_hat))
            # calc weight für resid_d0_t1
            weight_resid_d0_t1_1 = np.divide(t, np.divide(lambda_hat, p_hat))
            weight_resid_d0_t1 = np.multiply(weight_resid_d0_t1_1, weight_d0)
            # calc weight für resid_d0_t0
            weight_resid_d0_t0_1 = np.divide(
                1.0-t, np.divide(1.0-lambda_hat, p_hat))
            weight_resid_d0_t0 = -1.0 * \
                np.multiply(weight_resid_d0_t0_1, weight_d0)

            resid_d0 = np.multiply(t, y-resid_d0_t1) + \
                np.multiply(1.0-t, y-resid_d0_t0)
            G_2lambda_weight = np.divide(
                d-m_hat, np.multiply(p_hat, 1.0-m_hat))
            G_2lambda_1 = np.multiply(np.divide(
                1-2*lambda_hat, np.square(np.multiply(lambda_hat, 1.0-lambda_hat))), G_2lambda_weight)
            G_2lambda_2 = np.multiply(np.divide(1, np.multiply(
                lambda_hat, 1.0-lambda_hat)), G_2lambda_weight)
            G_2lambda = np.mean(np.multiply(G_2lambda_1, np.multiply(
                t-lambda_hat, resid_d0)) - np.multiply(G_2lambda_2, y))
            psi_b_3 = np.multiply(G_2lambda, t - lambda_hat)

        # set score elements
        psi_a = -1.0 * weight_psi_a

        # psi_b
        psi_b_1 = np.multiply(weight_g_d1_t1,  g_hat_d1_t1) + np.multiply(weight_g_d1_t0,  g_hat_d1_t0) \
            + np.multiply(weight_g_d0_t0,  g_hat_d0_t0) + \
            np.multiply(weight_g_d0_t1,  g_hat_d0_t1)
        psi_b_2 = np.multiply(weight_resid_d1_t1,  resid_d1_t1) + np.multiply(weight_resid_d1_t0,  resid_d1_t0) \
            + np.multiply(weight_resid_d0_t0,  resid_d0_t0) + \
            np.multiply(weight_resid_d0_t1,  resid_d0_t1)

        psi_b = psi_b_1 + psi_b_2 + psi_b_3

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        pass
