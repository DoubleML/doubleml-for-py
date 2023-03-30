import numpy as np
import pandas as pd
import warnings
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from .double_ml import DoubleML
from .double_ml_data import DoubleMLData
from .double_ml_score_mixins import LinearScoreMixin

from ._utils import _dml_cv_predict, _get_cond_smpls, _dml_tune, _check_finite_predictions, _check_is_propensity, \
    _trimm


class DoubleMLDID(LinearScoreMixin, DoubleML):
    """Double machine learning for difference-in-differences models with panel data (two time periods).

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
        A str (``'PA-1'``, ``'PA-2'`` or ``'DR'``) specifying the score function.
        Default is ``'PA-1'``.

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
                 score='PA-1',
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
        self._strata = self._dml_data.d


        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}

        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
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
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['PA-1', 'PA-2', 'DR']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        return

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('For repeated outcomes the data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'At the moment there are not DiD models with instruments implemented.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an DiD model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return


    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance g
        # get train indices for d == 0
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'],
                                 return_models=return_models)
    
        _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
        # adjust target values to consider only compatible subsamples
        g_hat0['targets'] = g_hat0['targets'].astype(float)
        g_hat0['targets'][d == 1] = np.nan
        
        # only relevant for experimental setting PA-2
        g_hat1 = {'preds': None, 'targets': None, 'models': None}
        if self.score == 'PA-2':
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'],
                                return_models=return_models)

            _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1['targets'] = g_hat1['targets'].astype(float)
            g_hat1['targets'][d == 0] = np.nan

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                return_models=return_models)
        _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
        _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        # nuisance estimates of the uncond. treatment prob.
        p_hat = np.full_like(d, np.nan, dtype='float64')
        for train_index, test_index in smpls:
            p_hat[test_index] = np.mean(d[train_index])

        psi_a, psi_b = self._score_elements(y, d, g_hat0['preds'], g_hat1['preds'], m_hat['preds'], p_hat)

        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_g0': g_hat0['preds'],
                                 'ml_g1': g_hat1['preds'],
                                 'ml_m': m_hat['preds']},
                'targets': {'ml_g0': g_hat0['targets'],
                            'ml_g1': g_hat1['targets'],
                            'ml_m': m_hat['targets']},
                'models': {'ml_g0': g_hat0['models'],
                           'ml_g1': g_hat1['models'],
                           'ml_m': m_hat['models']}
                }

        return psi_elements, preds


    def _score_elements(self, y, d, g_hat0, g_hat1, m_hat, p_hat):
        # trimm propensities and calc residuals
        m_hat = _trimm(m_hat, self.trimming_rule, self.trimming_threshold)
        y_resid_d0 = y - g_hat0

        if self.score == 'PA-1':
            psi_a = -1.0 * np.divide(d, p_hat)
            y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
            psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)
        
        elif self.score == 'PA-2':
            psi_a = -1.0 * np.ones_like(d)
            y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
            psi_b_1 = np.multiply(y_resid_d0_weight, y_resid_d0)
            psi_b_2 = np.multiply(1.0-np.divide(d, p_hat), g_hat1 - g_hat0)
            psi_b = psi_b_1 + psi_b_2

        else:
            assert self.score == 'DR'
            psi_a = -1.0 * np.divide(d, np.mean(d))
            propensity_weight = np.divide(m_hat, 1.0-m_hat)
            y_resid_d0_weight = np.divide(d, np.mean(d)) \
                - np.divide(np.multiply(1.0-d, propensity_weight), np.mean(np.multiply(1.0-d, propensity_weight)))
            psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)

        return psi_a, psi_b


    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                        search_mode, n_iter_randomized_search):
        pass
