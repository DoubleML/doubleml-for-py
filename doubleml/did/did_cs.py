import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
import warnings

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData
from ..double_ml_score_mixins import LinearScoreMixin

from ..utils._estimation import _dml_cv_predict, _trimm, _get_cond_smpls_2d, _dml_tune
from ..utils._checks import _check_score, _check_trimming, _check_finite_predictions, _check_is_propensity


class DoubleMLDIDCS(LinearScoreMixin, DoubleML):
    """Double machine learning for difference-in-difference with repeated cross-sections.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(d,t,X) = E[Y|D=d,T=t,X]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=1|X]`.
        Only relevant for ``score='observational'``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'observational'`` or ``'experimental'``) specifying the score function.
        The ``'experimental'`` scores refers to an A/B setting, where the treatment is independent
        from the pretreatment covariates.
        Default is ``'observational'``.

    in_sample_normalization : bool
        Indicates whether to use a sligthly different normalization from Sant'Anna and Zhao (2020).
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

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_did_SZ2020
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(42)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5)
    >>> data = make_did_SZ2020(n_obs=500, cross_sectional_data=True, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', t_col='t')
    >>> dml_did_obj = dml.DoubleMLDIDCS(obj_dml_data, ml_g, ml_m)
    >>> dml_did_obj.fit().summary
           coef   std err         t     P>|t|      2.5 %     97.5 %
    d -6.604603  8.725802 -0.756905  0.449107 -23.706862  10.497655
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m=None,
                 n_folds=5,
                 n_rep=1,
                 score='observational',
                 in_sample_normalization=True,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        self._check_data(self._dml_data)
        valid_scores = ['observational', 'experimental']
        _check_score(self.score, valid_scores, allow_callable=False)

        self._in_sample_normalization = in_sample_normalization
        if not isinstance(self.in_sample_normalization, bool):
            raise TypeError('in_sample_normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.in_sample_normalization))} passed.')

        # set stratication for resampling
        self._strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.t.reshape(-1, 1)
        if draw_sample_splitting:
            self.draw_sample_splitting()

        # check learners
        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        if self.score == 'observational':
            _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
            self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        else:
            assert self.score == 'experimental'
            if ml_m is not None:
                warnings.warn(('A learner ml_m has been provided for score = "experimental" but will be ignored. '
                               'A learner ml_m is not required for estimation.'))
            self._learner = {'ml_g': ml_g}

        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict'}

        if 'ml_m' in self._learner:
            self._predict_method['ml_m'] = 'predict_proba'
        self._initialize_ml_nuisance_params()

        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

    @property
    def in_sample_normalization(self):
        """
        Indicates whether the in sample normalization of weights are used.
        """
        return self._in_sample_normalization

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
        if self.score == 'observational':
            valid_learner = ['ml_g_d0_t0', 'ml_g_d0_t1',
                             'ml_g_d1_t0', 'ml_g_d1_t1', 'ml_m']
        else:
            assert self.score == 'experimental'
            valid_learner = ['ml_g_d0_t0', 'ml_g_d0_t1',
                             'ml_g_d1_t0', 'ml_g_d1_t1']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('For repeated cross sections the data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'At the moment there are no DiD models with instruments implemented.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all(
            (np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an DIDCS model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')

        binary_time = (type_of_target(obj_dml_data.t) == 'binary')
        zero_one_time = np.all(
            (np.power(obj_dml_data.t, 2) - obj_dml_data.t) == 0)

        if not (binary_time & zero_one_time):
            raise ValueError('Incompatible data. '
                             'To fit an DIDCS model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as time variable.')

        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
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
        smpls_d0_t0, smpls_d0_t1, smpls_d1_t0, smpls_d1_t1 = _get_cond_smpls_2d(smpls, d, t)
        if external_predictions['ml_g_d0_t0'] is not None:
            g_hat_d0_t0 = {'preds': external_predictions['ml_g_d0_t0'],
                           'targets': None,
                           'models': None}
        else:
            g_hat_d0_t0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d0_t0, n_jobs=n_jobs_cv,
                                          est_params=self._get_params('ml_g_d0_t0'), method=self._predict_method['ml_g'],
                                          return_models=return_models)

            g_hat_d0_t0['targets'] = g_hat_d0_t0['targets'].astype(float)
            g_hat_d0_t0['targets'][np.invert((d == 0) & (t == 0))] = np.nan
        if external_predictions['ml_g_d0_t1'] is not None:
            g_hat_d0_t1 = {'preds': external_predictions['ml_g_d0_t1'],
                           'targets': None,
                           'models': None}
        else:
            g_hat_d0_t1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d0_t1, n_jobs=n_jobs_cv,
                                          est_params=self._get_params('ml_g_d0_t1'), method=self._predict_method['ml_g'],
                                          return_models=return_models)
            g_hat_d0_t1['targets'] = g_hat_d0_t1['targets'].astype(float)
            g_hat_d0_t1['targets'][np.invert((d == 0) & (t == 1))] = np.nan
        if external_predictions['ml_g_d1_t0'] is not None:
            g_hat_d1_t0 = {'preds': external_predictions['ml_g_d1_t0'],
                           'targets': None,
                           'models': None}
        else:
            g_hat_d1_t0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d1_t0, n_jobs=n_jobs_cv,
                                          est_params=self._get_params('ml_g_d1_t0'), method=self._predict_method['ml_g'],
                                          return_models=return_models)
            g_hat_d1_t0['targets'] = g_hat_d1_t0['targets'].astype(float)
            g_hat_d1_t0['targets'][np.invert((d == 1) & (t == 0))] = np.nan
        if external_predictions['ml_g_d1_t1'] is not None:
            g_hat_d1_t1 = {'preds': external_predictions['ml_g_d1_t1'],
                           'targets': None,
                           'models': None}
        else:
            g_hat_d1_t1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls_d1_t1, n_jobs=n_jobs_cv,
                                          est_params=self._get_params('ml_g_d1_t1'), method=self._predict_method['ml_g'],
                                          return_models=return_models)
            g_hat_d1_t1['targets'] = g_hat_d1_t1['targets'].astype(float)
            g_hat_d1_t1['targets'][np.invert((d == 1) & (t == 1))] = np.nan

        # only relevant for observational or experimental setting
        m_hat = {'preds': None, 'targets': None, 'models': None}
        if self.score == 'observational':
            # nuisance m
            if external_predictions['ml_m'] is not None:
                m_hat = {'preds': external_predictions['ml_m'],
                         'targets': None,
                         'models': None}
            else:
                m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                        return_models=return_models)
                _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
                _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)
            m_hat['preds'] = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)

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

        # calculate residuals
        resid_d0_t0 = y - g_hat_d0_t0
        resid_d0_t1 = y - g_hat_d0_t1
        resid_d1_t0 = y - g_hat_d1_t0
        resid_d1_t1 = y - g_hat_d1_t1

        d1t1 = np.multiply(d, t)
        d1t0 = np.multiply(d, 1.0-t)
        d0t1 = np.multiply(1.0-d, t)
        d0t0 = np.multiply(1.0-d, 1.0-t)

        if self.score == 'observational':
            if self.in_sample_normalization:
                weight_psi_a = np.divide(d, np.mean(d))
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.mean(d1t1))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.mean(d1t0))

                prop_weighting = np.divide(m_hat, 1.0-m_hat)
                unscaled_d0_t1 = np.multiply(d0t1, prop_weighting)
                weight_resid_d0_t1 = -1.0 * np.divide(unscaled_d0_t1, np.mean(unscaled_d0_t1))

                unscaled_d0_t0 = np.multiply(d0t0, prop_weighting)
                weight_resid_d0_t0 = np.divide(unscaled_d0_t0, np.mean(unscaled_d0_t0))
            else:
                weight_psi_a = np.divide(d, p_hat)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.multiply(p_hat, lambda_hat))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.multiply(p_hat, 1.0-lambda_hat))

                prop_weighting = np.divide(m_hat, 1.0-m_hat)
                weight_resid_d0_t1 = -1.0 * np.multiply(np.divide(d0t1, np.multiply(p_hat, lambda_hat)),
                                                        prop_weighting)
                weight_resid_d0_t0 = np.multiply(np.divide(d0t0, np.multiply(p_hat, 1.0-lambda_hat)),
                                                 prop_weighting)
        else:
            assert self.score == 'experimental'
            if self.in_sample_normalization:
                weight_psi_a = np.ones_like(y)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.mean(d1t1))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.mean(d1t0))
                weight_resid_d0_t1 = -1.0 * np.divide(d0t1, np.mean(d0t1))
                weight_resid_d0_t0 = np.divide(d0t0, np.mean(d0t0))
            else:
                weight_psi_a = np.ones_like(y)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.multiply(p_hat, lambda_hat))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.multiply(p_hat, 1.0-lambda_hat))
                weight_resid_d0_t1 = -1.0 * np.divide(d0t1, np.multiply(1.0-p_hat, lambda_hat))
                weight_resid_d0_t0 = np.divide(d0t0, np.multiply(1.0-p_hat, 1.0-lambda_hat))

        # set score elements
        psi_a = -1.0 * weight_psi_a

        # psi_b
        psi_b_1 = np.multiply(weight_g_d1_t1,  g_hat_d1_t1) + \
            np.multiply(weight_g_d1_t0,  g_hat_d1_t0) + \
            np.multiply(weight_g_d0_t0,  g_hat_d0_t0) + \
            np.multiply(weight_g_d0_t1,  g_hat_d0_t1)
        psi_b_2 = np.multiply(weight_resid_d1_t1,  resid_d1_t1) + \
            np.multiply(weight_resid_d1_t0,  resid_d1_t0) + \
            np.multiply(weight_resid_d0_t0,  resid_d0_t0) + \
            np.multiply(weight_resid_d0_t1,  resid_d0_t1)

        psi_b = psi_b_1 + psi_b_2

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        y = self._dml_data.y
        d = self._dml_data.d
        t = self._dml_data.t

        m_hat = preds['predictions']['ml_m']
        g_hat_d0_t0 = preds['predictions']['ml_g_d0_t0']
        g_hat_d0_t1 = preds['predictions']['ml_g_d0_t1']
        g_hat_d1_t0 = preds['predictions']['ml_g_d1_t0']
        g_hat_d1_t1 = preds['predictions']['ml_g_d1_t1']

        d0t0 = np.multiply(1.0-d, 1.0-t)
        d0t1 = np.multiply(1.0-d, t)
        d1t0 = np.multiply(d, 1.0-t)
        d1t1 = np.multiply(d, t)

        g_hat = np.multiply(d0t0, g_hat_d0_t0) + np.multiply(d0t1, g_hat_d0_t1) + \
            np.multiply(d1t0, g_hat_d1_t0) + np.multiply(d1t1, g_hat_d1_t1)
        sigma2_score_element = np.square(y - g_hat)
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        # calc m(W,alpha) and Riesz representer
        p_hat = np.mean(d)
        lambda_hat = np.mean(t)
        if self.score == 'observational':
            propensity_weight_d0 = np.divide(m_hat, 1.0-m_hat)
            if self.in_sample_normalization:
                weight_d0t1 = np.multiply(d0t1, propensity_weight_d0)
                weight_d0t0 = np.multiply(d0t0, propensity_weight_d0)
                mean_weight_d0t1 = np.mean(weight_d0t1)
                mean_weight_d0t0 = np.mean(weight_d0t0)

                m_alpha = np.multiply(np.divide(d, p_hat),
                                      np.divide(1.0, np.mean(d1t1)) +
                                      np.divide(1.0, np.mean(d1t0)) +
                                      np.divide(propensity_weight_d0, mean_weight_d0t1) +
                                      np.divide(propensity_weight_d0, mean_weight_d0t0))

                rr = np.divide(d1t1, np.mean(d1t1)) - \
                    np.divide(d1t0, np.mean(d1t0)) - \
                    np.divide(weight_d0t1, mean_weight_d0t1) + \
                    np.divide(weight_d0t0, mean_weight_d0t0)
            else:
                m_alpha_1 = np.divide(1.0, lambda_hat) + np.divide(1.0, 1.0-lambda_hat)
                m_alpha = np.multiply(np.divide(d, np.square(p_hat)), np.multiply(m_alpha_1, 1.0 + propensity_weight_d0))

                rr_1 = np.divide(t, np.multiply(p_hat, lambda_hat)) + np.divide(1.0-t, np.multiply(p_hat, 1.0-lambda_hat))
                rr_2 = d + np.multiply(1.0-d, propensity_weight_d0)
                rr = np.multiply(rr_1, rr_2)
        else:
            assert self.score == 'experimental'
            if self.in_sample_normalization:
                m_alpha = np.divide(1.0, np.mean(d1t1)) + \
                    np.divide(1.0, np.mean(d1t0)) + \
                    np.divide(1.0, np.mean(d0t1)) + \
                    np.divide(1.0, np.mean(d0t0))
                rr = np.divide(d1t1, np.mean(d1t1)) - \
                    np.divide(d1t0, np.mean(d1t0)) - \
                    np.divide(d0t1, np.mean(d0t1)) + \
                    np.divide(d0t0, np.mean(d0t0))
            else:
                m_alpha = np.divide(1.0, np.multiply(p_hat, lambda_hat)) + \
                    np.divide(1.0, np.multiply(p_hat, 1.0-lambda_hat)) + \
                    np.divide(1.0, np.multiply(1.0-p_hat, lambda_hat)) + \
                    np.divide(1.0, np.multiply(1.0-p_hat, 1.0-lambda_hat))
                rr = np.divide(d1t1, np.multiply(p_hat, lambda_hat)) - \
                    np.divide(d1t0, np.multiply(p_hat, 1.0-lambda_hat)) - \
                    np.divide(d0t1, np.multiply(1.0-p_hat, lambda_hat)) + \
                    np.divide(d0t0, np.multiply(1.0-p_hat, 1.0-lambda_hat))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2 = np.mean(nu2_score_element)
        psi_nu2 = nu2_score_element - nu2

        element_dict = {'sigma2': sigma2,
                        'nu2': nu2,
                        'psi_sigma2': psi_sigma2,
                        'psi_nu2': psi_nu2,
                        'riesz_rep': rr,
                        }
        return element_dict

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        x, t = check_X_y(x, self._dml_data.t,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        # nuisance training sets conditional on d and t
        smpls_d0_t0, smpls_d0_t1, smpls_d1_t0, smpls_d1_t1 = _get_cond_smpls_2d(smpls, d, t)
        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0_t0 = [train_index for (train_index, _) in smpls_d0_t0]
        train_inds_d0_t1 = [train_index for (train_index, _) in smpls_d0_t1]
        train_inds_d1_t0 = [train_index for (train_index, _) in smpls_d1_t0]
        train_inds_d1_t1 = [train_index for (train_index, _) in smpls_d1_t1]

        g_d0_t0_tune_res = _dml_tune(y, x, train_inds_d0_t0,
                                     self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                     n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_d0_t1_tune_res = _dml_tune(y, x, train_inds_d0_t1,
                                     self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                     n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_d1_t0_tune_res = _dml_tune(y, x, train_inds_d1_t0,
                                     self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                     n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_d1_t1_tune_res = _dml_tune(y, x, train_inds_d1_t1,
                                     self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                     n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = list()
        if self.score == 'observational':
            m_tune_res = _dml_tune(d, x, train_inds,
                                   self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                                   n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_d0_t0_best_params = [xx.best_params_ for xx in g_d0_t0_tune_res]
        g_d0_t1_best_params = [xx.best_params_ for xx in g_d0_t1_tune_res]
        g_d1_t0_best_params = [xx.best_params_ for xx in g_d1_t0_tune_res]
        g_d1_t1_best_params = [xx.best_params_ for xx in g_d1_t1_tune_res]

        if self.score == 'observational':
            m_best_params = [xx.best_params_ for xx in m_tune_res]
            params = {'ml_g_d0_t0': g_d0_t0_best_params,
                      'ml_g_d0_t1': g_d0_t1_best_params,
                      'ml_g_d1_t0': g_d1_t0_best_params,
                      'ml_g_d1_t1': g_d1_t1_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g_d0_t0_tune': g_d0_t0_tune_res,
                        'g_d0_t1_tune': g_d0_t1_tune_res,
                        'g_d1_t0_tune': g_d1_t0_tune_res,
                        'g_d1_t1_tune': g_d1_t1_tune_res,
                        'm_tune': m_tune_res}
        else:
            params = {'ml_g_d0_t0': g_d0_t0_best_params,
                      'ml_g_d0_t1': g_d0_t1_best_params,
                      'ml_g_d1_t0': g_d1_t0_best_params,
                      'ml_g_d1_t1': g_d1_t1_best_params}
            tune_res = {'g_d0_t0_tune': g_d0_t0_tune_res,
                        'g_d0_t1_tune': g_d0_t1_tune_res,
                        'g_d1_t0_tune': g_d1_t0_tune_res,
                        'g_d1_t1_tune': g_d1_t1_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
