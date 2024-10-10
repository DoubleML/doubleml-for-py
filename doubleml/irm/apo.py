import numpy as np
import pandas as pd
import warnings

from sklearn.utils import check_X_y

from ..double_ml import DoubleML

from ..utils.blp import DoubleMLBLP
from ..double_ml_score_mixins import LinearScoreMixin

from ..utils._estimation import _dml_cv_predict, _dml_tune, _get_cond_smpls, _cond_targets, _trimm, \
    _normalize_ipw
from ..utils._checks import _check_score, _check_trimming, _check_weights, _check_finite_predictions, \
    _check_is_propensity, _check_binary_predictions


class DoubleMLAPO(LinearScoreMixin, DoubleML):
    """Double machine learning average potential outcomes for interactive regression models.

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

    treatment_level : int or float
        Chosen treatment level for average potential outcomes.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'APO'``) specifying the score function.
        Default is ``'APO'``.

    weights : array, dict or None
        An numpy array of weights for each individual observation. If None, then the ``'APO'`` score
        is applied (corresponds to weights equal to 1).
        An array has to be of shape ``(n,)``, where ``n`` is the number of observations.
        A dictionary can be used to specify weights which depend on the treatment variable.
        In this case, the dictionary has to contain two keys ``weights`` and ``weights_bar``, where the values
        have to be arrays of shape ``(n,)`` and ``(n, n_rep)``.
        Default is ``None``.

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

    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment_level,
                 n_folds=5,
                 n_rep=1,
                 score='APO',
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        # set up treatment level and check data
        self._treatment_level = treatment_level
        self._treated = self._dml_data.d == self._treatment_level

        self._check_data(self._dml_data)
        valid_scores = ['APO']
        _check_score(self.score, valid_scores, allow_callable=False)

        # set stratication for resampling
        self._strata = self._dml_data.d
        if draw_sample_splitting:
            self.draw_sample_splitting()

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

        self._normalize_ipw = normalize_ipw
        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

        # APO weights
        _check_weights(weights, score="ATE", n_obs=obj_dml_data.n_obs, n_rep=self.n_rep)
        self._initialize_weights(weights)

    @property
    def treatment_level(self):
        """
        Chosen treatment level for average potential outcomes.
        """
        return self._treatment_level

    @property
    def treated(self):
        """
        Indicator for treated observations (with the corresponding treatment level).
        """
        return self._treated

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
    def weights(self):
        """
        Specifies the weights for a weighted average potential outcome.
        """
        return self._weights

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _initialize_weights(self, weights):
        if weights is None:
            weights = np.ones(self._dml_data.n_obs)
        if isinstance(weights, np.ndarray):
            self._weights = {'weights': weights}
        else:
            assert isinstance(weights, dict)
            self._weights = weights

    def _get_weights(self):
        # standard case for APO/ATE
        weights = self._weights['weights']
        if 'weights_bar' not in self._weights.keys():
            weights_bar = self._weights['weights']
        else:
            weights_bar = self._weights['weights_bar'][:, self._i_rep]

        return weights, weights_bar

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        # use the treated indicator to get the correct sample splits
        x, treated = check_X_y(x, self.treated,
                               force_all_finite=False)

        # get train indices for d == treatment_level
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        g0_external = external_predictions['ml_g0'] is not None
        g1_external = external_predictions['ml_g1'] is not None
        m_external = external_predictions['ml_m'] is not None

        # nuisance g (g0 only relevant for sensitivity analysis)
        if g0_external:
            # use external predictions
            g_hat0 = {'preds': external_predictions['ml_g0'],
                      'targets': _cond_targets(y, cond_sample=(treated == 0)),
                      'models': None}
        else:
            g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
            g_hat0['targets'] = _cond_targets(g_hat0['targets'], cond_sample=(treated == 0))

        if self._dml_data.binary_outcome:
            _check_binary_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', self._dml_data.y_col)

        if g1_external:
            # use external predictions
            g_hat1 = {'preds': external_predictions['ml_g1'],
                      'targets': _cond_targets(y, cond_sample=(treated == 1)),
                      'models': None}
        else:
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat1['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1['targets'] = _cond_targets(g_hat1['targets'], cond_sample=(treated == 1))

        if self._dml_data.binary_outcome:
            _check_binary_predictions(g_hat1['preds'], self._learner['ml_g'], 'ml_g', self._dml_data.y_col)

        # nuisance m
        if m_external:
            # use external predictions
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': treated,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, treated, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        # also trimm external predictions
        m_hat['preds'] = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)

        psi_a, psi_b = self._score_elements(y, treated, g_hat0['preds'], g_hat1['preds'],
                                            m_hat['preds'], smpls)
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

    def _score_elements(self, y, treated, g_hat0, g_hat1, m_hat, smpls):
        if self.normalize_ipw:
            m_hat_adj = _normalize_ipw(m_hat, treated)
        else:
            m_hat_adj = m_hat

        u_hat = y - g_hat1
        weights, weights_bar = self._get_weights()
        psi_b = weights * g_hat1 + weights_bar * np.divide(np.multiply(treated, u_hat), m_hat_adj)
        psi_a = np.full_like(m_hat_adj, -1.0)

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        # set elments for readability
        y = self._dml_data.y
        treated = self.treated

        m_hat = preds['predictions']['ml_m']
        g_hat0 = preds['predictions']['ml_g0']
        g_hat1 = preds['predictions']['ml_g1']

        weights, weights_bar = self._get_weights()

        sigma2_score_element = np.square(y - np.multiply(treated, g_hat1) - np.multiply(1.0-treated, g_hat0))
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        # calc m(W,alpha) and Riesz representer
        m_alpha = np.multiply(weights, np.multiply(weights_bar, np.divide(1.0, m_hat)))
        rr = np.multiply(weights_bar, np.divide(treated, m_hat))

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
        x, treated = check_X_y(x, self.treated,
                               force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        g0_tune_res = _dml_tune(y, x, train_inds_d0,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g1_tune_res = _dml_tune(y, x, train_inds_d1,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = _dml_tune(treated, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g0': g0_best_params,
                  'ml_g1': g1_best_params,
                  'ml_m': m_best_params}
        tune_res = {'g0_tune': g0_tune_res,
                    'g1_tune': g1_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _check_data(self, obj_dml_data):
        if len(obj_dml_data.d_cols) > 1:
            raise ValueError('Only one treatment variable is allowed. ' +
                             f'Got {len(obj_dml_data.d_cols)} treatment variables.')

        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s).')

        # check if treatment level is valid
        if np.sum(self.treated) < 5:
            raise ValueError(
                'The number of treated observations is less than 5. ' +
                f'Number of treated observations: {np.sum(self.treated)} for treatment level {self.treatment_level}.'
            )

        if np.mean(self.treated) <= 0.05:
            warnings.warn(f'The proportion of observations with treatment level {self.treatment_level} is less than 5%.'
                          f' Got {np.mean(self.treated) * 100:.2f}%.')

        return

    def capo(self, basis, is_gate=False, **kwargs):
        """
        Calculate conditional average potential outcomes (CAPO) for a given basis.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
            where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

        is_gate : bool
            Indicates whether the basis is constructed for GATE/GAPOs (dummy-basis).
            Default is ``False``.

        **kwargs: dict
            Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit` e.g. ``cov_type``.

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model.
        """
        valid_score = ['APO']
        if self.score not in valid_score:
            raise ValueError('Invalid score ' + self.score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')

        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        # define the orthogonal signal
        orth_signal = self.psi_elements['psi_b'].reshape(-1)
        # fit the best linear predictor
        model = DoubleMLBLP(orth_signal, basis=basis, is_gate=is_gate)
        model.fit(**kwargs)
        return model

    def gapo(self, groups, **kwargs):
        """
        Calculate group average potential outcomes (GAPO) for groups.

        Parameters
        ----------
        groups : :class:`pandas.DataFrame`
            The group indicator for estimating the best linear predictor. Groups should be mutually exclusive.
            Has to be dummy coded with shape ``(n_obs, d)``, where ``n_obs`` is the number of observations
            and ``d`` is the number of groups or ``(n_obs, 1)`` and contain the corresponding groups (as str).

        **kwargs: dict
            Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit` e.g. ``cov_type``.

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model for group average potential outcomes.
        """
        if not isinstance(groups, pd.DataFrame):
            raise TypeError('Groups must be of DataFrame type. '
                            f'Groups of type {str(type(groups))} was passed.')

        if not all(groups.dtypes == bool) or all(groups.dtypes == int):
            if groups.shape[1] == 1:
                groups = pd.get_dummies(groups, prefix='Group', prefix_sep='_')
            else:
                raise TypeError('Columns of groups must be of bool type or int type (dummy coded). '
                                'Alternatively, groups should only contain one column.')

        if any(groups.sum(0) <= 5):
            warnings.warn('At least one group effect is estimated with less than 6 observations.')

        model = self.capo(groups, is_gate=True, **kwargs)
        return model
