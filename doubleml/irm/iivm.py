import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData
from ..double_ml_score_mixins import LinearScoreMixin

from ..utils._estimation import _dml_cv_predict, _get_cond_smpls, _dml_tune, _trimm, _normalize_ipw
from ..utils._checks import _check_score, _check_trimming, _check_finite_predictions, _check_is_propensity


class DoubleMLIIVM(LinearScoreMixin, DoubleML):
    """Double machine learning for interactive IV regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(Z,X) = E[Y|X,Z]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[Z|X]`.

    ml_r : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`r_0(Z,X) = E[D|X,Z]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'LATE'`` is the only choice) specifying the score function
        or a callable object / function with signature
        ``psi_a, psi_b = score(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls)``.
        Default is ``'LATE'``.

    subgroups: dict or None
        Dictionary with options to adapt to cases with and without the subgroups of always-takers and never-takes. The
        logical item ``always_takers`` speficies whether there are always takers in the sample. The logical item
        ``never_takers`` speficies whether there are never takers in the sample.
        Default is ``{'always_takers': True, 'never_takers': True}``.

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
    >>> from doubleml.datasets import make_iivm_data
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_r = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> data = make_iivm_data(theta=0.5, n_obs=1000, dim_x=20, alpha_x=1.0, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
    >>> dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_g, ml_m, ml_r)
    >>> dml_iivm_obj.fit().summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.378351  0.190648  1.984551  0.047194  0.004688  0.752015

    Notes
    -----
    **Interactive IV regression (IIVM)** models take the form

    .. math::

        Y = \\ell_0(D, X) + \\zeta, & &\\mathbb{E}(\\zeta | Z, X) = 0,

        Z = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where the treatment variable is binary, :math:`D \\in \\lbrace 0,1 \\rbrace`
    and the instrument is binary, :math:`Z \\in \\lbrace 0,1 \\rbrace`.
    Consider the functions :math:`g_0`, :math:`r_0` and :math:`m_0`, where :math:`g_0` maps the support of :math:`(Z,X)` to
    :math:`\\mathbb{R}` and :math:`r_0` and :math:`m_0` respectively map the support of :math:`(Z,X)` and :math:`X` to
    :math:`(\\varepsilon, 1-\\varepsilon)` for some :math:`\\varepsilon \\in (0, 1/2)`, such that

    .. math::

        Y = g_0(Z, X) + \\nu, & &\\mathbb{E}(\\nu| Z, X) = 0,

        D = r_0(Z, X) + U, & &\\mathbb{E}(U | Z, X) = 0,

        Z = m_0(X) + V, & &\\mathbb{E}(V | X) = 0.

    The target parameter of interest in this model is the local average treatment effect (LATE),

    .. math::

        \\theta_0 = \\frac{\\mathbb{E}[g_0(1, X)] - \\mathbb{E}[g_0(0,X)]}{\\mathbb{E}[r_0(1, X)] - \\mathbb{E}[r_0(0,X)]}.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 ml_r,
                 n_folds=5,
                 n_rep=1,
                 score='LATE',
                 subgroups=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        self._check_data(self._dml_data)
        valid_scores = ['LATE']
        _check_score(self.score, valid_scores, allow_callable=True)

        # set stratication for resampling
        self._strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.z.reshape(-1, 1)
        if draw_sample_splitting:
            self.draw_sample_splitting()

        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        _ = self._check_learner(ml_r, 'ml_r', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m, 'ml_r': ml_r}
        self._normalize_ipw = normalize_ipw
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba', 'ml_r': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba', 'ml_r': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        if subgroups is None:
            # this is the default for subgroups; via None to prevent a mutable default argument
            subgroups = {'always_takers': True, 'never_takers': True}
        else:
            if not isinstance(subgroups, dict):
                raise TypeError('Invalid subgroups ' + str(subgroups) + '. ' +
                                'subgroups must be of type dictionary.')
            if (not all(k in subgroups for k in ['always_takers', 'never_takers']))\
                    | (not all(k in ['always_takers', 'never_takers'] for k in subgroups)):
                raise ValueError('Invalid subgroups ' + str(subgroups) + '. ' +
                                 'subgroups must be a dictionary with keys always_takers and never_takers.')
            if not isinstance(subgroups['always_takers'], bool):
                raise TypeError("subgroups['always_takers'] must be True or False. "
                                f'Got {str(subgroups["always_takers"])}.')
            if not isinstance(subgroups['never_takers'], bool):
                raise TypeError("subgroups['never_takers'] must be True or False. "
                                f'Got {str(subgroups["never_takers"])}.')
        self.subgroups = subgroups
        self._external_predictions_implemented = True

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
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m', 'ml_r0', 'ml_r1']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an IIVM model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        one_instr = (obj_dml_data.n_instr == 1)
        err_msg = ('Incompatible data. '
                   'To fit an IIVM model with DML '
                   'exactly one binary variable with values 0 and 1 '
                   'needs to be specified as instrumental variable.')
        if one_instr:
            binary_instr = (type_of_target(obj_dml_data.z) == 'binary')
            zero_one_instr = np.all((np.power(obj_dml_data.z, 2) - obj_dml_data.z) == 0)
            if not (one_instr & binary_instr & zero_one_instr):
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z),
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = _get_cond_smpls(smpls, z)

        # nuisance g
        if external_predictions['ml_g0'] is not None:
            g_hat0 = {'preds': external_predictions['ml_g0'],
                      'targets': None,
                      'models': None}
        else:
            g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_z0, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat0['targets'] = g_hat0['targets'].astype(float)
            g_hat0['targets'][z == 1] = np.nan

        if self._dml_data.binary_outcome:
            binary_preds = (type_of_target(g_hat0['preds']) == 'binary')
            zero_one_preds = np.all((np.power(g_hat0['preds'], 2) - g_hat0['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                 f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

            _check_is_propensity(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls, eps=1e-12)
        if external_predictions['ml_g1'] is not None:
            g_hat1 = {'preds': external_predictions['ml_g1'],
                      'targets': None,
                      'models': None}
        else:
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_z1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat1['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1['targets'] = g_hat1['targets'].astype(float)
            g_hat1['targets'][z == 0] = np.nan

        if self._dml_data.binary_outcome:
            binary_preds = (type_of_target(g_hat1['preds']) == 'binary')
            zero_one_preds = np.all((np.power(g_hat1['preds'], 2) - g_hat1['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                 f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

            _check_is_propensity(g_hat1['preds'], self._learner['ml_g'], 'ml_g', smpls, eps=1e-12)

        # nuisance m
        if external_predictions['ml_m'] is not None:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)
        # also trimm external predictions
        m_hat['preds'] = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)

        # nuisance r
        r0 = external_predictions['ml_r0'] is not None
        if self.subgroups['always_takers']:
            if r0:
                r_hat0 = {'preds': external_predictions['ml_r0'],
                          'targets': None,
                          'models': None}
            else:
                r_hat0 = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls_z0, n_jobs=n_jobs_cv,
                                         est_params=self._get_params('ml_r0'), method=self._predict_method['ml_r'],
                                         return_models=return_models)
        else:
            r_hat0 = {'preds': np.zeros_like(d), 'targets': np.zeros_like(d), 'models': None}
        if not r0:
            _check_finite_predictions(r_hat0['preds'], self._learner['ml_r'], 'ml_r', smpls)
            # adjust target values to consider only compatible subsamples
            r_hat0['targets'] = r_hat0['targets'].astype(float)
            r_hat0['targets'][z == 1] = np.nan

        r1 = external_predictions['ml_r1'] is not None
        if self.subgroups['never_takers']:
            if r1:
                r_hat1 = {'preds': external_predictions['ml_r1'],
                          'targets': None,
                          'models': None}
            else:
                r_hat1 = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls_z1, n_jobs=n_jobs_cv,
                                         est_params=self._get_params('ml_r1'), method=self._predict_method['ml_r'],
                                         return_models=return_models)
        else:
            r_hat1 = {'preds': np.ones_like(d), 'targets': np.ones_like(d), 'models': None}
        if not r1:
            _check_finite_predictions(r_hat1['preds'], self._learner['ml_r'], 'ml_r', smpls)
            # adjust target values to consider only compatible subsamples
            r_hat1['targets'] = r_hat1['targets'].astype(float)
            r_hat1['targets'][z == 0] = np.nan

        psi_a, psi_b = self._score_elements(y, z, d,
                                            g_hat0['preds'], g_hat1['preds'], m_hat['preds'],
                                            r_hat0['preds'], r_hat1['preds'], smpls)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_g0': g_hat0['preds'],
                                 'ml_g1': g_hat1['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_r0': r_hat0['preds'],
                                 'ml_r1': r_hat1['preds']},
                 'targets': {'ml_g0': g_hat0['targets'],
                             'ml_g1': g_hat1['targets'],
                             'ml_m': m_hat['targets'],
                             'ml_r0': r_hat0['targets'],
                             'ml_r1': r_hat1['targets']},
                 'models': {'ml_g0': g_hat0['models'],
                            'ml_g1': g_hat1['models'],
                            'ml_m': m_hat['models'],
                            'ml_r0': r_hat0['models'],
                            'ml_r1': r_hat1['models']}
                 }

        return psi_elements, preds

    def _score_elements(self, y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls):
        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = y - g_hat1
        w_hat0 = d - r_hat0
        w_hat1 = d - r_hat1

        if self.normalize_ipw:
            m_hat_adj = _normalize_ipw(m_hat, d)
        else:
            m_hat_adj = m_hat

        if isinstance(self.score, str):
            assert self.score == 'LATE'
            psi_b = g_hat1 - g_hat0 \
                + np.divide(np.multiply(z, u_hat1), m_hat_adj) \
                - np.divide(np.multiply(1.0-z, u_hat0), 1.0 - m_hat_adj)
            psi_a = -1*(r_hat1 - r_hat0
                        + np.divide(np.multiply(z, w_hat1), m_hat_adj)
                        - np.divide(np.multiply(1.0-z, w_hat0), 1.0 - m_hat_adj))
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, z=z, d=d,
                                      g_hat0=g_hat0, g_hat1=g_hat1, m_hat=m_hat_adj, r_hat0=r_hat0, r_hat1=r_hat1,
                                      smpls=smpls)

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z),
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = _get_cond_smpls(smpls, z)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None,
                               'ml_r': None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_z0 = [train_index for (train_index, _) in smpls_z0]
        train_inds_z1 = [train_index for (train_index, _) in smpls_z1]

        g0_tune_res = _dml_tune(y, x, train_inds_z0,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g1_tune_res = _dml_tune(y, x, train_inds_z1,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(z, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        if self.subgroups['always_takers']:
            r0_tune_res = _dml_tune(d, x, train_inds_z0,
                                    self._learner['ml_r'], param_grids['ml_r'], scoring_methods['ml_r'],
                                    n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
            r0_best_params = [xx.best_params_ for xx in r0_tune_res]
        else:
            r0_tune_res = None
            r0_best_params = [None] * len(smpls)
        if self.subgroups['never_takers']:
            r1_tune_res = _dml_tune(d, x, train_inds_z1,
                                    self._learner['ml_r'], param_grids['ml_r'], scoring_methods['ml_r'],
                                    n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
            r1_best_params = [xx.best_params_ for xx in r1_tune_res]
        else:
            r1_tune_res = None
            r1_best_params = [None] * len(smpls)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g0': g0_best_params,
                  'ml_g1': g1_best_params,
                  'ml_m': m_best_params,
                  'ml_r0': r0_best_params,
                  'ml_r1': r1_best_params}

        tune_res = {'g0_tune': g0_tune_res,
                    'g1_tune': g1_tune_res,
                    'm_tune': m_tune_res,
                    'r0_tune': r0_tune_res,
                    'r1_tune': r1_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        pass
