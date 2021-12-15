import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from .double_ml import DoubleML
from ._utils import _dml_cv_predict, _get_cond_smpls, _dml_tune, _check_finite_predictions


class DoubleMLIIVM(DoubleML):
    """Double machine learning for interactive IV regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(Z,X) = E[Y|X,Z]`.

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

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

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
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
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
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        _ = self._check_learner(ml_r, 'ml_r', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m, 'ml_r': ml_r}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba', 'ml_r': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')

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
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m', 'ml_r0', 'ml_r1']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['LATE']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. '
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not(one_treat & binary_treat & zero_one_treat):
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
            if not(one_instr & binary_instr & zero_one_instr):
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        return

    def _nuisance_est(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z),
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = _get_cond_smpls(smpls, z)

        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_z0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat0, self._learner['ml_g'], 'ml_g', smpls)
        g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_z1, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat1, self._learner['ml_g'], 'ml_g', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        # nuisance r
        if self.subgroups['always_takers']:
            r_hat0 = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls_z0, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_r0'), method=self._predict_method['ml_r'])
        else:
            r_hat0 = np.zeros_like(d)
        _check_finite_predictions(r_hat0, self._learner['ml_r'], 'ml_r', smpls)

        if self.subgroups['never_takers']:
            r_hat1 = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls_z1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_r1'), method=self._predict_method['ml_r'])
        else:
            r_hat1 = np.ones_like(d)
        _check_finite_predictions(r_hat1, self._learner['ml_r'], 'ml_r', smpls)

        psi_a, psi_b = self._score_elements(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls)
        preds = {'ml_g0': g_hat0,
                 'ml_g1': g_hat1,
                 'ml_m': m_hat,
                 'ml_r0': r_hat0,
                 'ml_r1': r_hat1}

        return psi_a, psi_b, preds

    def _score_elements(self, y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls):
        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = y - g_hat1
        w_hat0 = d - r_hat0
        w_hat1 = d - r_hat1

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        if isinstance(self.score, str):
            assert self.score == 'LATE'
            psi_b = g_hat1 - g_hat0 \
                + np.divide(np.multiply(z, u_hat1), m_hat) \
                - np.divide(np.multiply(1.0-z, u_hat0), 1.0 - m_hat)
            psi_a = -1*(r_hat1 - r_hat0
                        + np.divide(np.multiply(z, w_hat1), m_hat)
                        - np.divide(np.multiply(1.0-z, w_hat0), 1.0 - m_hat))
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls)

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
