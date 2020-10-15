import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML, DoubleMLData
from ._helper import check_binary_vector
from ._helper import _dml_cv_predict


class DoubleMLIIVM(DoubleML):
    """
    Double machine learning for interactive IV regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`g_0(Z,X) = E[Y|X,Z]`.

    ml_m : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestClassifier`)
        for the nuisance function :math:`m_0(X) = E[Z|X]`.

    ml_r : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestClassifier`)
        for the nuisance function :math:`r_0(Z,X) = E[D|X,Z]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'LATE'`` is the only choice) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls)``.
        Default is ``'LATE'``.

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
    >>> ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
    >>> ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
    >>> ml_r = RandomForestClassifier(max_depth=2, n_estimators=10)
    >>> data = make_iivm_data(return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
    >>> dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_g, ml_m, ml_r)
    >>> dml_iivm_obj.fit()
    >>> dml_iivm_obj.summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.933779  1.049043  0.890125  0.373399 -1.122308  2.989866

    Notes
    -----
    .. include:: ../../shared/models/iivm.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 ml_r,
                 n_folds=5,
                 n_rep=1,
                 score='LATE',
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
        self._learner = {'ml_g': ml_g,
                         'ml_m': ml_m,
                         'ml_r': ml_r}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('invalid trimming_rule ' + trimming_rule +
                             '\n valid trimming_rule ' + ' or '.join(valid_trimming_rule))
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m', 'ml_r0', 'ml_r1']
        self._params = {learner: {key: [None] * self.n_rep for key in self.d_cols} for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['LATE']
            if score not in valid_score:
                raise ValueError('invalid score ' + score +
                                 '\n valid score ' + valid_score)
        else:
            if not callable(score):
                raise ValueError('score should be either a string or a callable.'
                                 ' %r was passed' % score)
        return score

    def _check_data(self, obj_dml_data):
        assert obj_dml_data.n_treat == 1
        check_binary_vector(obj_dml_data.d, variable_name='d')
        check_binary_vector(obj_dml_data.z, variable_name='z')
        return
    
    def _get_cond_smpls(self, smpls, z):
        smpls_z0 = [(np.intersect1d(np.where(z == 0)[0], train),
                     test) for train, test in smpls]
        smpls_z1 = [(np.intersect1d(np.where(z == 1)[0], train),
                     test) for train, test in smpls]
        return smpls_z0, smpls_z1
    
    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = self._get_cond_smpls(smpls, z)
        
        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], X, y, smpls=smpls_z0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'))
        g_hat1 = _dml_cv_predict(self._learner['ml_g'], X, y, smpls=smpls_z1, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g1'))
        
        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], X, z, smpls=smpls, method='predict_proba', n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'))[:, 1]
        
        # nuisance r
        r_hat0 = _dml_cv_predict(self._learner['ml_r'], X, d, smpls=smpls_z0, method='predict_proba', n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_r0'))[:, 1]
        r_hat1 = _dml_cv_predict(self._learner['ml_r'], X, d, smpls=smpls_z1, method='predict_proba', n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_r1'))[:, 1]

        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = y - g_hat1
        w_hat0 = d - r_hat0
        w_hat1 = d - r_hat1

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            if score == 'LATE':
                psi_b = g_hat1 - g_hat0 \
                                + np.divide(np.multiply(z, u_hat1), m_hat) \
                                - np.divide(np.multiply(1.0-z, u_hat0), 1.0 - m_hat)
                psi_a = -1*(r_hat1 - r_hat0 \
                                    + np.divide(np.multiply(z, w_hat1), m_hat) \
                                    - np.divide(np.multiply(1.0-z, w_hat0), 1.0 - m_hat))
        elif callable(self.score):
            psi_a, psi_b = self.score(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = self._get_cond_smpls(smpls, z)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None,
                               'ml_r': None}

        g0_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            g0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            g0_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                          scoring=scoring_methods['ml_g'],
                                          cv=g0_tune_resampling)
            train_index_z0 = smpls_z0[idx][0]
            g0_tune_res[idx] = g0_grid_search.fit(X[train_index_z0, :], y[train_index_z0])

        g1_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            g1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            g1_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                          scoring=scoring_methods['ml_g'],
                                          cv=g1_tune_resampling)
            train_index_z1 = smpls_z1[idx][0]
            g1_tune_res[idx] = g1_grid_search.fit(X[train_index_z1, :], y[train_index_z1])

        m_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            m_grid_search = GridSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                         scoring=scoring_methods['ml_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], z[train_index])

        r0_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            r0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            r0_grid_search = GridSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                          scoring=scoring_methods['ml_r'],
                                          cv=r0_tune_resampling)
            train_index_z0 = smpls_z0[idx][0]
            r0_tune_res[idx] = r0_grid_search.fit(X[train_index_z0, :], d[train_index_z0])

        r1_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            r1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            r1_grid_search = GridSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                          scoring=scoring_methods['ml_r'],
                                          cv=r1_tune_resampling)
            train_index_z1 = smpls_z1[idx][0]
            r1_tune_res[idx] = r1_grid_search.fit(X[train_index_z1, :], d[train_index_z1])

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        r0_best_params = [xx.best_params_ for xx in r0_tune_res]
        r1_best_params = [xx.best_params_ for xx in r1_tune_res]

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
