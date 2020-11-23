import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .double_ml import DoubleML
from ._helper import _dml_cv_predict, _get_cond_smpls


class DoubleMLIRM(DoubleML):
    """
    Double machine learning for interactive regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`g_0(D,X) = E[Y|X,D]`.

    ml_m : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestClassifier`)
        for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'ATE'`` or ``'ATTE'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat0, g_hat1, m_hat, smpls)``.
        Default is ``'ATE'``.

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
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
    >>> dml_irm_obj.fit().summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.414073  0.238529  1.735941  0.082574 -0.053436  0.881581

    Notes
    -----
    .. include:: ../../shared/models/irm.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
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
        self._learner = {'ml_g': self._check_learner(ml_g, 'ml_g'),
                         'ml_m': self._check_learner(ml_m, 'ml_m', classifier=True)}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('invalid trimming_rule ' + trimming_rule +
                             '\n valid trimming_rule ' + ' or '.join(valid_trimming_rule))
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['ATE', 'ATTE']
            if score not in valid_score:
                raise ValueError('invalid score ' + score +
                                 '\n valid score ' + ' or '.join(valid_score))
        else:
            if not callable(score):
                raise ValueError('score should be either a string or a callable.'
                                 ' %r was passed' % score)
        return score

    def _check_data(self, obj_dml_data):
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data.\n'
                             ' and '.join(obj_dml_data.z_cols) +
                             'have been set as instrumental variable(s).\n'
                             'To fit an interactiv IV regression model use DoubleMLIIVM instead of DoubleMLIRM.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.any((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not(one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data.\n'
                             'To fit an IIVM model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return
    
    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        
        # fraction of treated for ATTE
        p_hat = None
        if self.score == 'ATTE':
            p_hat = np.zeros_like(d, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'))
        g_hat1 = None
        if (self.score == 'ATE') | callable(self.score):
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'))
        
        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, method='predict_proba', n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'))[:, 1]

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = None
        if self.score == 'ATE':
            u_hat1 = y - g_hat1
        
        if isinstance(self.score, str):
            if self.score == 'ATE':
                psi_b = g_hat1 - g_hat0 \
                    + np.divide(np.multiply(d, u_hat1), m_hat) \
                    - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
                psi_a = np.full_like(m_hat, -1.0)
            else:
                assert self.score == 'ATTE'
                psi_b = np.divide(np.multiply(d, u_hat0), p_hat) \
                    - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                                np.multiply(p_hat, (1.0 - m_hat)))
                psi_a = - np.divide(d, p_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y, d, g_hat0, g_hat1, m_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                            search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        g0_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            g0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                g0_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                              scoring=scoring_methods['ml_g'],
                                              cv=g0_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                g0_grid_search = RandomizedSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                                    scoring=scoring_methods['ml_g'],
                                                    cv=g0_tune_resampling, n_jobs=n_jobs_cv,
                                                    n_iter=n_iter_randomized_search)
            train_index_d0 = smpls_d0[idx][0]
            g0_tune_res.append(g0_grid_search.fit(x[train_index_d0, :], y[train_index_d0]))

        g1_tune_res = list()
        if self.score == 'ATE':
            for idx, (train_index, test_index) in enumerate(smpls):
                g1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
                if search_mode == 'grid_search':
                    g1_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                                  scoring=scoring_methods['ml_g'],
                                                  cv=g1_tune_resampling, n_jobs=n_jobs_cv)
                else:
                    assert search_mode == 'randomized_search'
                    g1_grid_search = RandomizedSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                                        scoring=scoring_methods['ml_g'],
                                                        cv=g1_tune_resampling, n_jobs=n_jobs_cv,
                                                        n_iter=n_iter_randomized_search)
                train_index_d1 = smpls_d1[idx][0]
                g1_tune_res.append(g1_grid_search.fit(x[train_index_d1, :], y[train_index_d1]))

        m_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                m_grid_search = GridSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                             scoring=scoring_methods['ml_m'],
                                             cv=m_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                m_grid_search = RandomizedSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                                   scoring=scoring_methods['ml_m'],
                                                   cv=m_tune_resampling, n_jobs=n_jobs_cv,
                                                   n_iter=n_iter_randomized_search)
            m_tune_res.append(m_grid_search.fit(x[train_index, :], d[train_index]))

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        if self.score == 'ATTE':
            params = {'ml_g0': g0_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'm_tune': m_tune_res}
        else:
            g1_best_params = [xx.best_params_ for xx in g1_tune_res]
            params = {'ml_g0': g0_best_params,
                      'ml_g1': g1_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'g1_tune': g1_tune_res,
                        'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
