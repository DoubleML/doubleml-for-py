import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML, DoubleMLData
from ._helper import _dml_cv_predict
from ._helper import check_binary_vector


class DoubleMLIRM(DoubleML):
    """
    Double machine learning for interactive regression models

    Parameters
    ----------
    obj_dml_data :
        ToDo
    ml_learners :
        ToDo
    n_folds :
        ToDo
    n_rep :
        ToDo
    score :
        ToDo
    dml_procedure :
        ToDo
    draw_sample_splitting :
        ToDo
    apply_cross_fitting :
        ToDo

    Examples
    --------
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> ml_learners = {'ml_m': RandomForestClassifier(max_depth=2, n_estimators=10),
    >>>                'ml_g': RandomForestRegressor(max_depth=2, n_estimators=10)}
    >>> data = make_irm_data()
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_learners)
    >>> dml_irm_obj.fit()
    >>> dml_irm_obj.summary

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
        self._learner = {'ml_g': ml_g,
                         'ml_m': ml_m}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('invalid trimming_rule ' + trimming_rule +
                             '\n valid trimming_rule ' + ' or '.join(valid_trimming_rule))
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self.d_cols} for learner in valid_learner}

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
        assert obj_dml_data.z_cols is None
        assert obj_dml_data.n_treat == 1
        check_binary_vector(obj_dml_data.d, variable_name='d')
        return
    
    def _get_cond_smpls(self, smpls, d):
        smpls_d0 = [(np.intersect1d(np.where(d == 0)[0], train),
                      test) for train, test in smpls]
        smpls_d1 = [(np.intersect1d(np.where(d == 1)[0], train),
                      test) for train, test in smpls]
        return smpls_d0, smpls_d1
    
    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        score = self.score
        self._check_score(score)
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = self._get_cond_smpls(smpls, d)
        
        # fraction of treated for ATTE
        if score == 'ATTE':
            p_hat = np.zeros_like(d, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], X, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'))
        if (score == 'ATE') | callable(self.score):
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], X, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'))
        
        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], X, d, smpls=smpls, method='predict_proba', n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'))[:, 1]

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        # compute residuals
        u_hat0 = y - g_hat0
        if score == 'ATE':
            u_hat1 = y - g_hat1
        
        if isinstance(self.score, str):
            if score == 'ATE':
                psi_b = g_hat1 - g_hat0 \
                                + np.divide(np.multiply(d, u_hat1), m_hat) \
                                - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
                psi_a = np.full_like(m_hat, -1.0)
            elif score == 'ATTE':
                psi_b = np.divide(np.multiply(d, u_hat0), p_hat) \
                                - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                                            np.multiply(p_hat, (1.0 - m_hat)))
                psi_a = - np.divide(d, p_hat)
        elif callable(self.score):
            psi_a, psi_b = self.score(y, d,
                                              g_hat0, g_hat1, m_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        score = self.score

        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = self._get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        g0_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            g0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            g0_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                          scoring=scoring_methods['ml_g'],
                                          cv=g0_tune_resampling)
            train_index_d0 = smpls_d0[idx][0]
            g0_tune_res[idx] = g0_grid_search.fit(X[train_index_d0, :], y[train_index_d0])

        if score == 'ATE':
            g1_tune_res = [None] * len(smpls)
            for idx, (train_index, test_index) in enumerate(smpls):
                g1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
                g1_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                              scoring=scoring_methods['ml_g'],
                                              cv=g1_tune_resampling)
                train_index_d1 = smpls_d1[idx][0]
                g1_tune_res[idx] = g1_grid_search.fit(X[train_index_d1, :], y[train_index_d1])

        m_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            m_grid_search = GridSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                         scoring=scoring_methods['ml_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], d[train_index])

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        if score == 'ATTE':
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
