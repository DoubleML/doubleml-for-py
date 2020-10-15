import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML
from ._helper import _dml_cv_predict, _check_and_duplicate_params


class DoubleMLPLR(DoubleML):
    """
    Double machine learning for partially linear regression models

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
    >>> from doubleml.datasets import make_plr_CCDDHNR2018
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    >>> ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner)}
    >>> obj_dml_data = make_plr_CCDDHNR2018()
    >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_learners)
    >>> dml_plr_obj.fit()
    >>> dml_plr_obj.summary
         coef   std err         t         P>|t|    2.5 %    97.5 %
    d  0.608353  0.101595  5.988004  2.124312e-09  0.40923  0.807477

    Notes
    -----
    .. include:: ../../shared/models/plr.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        self.ml_g = ml_g
        self.ml_m = ml_m
        self._initialize_ml_nuisance_params()

    @property
    def learner(self):
        learner = {'ml_g': self.ml_g,
                   'ml_m': self.ml_m}
        return learner

    @property
    def params(self):
        params = {'ml_g': self._g_params,
                  'ml_m': self._m_params}
        return params
    
    @property
    def g_params(self):
        return self._g_params

    @property
    def m_params(self):
        return self._m_params

    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.

    @property
    def __g_params(self):
        return self._g_params[self.d_cols[self._i_treat]][self._i_rep]

    @property
    def __m_params(self):
        return self._m_params[self.d_cols[self._i_treat]][self._i_rep]

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['IV-type', 'partialling out']
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
        return

    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = _dml_cv_predict(self.ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__g_params)
        
        # nuisance m
        m_hat = _dml_cv_predict(self.ml_m, X, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__m_params)

        # compute residuals
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.multiply(v_hat, d)

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            if score == 'IV-type':
                psi_a = -v_hatd
            elif score == 'partialling out':
                psi_a = -np.multiply(v_hat, v_hat)
            psi_b = np.multiply(v_hat, u_hat)
        elif callable(self.score):
            psi_a, psi_b = self.score(y, d, g_hat, m_hat, smpls)
        
        return psi_a, psi_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        g_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g
            g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            g_grid_search = GridSearchCV(self.ml_g, param_grids['ml_g'],
                                         scoring=scoring_methods['ml_g'],
                                         cv=g_tune_resampling)
            g_tune_res[idx] = g_grid_search.fit(X[train_index, :], y[train_index])

        m_tune_res = [None] * len(smpls)
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            m_grid_search = GridSearchCV(self.ml_m, param_grids['ml_m'],
                                         scoring=scoring_methods['ml_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], d[train_index])

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return(res)

    def _initialize_ml_nuisance_params(self):
        self._g_params = {key: [None] * self.n_rep for key in self.d_cols}
        self._m_params = {key: [None] * self.n_rep for key in self.d_cols}

    def set_ml_nuisance_params(self, learner, treat_var, params):
        valid_learner = ['ml_g', 'ml_m']
        if learner not in valid_learner:
            raise ValueError('invalid nuisance learner' + learner +
                             '\n valid nuisance learner ' + ' or '.join(valid_learner))
        if treat_var not in self.d_cols:
            raise ValueError('invalid treatment variable' + learner +
                             '\n valid treatment variable ' + ' or '.join(self.d_cols))

        all_params = _check_and_duplicate_params(params, self.n_rep, self.n_folds, self.apply_cross_fitting)

        if learner == 'ml_g':
            self._g_params[treat_var] = all_params
        elif learner == 'ml_m':
            self._m_params[treat_var] = all_params
