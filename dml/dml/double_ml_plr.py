import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML
from .helper import _dml_cross_val_predict


class DoubleMLPLR(DoubleML):
    """
    Double Machine Learning for Partially Linear Regression
    """
    def __init__(self,
                 n_folds,
                 ml_learners,
                 dml_procedure,
                 inf_model,
                 se_reestimate=False,
                 n_rep_cross_fit=1):
        super().__init__(n_folds,
                         ml_learners,
                         dml_procedure,
                         inf_model,
                         se_reestimate,
                         n_rep_cross_fit)
        self._g_params = None
        self._m_params = None

    def _check_inf_method(self, inf_model):
        valid_inf_model = ['IV-type', 'DML2018']
        if inf_model not in valid_inf_model:
            raise ValueError('invalid inf_model ' + inf_model +
                             '\n valid inf_model ' + ' or '.join(valid_inf_model))
        return inf_model

    def _check_data(self, obj_dml_data):
        assert obj_dml_data.z_col is None
        return

    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        ml_g = self.ml_learners['ml_g']
        ml_m = self.ml_learners['ml_m']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = _dml_cross_val_predict(ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv,
                                       est_params=self._g_params)
        
        # nuisance m
        m_hat = _dml_cross_val_predict(ml_m, X, d, smpls=smpls, n_jobs=n_jobs_cv,
                                       est_params=self._m_params)
        
        # compute residuals
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.multiply(v_hat, d)

        inf_model = self.inf_model
        if inf_model == 'IV-type':
            score_a = -v_hatd
        elif inf_model == 'DML2018':
            score_a = -np.multiply(v_hat, v_hat)
        else:
            raise ValueError('invalid inf_model')
        score_b = np.multiply(v_hat, u_hat)

        return score_a, score_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        ml_g = self.ml_learners['ml_g']
        ml_m = self.ml_learners['ml_m']

        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_g': None,
                               'scoring_methods_m': None}

        g_tune_res = [None] * len(smpls)
        m_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g
            g_tune_resampling = KFold(n_splits=n_folds_tune)
            g_grid_search = GridSearchCV(ml_g, param_grids['param_grid_g'],
                                         scoring=scoring_methods['scoring_methods_g'],
                                         cv=g_tune_resampling)
            g_tune_res[idx] = g_grid_search.fit(X[train_index, :], y[train_index])

            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(ml_m, param_grids['param_grid_m'],
                                         scoring=scoring_methods['scoring_methods_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], d[train_index])

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'g_params': g_best_params,
                  'm_params': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return(res)

    def __set_ml_nuisance_params(self, params):
        self._g_params = params['g_params']
        self._m_params = params['m_params']
