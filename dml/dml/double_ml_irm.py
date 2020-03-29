import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML, DoubleMLData
from .helper import _dml_cross_val_predict
from .helper import check_binary_vector


class DoubleMLIRM(DoubleML):
    """
    Double Machine Learning for Interactive Regression Model
    """
    def __init__(self,
                 data,
                 x_cols,
                 y_col,
                 d_cols,
                 n_folds,
                 ml_learners,
                 dml_procedure,
                 inf_model,
                 se_reestimate=False,
                 n_rep_cross_fit=1):
        obj_dml_data = DoubleMLData(data, x_cols, y_col, d_cols)
        super().__init__(obj_dml_data,
                         n_folds,
                         ml_learners,
                         dml_procedure,
                         inf_model,
                         se_reestimate,
                         n_rep_cross_fit)
        self._g0_params = None
        self._g1_params = None
        self._m_params = None

    def _check_inf_method(self, inf_model):
        valid_inf_model = ['ATE', 'ATTE']
        if inf_model not in valid_inf_model:
            raise ValueError('invalid inf_model ' + inf_model +
                             '\n valid inf_model ' + ' or '.join(valid_inf_model))
        return inf_model

    def _check_data(self, obj_dml_data):
        assert obj_dml_data.z_col is None
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
        inf_model = self.inf_model

        ml_g0 = clone(self.ml_learners['ml_g'])
        ml_g1 = clone(self.ml_learners['ml_g'])
        ml_m = self.ml_learners['ml_m']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = self._get_cond_smpls(smpls, d)
        
        # fraction of treated for ATTE
        if inf_model == 'ATTE':
            p_hat = np.zeros_like(d, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        # nuisance g
        g_hat0 = _dml_cross_val_predict(ml_g0, X, y, smpls=smpls_d0, n_jobs=n_jobs_cv)
        if inf_model == 'ATE':
            g_hat1 = _dml_cross_val_predict(ml_g1, X, y, smpls=smpls_d1, n_jobs=n_jobs_cv)
        
        # nuisance m
        m_hat = _dml_cross_val_predict(ml_m, X, d, smpls=smpls, method='predict_proba', n_jobs=n_jobs_cv)[:, 1]
        
        # compute residuals
        u_hat0 = y - g_hat0
        if inf_model == 'ATE':
            u_hat1 = y - g_hat1

        if inf_model == 'ATE':
            score_b = g_hat1 - g_hat0 \
                            + np.divide(np.multiply(d, u_hat1), m_hat) \
                            - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
            score_a = np.full_like(m_hat, -1.0)
        elif inf_model == 'ATTE':
            score_b = np.divide(np.multiply(d, u_hat0), p_hat) \
                            - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                                        np.multiply(p_hat, (1.0 - m_hat)))
            score_a = - np.divide(d, p_hat)
        else:
            raise ValueError('invalid inf_model')

        return score_a, score_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        inf_model = self.inf_model

        ml_g0 = clone(self.ml_learners['ml_g'])
        ml_g1 = clone(self.ml_learners['ml_g'])
        ml_m = self.ml_learners['ml_m']

        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = self._get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_g0': None,
                               'scoring_methods_g1': None,
                               'scoring_methods_m': None}

        g0_tune_res = [None] * len(smpls)
        g1_tune_res = [None] * len(smpls)
        m_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g0
            g0_tune_resampling = KFold(n_splits=n_folds_tune)
            g0_grid_search = GridSearchCV(ml_g0, param_grids['param_grid_g0'],
                                         scoring=scoring_methods['scoring_methods_g0'],
                                         cv=g0_tune_resampling)

            train_index_d0 = smpls_d0[idx][0]
            g0_tune_res[idx] = g0_grid_search.fit(X[train_index_d0, :], y[train_index_d0])

            if inf_model == 'ATE':
                # cv for ml_g1
                g1_tune_resampling = KFold(n_splits=n_folds_tune)
                g1_grid_search = GridSearchCV(ml_g1, param_grids['param_grid_g1'],
                                              scoring=scoring_methods['scoring_methods_g1'],
                                              cv=g1_tune_resampling)

                train_index_d1 = smpls_d1[idx][0]
                g1_tune_res[idx] = g1_grid_search.fit(X[train_index_d1, :], y[train_index_d1])
            else:
                g1_tune_res[idx] = {'best_params_': None}

            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(ml_m, param_grids['param_grid_m'],
                                         scoring=scoring_methods['scoring_methods_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], d[train_index])

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'g0_params': g0_best_params,
                  'g1_params': g1_best_params,
                  'm_params': m_best_params}

        tune_res = {'g0_tune': g0_tune_res,
                    'g1_tune': g1_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return(res)

    def _set_ml_nuisance_params(self, params):
        self._g0_params = params['g0_params']
        self._g1_params = params['g1_params']
        self._m_params = params['m_params']

