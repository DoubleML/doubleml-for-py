import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict

from .double_ml import DoubleML


class DoubleMLPLR(DoubleML):
    """
    Double Machine Learning for Partially Linear Regression
    """

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
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = cross_val_predict(ml_g, X, y, cv = smpls, n_jobs=n_jobs_cv)
        
        # nuisance m
        m_hat = cross_val_predict(ml_m, X, d, cv = smpls, n_jobs=n_jobs_cv)
        
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

