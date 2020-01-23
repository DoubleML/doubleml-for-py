import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict

from .double_ml import DoubleML


class DoubleMLPLIV(DoubleML):
    """
    Double Machine Learning for Partially Linear IV regression model
    """

    def _check_inf_method(self, inf_model):
        valid_inf_model = ['DML2018']
        if inf_model not in valid_inf_model:
            raise ValueError('invalid inf_model ' + inf_model +
                             '\n valid inf_model ' + valid_inf_model)
        return inf_model

    def _check_data(self, obj_dml_data):
        return
    
    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        ml_r = self.ml_learners['ml_r']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        m_hat = cross_val_predict(ml_m, X, z, cv = smpls)
        
        # nuisance r
        r_hat = cross_val_predict(ml_r, X, d, cv = smpls)
        
        # compute residuals
        u_hat = y - g_hat
        v_hat = z - m_hat
        w_hat = d - r_hat

        inf_model = self.inf_model
        if inf_model == 'DML2018':
            score_a = -np.multiply(w_hat, v_hat)
        else:
            # check whether its worth implementing the IV_type here as well
            # In CCDHNR equation (4.7) a score of this type is provided;
            # however in the following paragraph it is explained that one might
            # still need to estimate the DML2018 type first
            raise ValueError('invalid inf_model')
        score_b = np.multiply(v_hat, u_hat)

        return score_a, score_b

