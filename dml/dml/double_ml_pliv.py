import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict

from .double_ml import DoubleML


class DoubleMLPLIV(DoubleML):
    """
    Double Machine Learning for Partially Linear IV regression model
    """
    
    def _check_data(self, obj_dml_data):
        return
    
    def _ml_nuisance(self, obj_dml_data):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        ml_r = self.ml_learners['ml_r']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)
        
        smpls = self._smpls
        
        # nuisance g
        self.g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, z, cv = smpls)
        
        # nuisance r
        self.r_hat = cross_val_predict(ml_r, X, d, cv = smpls)
        
        # compute residuals
        self._u_hat = y - self.g_hat
        self._v_hat = z - self.m_hat
        self._w_hat = d - self.r_hat
        self._v_hatd = np.multiply(self._v_hat, d)
        
        # compute score elements
        self._compute_score_elements()
    
    def _compute_score_elements(self):
        inf_model = self.inf_model
        
        u_hat = self._u_hat
        v_hat = self._v_hat
        w_hat = self._w_hat
        
        if inf_model == 'DML2018':
            self.score_a = -np.multiply(w_hat, v_hat)
        else:
            # check whether its worth implementing the IV_type here as well
            # In CCDHNR equation (4.7) a score of this type is provided;
            # however in the following paragraph it is explained that one might
            # still need to estimate the DML2018 type first
            raise ValueError('invalid inf_model')
        self.score_b = np.multiply(v_hat, u_hat)

