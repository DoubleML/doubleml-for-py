import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict

from .double_ml import DoubleML

class DoubleMLPLR(DoubleML):
    """
    Double Machine Learning for Partially Linear Regression
    """
    
    def _check_data(self, obj_dml_data):
        assert obj_dml_data.z_col is None
        return
    
    def _ml_nuisance(self, obj_dml_data):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(obj_dml_data.X, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        smpls = self._smpls
        
        # nuisance g
        self.g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, d, cv = smpls)
        
        # compute residuals
        self._u_hat = y - self.g_hat
        self._v_hat = d - self.m_hat
        self._v_hatd = np.multiply(self._v_hat, d)
        
        # compute score elements
        self._compute_score_elements()
    
    def _compute_score_elements(self):
        inf_model = self.inf_model
        
        u_hat = self._u_hat
        v_hat = self._v_hat
        v_hatd = self._v_hatd
        
        if inf_model == 'IV-type':
            self.score_a = -v_hatd
        elif inf_model == 'DML2018':
            self.score_a = -np.multiply(v_hat,v_hat)
        else:
            raise ValueError('invalid inf_model')
        self.score_b = np.multiply(v_hat,u_hat)
    
