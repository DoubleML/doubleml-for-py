import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict

from .double_ml import DoubleML
from .helper import check_binary_vector


class DoubleMLIIVM(DoubleML):
    """
    Double Machine Learning for Interactive IV Model
    """
    
    def _check_data(self, obj_dml_data):
        assert obj_dml_data.n_treat == 1
        check_binary_vector(obj_dml_data.d, variable_name='d')
        check_binary_vector(obj_dml_data.z, variable_name='z')
        return
    
    def _get_cond_smpls(self, z):
        smpls = self._smpls
        self._smpls_z0 = [(np.intersect1d(np.where(z == 0)[0], train),
                           test) for train, test in smpls]
        self._smpls_z1 = [(np.intersect1d(np.where(z == 1)[0], train),
                           test) for train, test in smpls]
    
    def _ml_nuisance(self, obj_dml_data):
        
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        ml_r = self.ml_learners['ml_r']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # get train indices for z == 0 and z == 1
        self._get_cond_smpls(z)
        
        smpls = self._smpls
        smpls_z0 = self._smpls_z0
        smpls_z1 = self._smpls_z1
        
        # nuisance g
        self.g_hat0 = cross_val_predict(ml_g, X, y, cv = smpls_z0)
        self.g_hat1 = cross_val_predict(ml_g, X, y, cv = smpls_z1)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, z, cv = smpls, method='predict_proba')[:, 1]
        
        # nuisance r
        self.r_hat0 = cross_val_predict(ml_r, X, d, cv = smpls_z0, method='predict_proba')[:, 1]
        self.r_hat1 = cross_val_predict(ml_r, X, d, cv = smpls_z1, method='predict_proba')[:, 1]
        
        # compute residuals
        self._u_hat0 = y - self.g_hat0
        self._u_hat1 = y - self.g_hat1
        self._v_hat = z - self.m_hat
        self._w_hat0 = d - self.r_hat0
        self._w_hat1 = d - self.r_hat1

        # add z to class for score computations
        self._z = z

    def _compute_score_elements(self):
        inf_model = self.inf_model
        if inf_model == 'LATE':
            self.score_b = self.g_hat1 - self.g_hat0 \
                            + np.divide(np.multiply(self._z, self._u_hat1), self.m_hat) \
                            - np.divide(np.multiply(1.0-self._z, self._u_hat1), 1.0 - self.m_hat)
            self.score_a = -1*(self.r_hat1 - self.r_hat0 \
                                + np.divide(np.multiply(self._z, self._w_hat1), self.m_hat) \
                                - np.divide(np.multiply(1.0-self._z, self._w_hat0), 1.0 - self.m_hat))
        else:
            raise ValueError('invalid inf_model')

