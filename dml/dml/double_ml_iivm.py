import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

from .double_ml import DoubleML
from .helper import check_binary_vector


class DoubleMLIIVM(DoubleML):
    """
    Double Machine Learning for Interactive IV Model
    """

    def _check_inf_method(self, inf_model):
        valid_inf_model = ['LATE']
        if inf_model not in valid_inf_model:
            raise ValueError('invalid inf_model ' + inf_model +
                             '\n valid inf_model ' + valid_inf_model)
        return inf_model

    def _check_data(self, obj_dml_data):
        assert obj_dml_data.n_treat == 1
        check_binary_vector(obj_dml_data.d, variable_name='d')
        check_binary_vector(obj_dml_data.z, variable_name='z')
        return
    
    def _get_cond_smpls(self, smpls, z):
        smpls_z0 = [(np.intersect1d(np.where(z == 0)[0], train),
                     test) for train, test in smpls]
        smpls_z1 = [(np.intersect1d(np.where(z == 1)[0], train),
                     test) for train, test in smpls]
        return smpls_z0, smpls_z1
    
    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls):
        
        ml_m = self.ml_learners['ml_m']
        ml_g0 = clone(self.ml_learners['ml_g'])
        ml_g1 = clone(self.ml_learners['ml_g'])
        ml_r0 = clone(self.ml_learners['ml_r'])
        ml_r1 = clone(self.ml_learners['ml_r'])
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
        X, d = check_X_y(X, obj_dml_data.d)

        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = self._get_cond_smpls(smpls, z)
        
        # nuisance g
        g_hat0 = cross_val_predict(ml_g0, X, y, cv=smpls_z0)
        g_hat1 = cross_val_predict(ml_g1, X, y, cv=smpls_z1)
        
        # nuisance m
        m_hat = cross_val_predict(ml_m, X, z, cv=smpls, method='predict_proba')[:, 1]
        
        # nuisance r
        r_hat0 = cross_val_predict(ml_r0, X, d, cv=smpls_z0, method='predict_proba')[:, 1]
        r_hat1 = cross_val_predict(ml_r1, X, d, cv=smpls_z1, method='predict_proba')[:, 1]
        
        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = y - g_hat1
        w_hat0 = d - r_hat0
        w_hat1 = d - r_hat1

        inf_model = self.inf_model
        if inf_model == 'LATE':
            score_b = g_hat1 - g_hat0 \
                            + np.divide(np.multiply(z, u_hat1), m_hat) \
                            - np.divide(np.multiply(1.0-z, u_hat0), 1.0 - m_hat)
            score_a = -1*(r_hat1 - r_hat0 \
                                + np.divide(np.multiply(z, w_hat1), m_hat) \
                                - np.divide(np.multiply(1.0-z, w_hat0), 1.0 - m_hat))
        else:
            raise ValueError('invalid inf_model')

        return score_a, score_b

