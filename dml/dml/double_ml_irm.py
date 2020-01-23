import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

from .double_ml import DoubleML
from .helper import check_binary_vector


class DoubleMLIRM(DoubleML):
    """
    Double Machine Learning for Interactive Regression Model
    """

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
    
    def _get_cond_smpls(self, d):
        smpls = self._smpls
        smpls_d0 = [(np.intersect1d(np.where(d == 0)[0], train),
                      test) for train, test in smpls]
        smpls_d1 = [(np.intersect1d(np.where(d == 1)[0], train),
                      test) for train, test in smpls]
        return smpls_d0, smpls_d1
    
    def _ml_nuisance_and_score_elements(self, obj_dml_data):
        inf_model = self.inf_model
        
        ml_m = self.ml_learners['ml_m']
        ml_g0 = clone(self.ml_learners['ml_g'])
        ml_g1 = clone(self.ml_learners['ml_g'])
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)

        smpls = self._smpls
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = self._get_cond_smpls(d)
        
        # fraction of treated for ATTE
        if inf_model == 'ATTE':
            p_hat = np.zeros_like(d, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        # nuisance g
        g_hat0 = cross_val_predict(ml_g0, X, y, cv = smpls_d0)
        if inf_model == 'ATE':
            g_hat1 = cross_val_predict(ml_g1, X, y, cv = smpls_d1)
        
        # nuisance m
        m_hat = cross_val_predict(ml_m, X, d, cv = smpls, method='predict_proba')[:, 1]
        
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

