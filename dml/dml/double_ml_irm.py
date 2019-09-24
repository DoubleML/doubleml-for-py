import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml import DoubleML
from .helper import check_binary_vector

class DoubleMLPIRM(DoubleML):
    """
    Double Machine Learning for Interactive Regression Model
    """
    
    def _est_nuisance(self, obj_dml_data):
        assert obj_dml_data.z is None
        # get train indices for d==0 and d==1
        self._get_cond_smpls(obj_dml_data.d)
        self._ml_nuisance(obj_dml_data.X, obj_dml_data.y,
                          obj_dml_data.d)
        self._compute_score_elements(obj_dml_data.d)
        
        
    def _get_cond_smpls(self, d):
        smpls = self._smpls
        self._smpls_d0 = [(np.intersect1d(np.where(d==0)[0], train),
                           test) for train, test in smpls]
        self._smpls_d1 = [(np.intersect1d(np.where(d==1)[0], train),
                           test) for train, test in smpls]
    
    def _ml_nuisance(self, X, y, d):
        inf_model = self.inf_model
        
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        smpls_d0 = self._smpls_d0
        smpls_d1 = self._smpls_d1
        
        # fraction of treated for ATTE
        if inf_model == 'ATTE':
            self._p_hat = np.zeros_like(d, dtype='float64')
            for _, test_index in smpls:
                self._p_hat[test_index] = np.mean(d[test_index])
        
        
        # nuisance g
        self.g_hat0 = cross_val_predict(ml_g, X, y, cv = smpls_d0)
        if inf_model == 'ATE':
            self.g_hat1 = cross_val_predict(ml_g, X, y, cv = smpls_d1)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, d, cv = smpls, method='predict_proba')[:, 1]
        
        # compute residuals
        self._u_hat0 = y - self.g_hat0
        if inf_model == 'ATE':
            self._u_hat1 = y - self.g_hat1
    
    def _compute_score_elements(self, d):
        inf_model = self.inf_model
        
        if inf_model == 'ATE':
            self.score_b = self.g_hat1 - self.g_hat0 \
                            + np.divide(np.multiply(d, self._u_hat1), self.m_hat) \
                            - np.divide(np.multiply(1.0-d, self._u_hat0), 1.0 - self.m_hat)
            self.score_a = np.full_like(self.m_hat, -1.0)
        elif inf_model == 'ATTE':
            self.score_b = np.divide(np.multiply(d, self._u_hat0), self._p_hat) \
                            - np.divide(np.multiply(self.m_hat, np.multiply(1.0-d, self._u_hat0)),
                                        np.multiply(self._p_hat, (1.0 - self.m_hat)))
            self.score_a = - np.divide(d, self._p_hat)
        else:
            raise ValueError('invalid inf_model')
    
    
    def fit(self, X, y, d):
        """
        Fit doubleML model for PLR
        Parameters
        ----------
        X : 
        y : 
        d : 
        Returns
        -------
        self: resturns an instance of DoubleMLPLR
        """
        check_binary_vector(d, variable_name='d')
        self._fit_double_ml(X, y, d)
        
        return

