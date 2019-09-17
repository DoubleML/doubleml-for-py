import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml_im import DoubleMLIM

class DoubleMLPIRM(DoubleMLIM):
    """
    Double Machine Learning for Interactive Regression Model
    """
    
    def _get_cond_smpls(self, d):
        smpls = self._smpls
        self._smpls_d0 = [(np.intersect1d(np.where(d==0)[0], train),
                           test) for train, test in smpls]
        self._smpls_d1 = [(np.intersect1d(np.where(d==1)[0], train),
                           test) for train, test in smpls]
    
    def _ml_nuisance(self, X, y, d):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        smpls_d0 = self._smpls_d0
        smpls_d1 = self._smpls_d1
        
        # nuisance g
        inf_model = self.inf_model
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
            self._score_b = self.g_hat1 - self.g_hat0 \
                            + np.divide(np.multiply(d, self._u_hat1), self.m_hat) \
                            - np.divide(np.multiply(1.0-d, self._u_hat0), 1.0 - self.m_hat)
            self._score_a = np.full_like(self._score_b, -1.0)
        elif inf_model == 'ATTE':
            p = np.mean(d)
            self._score_b = np.multiply(d, self._u_hat0) / p \
                            - np.divide(np.multiply(self.m_hat, np.multiply(1.0-d, self._u_hat0)),
                                        p*(1.0 - self.m_hat))
            self._score_a = - d / p
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
        self._fit_double_ml_im(X, y, d)
        
        return
    
