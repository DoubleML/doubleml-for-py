import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml_im import DoubleMLIM

class DoubleMLPIIVM(DoubleMLIM):
    """
    Double Machine Learning for Interactive IV Model
    """
    

    def _get_cond_smpls(self, z):
        smpls = self._smpls
        self._smpls_z0 = [(np.intersect1d(np.where(z==0)[0], train),
                           test) for train, test in smpls]
        self._smpls_z1 = [(np.intersect1d(np.where(z==1)[0], train),
                           test) for train, test in smpls]
    
    def _ml_nuisance(self, X, y, d, z):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        ml_r = self.ml_learners['ml_r']
        
        X, y = check_X_y(X, y)
        X, z = check_X_y(X, z)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        smpls_z0 = self._smpls_z0
        smpls_z1 = self._smpls_z1
        
        # nuisance g
        inf_model = self.inf_model
        self.g_hat0 = cross_val_predict(ml_g, X, y, cv = smpls_z0)
        self.g_hat1 = cross_val_predict(ml_g, X, y, cv = smpls_z1)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, z, cv = smpls, method='predict_proba')[:, 1]
        
        # nuisance r
        inf_model = self.inf_model
        self.r_hat0 = cross_val_predict(ml_r, X, d, cv = smpls_z0, method='predict_proba')[:, 1]
        self.r_hat1 = cross_val_predict(ml_r, X, d, cv = smpls_z1, method='predict_proba')[:, 1]
        
        # compute residuals
        self._u_hat0 = y - self.g_hat0
        self._u_hat1 = y - self.g_hat1
        self._v_hat = z - self.m_hat
        self._w_hat0 = d - self.r_hat0
        self._w_hat1 = d - self.r_hat1
    
    def _compute_score_elements(self, z):
        inf_model = self.inf_model
        if inf_model == 'LATE':
            self.score_b = self.g_hat1 - self.g_hat0 \
                            + np.divide(np.multiply(z, self._u_hat1), self.m_hat) \
                            - np.divide(np.multiply(1.0-z, self._u_hat1), 1.0 - self.m_hat)
            self.score_a = -1*(self.r_hat1 - self.r_hat0 \
                                + np.divide(np.multiply(z, self._w_hat1), self.m_hat) \
                                - np.divide(np.multiply(1.0-z, self._w_hat0), 1.0 - self.m_hat))
        else:
            raise ValueError('invalid inf_model')
    
    
    def fit(self, X, y, d, z):
        """
        Fit doubleML model for PLR
        Parameters
        ----------
        X : 
        y : 
        d : 
        z : 
        Returns
        -------
        self: resturns an instance of DoubleMLPLR
        """
        self._fit_double_ml_im(X, y, d, z)
        
        return
    
