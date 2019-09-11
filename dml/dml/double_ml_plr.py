import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml_pl import DoubleMLPL

class DoubleMLPLR(DoubleMLPL):
    """
    Double Machine Learning for Partially Linear Regression
    """
    
    def _initialize_arrays_nuisance(self, n_obs, n_cols_d):
        self.g_hat = np.full((n_obs, n_cols_d), np.nan)
        self.m_hat = np.full((n_obs, n_cols_d), np.nan)
        self._u_hat = np.full((n_obs, n_cols_d), np.nan)
        self._v_hat = np.full((n_obs, n_cols_d), np.nan)
        self._v_hatd = np.full((n_obs, n_cols_d), np.nan)
    
    def _ml_nuisance(self, X, y, d, ind_d):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        
        # nuisance g
        self.g_hat[:, ind_d] = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        self.m_hat[:, ind_d] = cross_val_predict(ml_m, X, d, cv = smpls)
        
        # compute residuals
        self._u_hat[:, ind_d] = y - self.g_hat[:, ind_d]
        self._v_hat[:, ind_d] = d - self.m_hat[:, ind_d]
        self._v_hatd[:, ind_d] = np.multiply(self._v_hat[:, ind_d], d)
    
    def _compute_score_elements(self, ind_d):
        inf_model = self.inf_model
        
        u_hat = self._u_hat[:, ind_d]
        v_hat = self._v_hat[:, ind_d]
        v_hatd = self._v_hatd[:, ind_d]
        
        if inf_model == 'IV-type':
            self._score_a[:, ind_d] = -v_hatd
        elif inf_model == 'DML2018':
            self._score_a[:, ind_d] = -np.multiply(v_hat,v_hat)
        else:
            raise ValueError('invalid inf_model')
        self._score_b[:, ind_d] = np.multiply(v_hat,u_hat)
    
    
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
        self._fit(X, y, d)
        
        return
    
