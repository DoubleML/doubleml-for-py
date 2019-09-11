import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml_pl import DoubleMLPL

class DoubleMLPLIV(DoubleMLPL):
    """
    Double Machine Learning for Partially Linear IV regression model
    """
    
    def _initialize_arrays_nuisance(self,n_obs, n_cols_d):
        self.g_hat = np.full((n_obs, n_cols_d), np.nan)
        self.m_hat = np.full((n_obs, n_cols_d), np.nan)
        self.r_hat = np.full((n_obs, n_cols_d), np.nan)
        self._u_hat = np.full((n_obs, n_cols_d), np.nan)
        self._v_hat = np.full((n_obs, n_cols_d), np.nan)
        self._w_hat = np.full((n_obs, n_cols_d), np.nan)
        self._v_hatd = np.full((n_obs, n_cols_d), np.nan)
    
    def _ml_nuisance(self, X, y, d, ind_d, z):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        ml_r = self.ml_learners['ml_r']
        
        X, y = check_X_y(X, y)
        X, z = check_X_y(X, z)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        
        # nuisance g
        self.g_hat[:, ind_d] = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        self.m_hat[:, ind_d] = cross_val_predict(ml_m, X, z, cv = smpls)
        
        # nuisance r
        self.r_hat[:, ind_d] = cross_val_predict(ml_r, X, d, cv = smpls)
        
        # compute residuals
        self._u_hat[:, ind_d] = y - self.g_hat[:, ind_d]
        self._v_hat[:, ind_d] = z - self.m_hat[:, ind_d]
        self._w_hat[:, ind_d] = d - self.r_hat[:, ind_d]
        self._v_hatd[:, ind_d] = np.multiply(self._v_hat[:, ind_d], d)
    
    def _compute_score_elements(self, ind_d):
        inf_model = self.inf_model
        
        u_hat = self._u_hat[:, ind_d]
        v_hat = self._v_hat[:, ind_d]
        w_hat = self._w_hat[:, ind_d]
        v_hatd = self._v_hatd[:, ind_d]
        
        if inf_model == 'DML2018':
            self._score_a[:, ind_d] = -np.multiply(w_hat,v_hat)
        else:
            # check whether its worth implementing the IV_type here as well
            # In CCDHNR equation (4.7) a score of this type is provided;
            # however in the following paragraph it is explained that one might
            # still need to estimate the DML2018 type first
            raise ValueError('invalid inf_model')
        self._score_b[:, ind_d] = np.multiply(v_hat,u_hat)
    
    
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
        self._fit(X, y, d, z)
        
        return
    
