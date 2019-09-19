import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.utils.multiclass import type_of_target

from scipy.stats import norm

from .double_ml import DoubleML

class DoubleMLIM(DoubleML):
    """
    Double Machine Learning for Interactive Models (IRM & IIVM)
    """
    
    def _fit_double_ml_im(self, X, y, d, z=None):
        """
        Fit doubleML model for PLR & PLIV
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
        
        self.n_obs = X.shape[0]
        assert d.ndim == 1
        self.n_treat = 1
        
        # assure D binary
        assert type_of_target(d) == 'binary', 'variable d must be binary'
        
        if np.any(np.power(d,2) - d != 0):
            raise ValueError('variable d must be binary with values 0 and 1')
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        # perform sample splitting
        self._split_samples(X)
        
        
        if z is None:
            # get train indices for d==0 and d==1
            self._get_cond_smpls(d)
            self._ml_nuisance(X, y, d)
            self._compute_score_elements(d)
        else:
            # get train indices for d==0 and d==1
            self._get_cond_smpls(z)
            self._ml_nuisance(X, y, d, z)
            self._compute_score_elements(z)
            
        # estimate the causal parameter(s)
        self._est_causal_pars()
            
        t = self.coef_ / self.se_
        pval = 2 * norm.cdf(-np.abs(t))
        self.t_ = t
        self.pval_ = pval
        
        return
    
    def bootstrap(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        # can be asserted here 
        #n_cols_d = len(self.coef_)
        #n_obs = self.score.shape[0]
        
        self.boot_coef_ = self._bootstrap_single_treat(method, n_rep)
        
        return
