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
    
    def _fit_nuisance_and_causal(self, X, y, d, z=None):
        # only a single treatment variable is allowed
        assert self.n_treat == 1
        self._i_d = 0
        D = d[:, 0]
        
        # assure D binary
        assert type_of_target(D) == 'binary', 'variable d must be binary'
        
        if np.any(np.power(D,2) - D != 0):
            raise ValueError('variable d must be binary with values 0 and 1')
        
        if z is None:
            # get train indices for d==0 and d==1
            self._get_cond_smpls(D)
            self._ml_nuisance(X, y, D)
            self._compute_score_elements(D)
        else:
            # get train indices for d==0 and d==1
            self._get_cond_smpls(z)
            self._ml_nuisance(X, y, D, z)
            self._compute_score_elements(z)
            
        # estimate the causal parameter(s)
        self._est_causal_pars()

