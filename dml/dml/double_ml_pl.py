import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from .double_ml import DoubleML

class DoubleMLPL(DoubleML):
    """
    Double Machine Learning for Partially Linear Models (PLR & PLIV)
    """
    
    def _fit_nuisance_and_causal(self, X, y, d, z=None):
        Xd = np.hstack((X,d))
        n_cols_X = X.shape[1]
        
        for i_d in range(self.n_treat):
            self._i_d = i_d
            
            this_Xd = np.delete(Xd, n_cols_X + i_d, axis=1)
            # ml estimation of nuisance models
            if z is None:
                self._ml_nuisance(this_Xd, y, d[:, i_d])
            else:
                self._ml_nuisance(this_Xd, y, d[:, i_d], z)
            self._compute_score_elements()
            
            # estimate the causal parameter(s)
            self._est_causal_pars()

