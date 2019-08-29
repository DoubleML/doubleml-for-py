import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from dml.double_ml import DoubleML

class DoubleMLPL(DoubleML):
    """
    Double Machine Learning for Partially Linear Models (PLR & PLIV)
    """

    def _orth_est(self, inds = None):
        """
        Estimate the structural parameter in a partially linear regression model (PLR).
        Parameters
        """
        score_a = self._score_a
        score_b = self._score_b
        
        if inds is not None:
            score_a = score_a[inds]
            score_b = score_b[inds]
        
        theta = -np.mean(score_b)/np.mean(score_a)
        
        return theta

    def _compute_score(self):
        self._score = self._score_a * self.coef_ + self._score_b
    
    def _var_est(self, inds = None):
        """
        Estimate the structural parameter in a partially linear regression model (PLR).
        Parameters
        """
        score_a = self._score_a
        score = self._score
        
        if inds is not None:
            score_a = score_a[inds]
            score = score[inds]
        
        # don't understand yet the additional 1/n_obs
        n_obs = len(score)
        J = np.mean(score_a)
        sigma2_hat = 1/n_obs * np.mean(np.power(score, 2)) / np.power(J, 2)
        
        return sigma2_hat