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
        
    def _est_boot_pars(self):
        boot = self.boot
        
        score = self._score
        J = np.mean(self._score_a)
        se = self.se_
        
        n_obs = len(score)
        boot_coef = np.zeros(self.n_rep)
        
        for i_rep in range(self.n_rep):
            if boot == 'Bayes':
                weights = np.random.exponential(scale=1.0, size=n_obs) - 1.
            elif boot == 'normal':
                weights = np.random.normal(loc=0.0, scale=1.0, size=n_obs)
            elif boot == 'wild':
                xx = np.random.normal(loc=0.0, scale=1.0, size=n_obs)
                yy = np.random.normal(loc=0.0, scale=1.0, size=n_obs)
                weights = xx / np.sqrt(2) + (np.power(yy,2) - 1)/2
            else:
                raise ValueError('invalid inf_model')
            
            boot_coef[i_rep] = np.mean(np.multiply(np.divide(weights, se),
                                                   score / J))
        return boot_coef
        
        
        