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
        Estimate the structural parameter in a partially linear model (PLR &
        PLIV)
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
        Estimate the standard errors of the structural parameter in a partially
        linear model (PLR & PLIV)
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
        
    def bootstrap(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        score = self._score
        J = np.mean(self._score_a)
        se = self.se_
        
        n_obs = len(score)
        
        if method == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=(n_rep, n_obs)) - 1.
        elif method == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
        elif method == 'wild':
            xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
            yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
            weights = xx / np.sqrt(2) + (np.power(yy,2) - 1)/2
        else:
            raise ValueError('invalid boot method')
        
        boot_coef = np.matmul(weights, score) / (n_obs * se * J)
        
        # alternatives (profiling not clear yet)
        # boot_coef = np.mean(np.multiply(weights, score),1) / (se * J)
        # boot_coef = np.dot(weights, score) / (n_obs * se * J)
        # boot_coef = np.linalg.multi_dot(weights, score) / (n_obs * se * J)
        
        self.boot_coef_ = boot_coef
        
        return
        
        
        