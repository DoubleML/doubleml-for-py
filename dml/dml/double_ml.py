import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

class DoubleML:
    """
    Double Machine Learning
    """
    def __init__(self,
                 resampling,
                 ml_learners,
                 dml_procedure,
                 inf_model):
        self.resampling = resampling
        self.ml_learners = ml_learners
        self.dml_procedure = dml_procedure
        self.inf_model = inf_model
    
    def _split_samples(self, X):
        resampling = self.resampling
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        self._smpls = smpls
    
    def _est_causal_pars(self):
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        resampling = self.resampling
        smpls = self._smpls
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = self._orth_est(test_index)
            theta_hat = np.mean(thetas)
            self.coef_ = theta_hat
            self._compute_score()
            
            vars = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                vars[idx] = self._var_est(test_index)
            self.se_ = np.sqrt(np.mean(vars))
            
        elif dml_procedure == 'dml2':
            theta_hat = self._orth_est()
            self.coef_ = theta_hat
            self._compute_score()
            
            self.se_ = np.sqrt(self._var_est())
            
        else:
            raise ValueError('invalid dml_procedure')
    
    def _var_est(self, inds = None):
        """
        Estimate the standard errors of the structural parameter
        """
        score_a = self._score_a
        score = self._score
        
        if inds is not None:
            score_a = score_a[inds]
            score = score[inds]
        
        # don't understand yet the additional 1/n_obs
        n_obs_sample = len(score)
        J = np.mean(score_a)
        sigma2_hat = 1/n_obs_sample * np.mean(np.power(score, 2)) / np.power(J, 2)
        
        return sigma2_hat    
        
    def _orth_est(self, inds = None):
        """
        Estimate the structural parameter
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

