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
                 inf_model,
                 boot = 'normal',
                 n_rep = 500):
        self.resampling = resampling
        self.ml_learners = ml_learners
        self.dml_procedure = dml_procedure
        self.inf_model = inf_model
        self.boot = boot
        self.n_rep = n_rep
    
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

