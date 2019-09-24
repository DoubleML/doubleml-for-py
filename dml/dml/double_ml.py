import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from abc import ABC, abstractmethod

from .helper import assure_2d_array, double_ml_data_from_arrays

class DoubleML(ABC):
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
    
    @property 
    def score(self):
        return self._score
    
    @score.setter
    def score(self, score):
        self._score = score
    
    @property 
    def score_a(self):
        return self._score_a
    
    @score_a.setter
    def score_a(self, score_a):
        self._score_a[:, self._i_d] = score_a
    
    @property 
    def score_b(self):
        return self._score_b
    
    @score_b.setter
    def score_b(self, score_b):
        self._score_b[:, self._i_d] = score_b
    
    @property 
    def coef_(self):
        return self._coef_
    
    @coef_.setter
    def coef_(self, coef_):
        self._coef_[self._i_d] = coef_
    
    @property 
    def se_(self):
        return self._se_
    
    @se_.setter
    def se_(self, se_):
        self._se_[self._i_d] = se_
    
    @property 
    def boot_coef_(self):
        return self._boot_coef_
    
    @boot_coef_.setter
    def boot_coef_(self, boot_coef_):
        self._boot_coef_[self._i_d, :] = boot_coef_
    
    # the private properties with __ always deliver the single treatment subselection
    @property 
    def __score(self):
        return self._score[:, self._i_d]
    
    @property 
    def __score_a(self):
        return self._score_a[:, self._i_d]
    
    @property 
    def __score_b(self):
        return self._score_b[:, self._i_d]
    
    @property 
    def __coef_(self):
        return self._coef_[self._i_d]
    
    @property 
    def __se_(self):
        return self._se_[self._i_d]
    
    def _initialize_arrays(self):
        self._score = np.full((self.n_obs, self.n_treat), np.nan)
        self._score_a = np.full((self.n_obs, self.n_treat), np.nan)
        self._score_b = np.full((self.n_obs, self.n_treat), np.nan)
        
        self._coef_ = np.full(self.n_treat, np.nan)
        self._se_ = np.full(self.n_treat, np.nan)
    
    def _initialize_boot_arrays(self, n_rep):
        self._boot_coef_ = np.full((self.n_treat, n_rep), np.nan)
    
    def _split_samples(self, X):
        resampling = self.resampling
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        self._smpls = smpls
    
    def _fit_double_ml(self, X, y, d, z=None):
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
        
        obj_dml_data = double_ml_data_from_arrays(X, y, d, z)
        self.n_treat = obj_dml_data.n_treat
        self.n_obs = obj_dml_data.n_obs
        
        self._initialize_arrays()
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        # perform sample splitting
        self._split_samples(obj_dml_data.X)
        
        for i_d in range(self.n_treat):
            self._i_d = i_d
            
            # this step could be skipped for the single treatment variable case
            if self.n_treat > 1:
                obj_dml_data.extract_X_d(obj_dml_data.d_cols[i_d])
            
            # ml estimation of nuisance models
            self._est_nuisance(obj_dml_data)
                
            # estimate the causal parameter(s)
            self._est_causal_pars()
            
        t = self.coef_ / self.se_
        pval = 2 * norm.cdf(-np.abs(t))
        self.t_ = t
        self.pval_ = pval
    
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
        score_a = self.__score_a
        score = self.__score
        
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
        score_a = self.__score_a
        score_b = self.__score_b
        
        if inds is not None:
            score_a = score_a[inds]
            score_b = score_b[inds]
        
        theta = -np.mean(score_b)/np.mean(score_a)
        
        return theta
    
    def _compute_score(self):
        self.score = self.score_a * self.coef_ + self.score_b
    
    @abstractmethod
    def fit(self):
        pass
    
    def bootstrap(self, method, n_rep):
        if (not hasattr(self, 'coef_')) or (self.coef_ is None):
            raise ValueError('apply fit() before bootstrap()')
        
        self._initialize_boot_arrays(n_rep)
        
        for i_d in range(self.n_treat):
            self._i_d = i_d
            self._bootstrap_single_treat(method, n_rep)

    def _bootstrap_single_treat(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        score = self.__score
        J = np.mean(self.__score_a)
        se = self.__se_
        
        if method == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=(n_rep, self.n_obs)) - 1.
        elif method == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
        elif method == 'wild':
            xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
            yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
            weights = xx / np.sqrt(2) + (np.power(yy,2) - 1)/2
        else:
            raise ValueError('invalid boot method')
        
        self.boot_coef_ = np.matmul(weights, score) / (self.n_obs * se * J)

