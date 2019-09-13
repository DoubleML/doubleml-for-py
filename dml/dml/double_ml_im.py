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

        # no need for initialization; assume a single treatment variable
        #self._score = np.full((self.n_obs, self.n_treat), np.nan)
        #self._score_a = np.full((self.n_obs, self.n_treat), np.nan)
        #self._score_b = np.full((self.n_obs, self.n_treat), np.nan)
        #self._initialize_arrays_nuisance()
        
    
#    def _orth_est(self, inds = None):
#        """
#        Estimate the structural parameter in a partially linear model (PLR &
#        PLIV)
#        """
#        score_a = self._score_a[:, self.ind_d]
#        score_b = self._score_b[:, self.ind_d]
#        
#        if inds is not None:
#            score_a = score_a[inds]
#            score_b = score_b[inds]
#        
#        theta = -np.mean(score_b)/np.mean(score_a)
#        
#        return theta

#    def _compute_score(self):
#        self._score[:, self.ind_d] = self._score_a[:, self.ind_d] * self.coef_[self.ind_d] + self._score_b[:, self.ind_d]
    
    def _var_est(self, inds = None):
        """
        Estimate the standard errors of the structural parameter in a partially
        linear model (PLR & PLIV)
        """
        score_a = self._score_a[:, self.ind_d]
        score = self._score[:, self.ind_d]
        
        if inds is not None:
            score_a = score_a[inds]
            score = score[inds]
        
        # don't understand yet the additional 1/n_obs
        n_obs_sample = len(score)
        J = np.mean(score_a)
        sigma2_hat = 1/n_obs_sample * np.mean(np.power(score, 2)) / np.power(J, 2)
        
        return sigma2_hat
    
    def _fit(self, X, y, d, z=None):
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
        
        self.n_treat = d.shape[1]
        self.n_obs = X.shape[0]
        assert self.n_treat == 1
        self.ind_d = 0
        
        # assure D binary
        assert type_of_target(d) == 'binary', 'variable d must be binary'
        
        if np.all(np.power(d,2) - d) == 0:
            raise ValueError('variable d must be binary with values 0 and 1')
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        # perform sample splitting
        self._split_samples(X)
        
        # get train indices for d==0 and d==1
        self._get_cond_smpls(d)
        
        if z is None:
            self._ml_nuisance(X, y, d)
        else:
            self._ml_nuisance(X, y, d, z)
        self._compute_score_elements(d)
            
        # estimate the causal parameter(s)
        self._est_causal_pars()
        
        self.i_d = None
            
        t = self.coef_ / self.se_
        pval = 2 * norm.cdf(-np.abs(t))
        self.t_ = t
        self.pval_ = pval
        
        return
    
    def _get_cond_smpls(self, d):
        smpls = self._smpls
        self._smpls_d0 = [(np.intersect1d(np.where(d==0)[0], train),
                           test) for train, test in smpls]
        self._smpls_d1 = [(np.intersect1d(np.where(d==1)[0], train),
                           test) for train, test in smpls]
    
    def bootstrap(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        # can be asserted here 
        #n_cols_d = len(self.coef_)
        #n_obs = self._score.shape[0]
        
        score = self._score
        J = np.mean(self._score_a)
        se = self.se_
        
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
        
        boot_coef = np.matmul(weights, score) / (self.n_obs * se * J)
        
        self.boot_coef_ = boot_coef
        
        return
