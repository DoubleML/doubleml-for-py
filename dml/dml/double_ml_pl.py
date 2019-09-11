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


    def _initialize_arrays(self, n_obs, n_cols_d):
        self.coef_ = np.full(n_cols_d, np.nan)
        self.se_ = np.full(n_cols_d, np.nan)

        self._score = np.full((n_obs, n_cols_d), np.nan)
        self._score_a = np.full((n_obs, n_cols_d), np.nan)
        self._score_b = np.full((n_obs, n_cols_d), np.nan)
        self._initialize_arrays_nuisance(n_obs, n_cols_d)
        
    
    def _orth_est(self, ind_d, inds = None):
        """
        Estimate the structural parameter in a partially linear model (PLR &
        PLIV)
        """
        score_a = self._score_a[:, ind_d]
        score_b = self._score_b[:, ind_d]
        
        if inds is not None:
            score_a = score_a[inds]
            score_b = score_b[inds]
        
        theta = -np.mean(score_b)/np.mean(score_a)
        
        return theta

    def _compute_score(self, ind_d):
        self._score[:, ind_d] = self._score_a[:, ind_d] * self.coef_[ind_d] + self._score_b[:, ind_d]
    
    def _var_est(self, ind_d, inds = None):
        """
        Estimate the standard errors of the structural parameter in a partially
        linear model (PLR & PLIV)
        """
        score_a = self._score_a[:, ind_d]
        score = self._score[:, ind_d]
        
        if inds is not None:
            score_a = score_a[inds]
            score = score[inds]
        
        # don't understand yet the additional 1/n_obs
        n_obs = len(score)
        J = np.mean(score_a)
        sigma2_hat = 1/n_obs * np.mean(np.power(score, 2)) / np.power(J, 2)
        
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
        
        X = assure_2d_array(X)
        d = assure_2d_array(d)
        Xd = np.hstack((X,d))
        
        n_cols_X = X.shape[1]
        n_cols_d = d.shape[1]
        n_obs = X.shape[0]
        
        self._initialize_arrays(n_obs, n_cols_d)
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        # perform sample splitting
        self._split_samples(Xd)
        
        for i_d in range(n_cols_d):
            this_Xd = np.delete(Xd, n_cols_X + i_d, axis=1)
            # ml estimation of nuisance models
            if z is None:
                self._ml_nuisance(this_Xd, y, d[:, i_d], i_d)
            else:
                self._ml_nuisance(this_Xd, y, d[:, i_d], i_d, z)
            self._compute_score_elements(i_d)
            
            # estimate the causal parameter(s)
            self._est_causal_pars(i_d)
            
        t = self.coef_ / self.se_
        pval = 2 * norm.cdf(-np.abs(t))
        self.t_ = t
        self.pval_ = pval
        
        return
        
    def bootstrap(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        n_cols_d = len(self.coef_)
        n_obs = self._score.shape[0]
        
        boot_coef = np.full((n_cols_d, n_rep), np.nan)
        
        for i_d in range(n_cols_d):
            
            score = self._score[:, i_d]
            J = np.mean(self._score_a[:, i_d])
            se = self.se_[i_d]
            
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
            
            boot_coef[i_d, :] = np.matmul(weights, score) / (n_obs * se * J)
            
            # alternatives (profiling not clear yet)
            # boot_coef = np.mean(np.multiply(weights, score),1) / (se * J)
            # boot_coef = np.dot(weights, score) / (n_obs * se * J)
            # boot_coef = np.linalg.multi_dot(weights, score) / (n_obs * se * J)
        
        self.boot_coef_ = boot_coef
        
        return
        
        
def assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x

