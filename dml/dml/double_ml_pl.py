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
    
    def _initialize_arrays(self):
        par_dict = dict()
        par_dict['_score'] = np.full((self.n_obs, self.n_treat), np.nan)
        par_dict['_score_a'] = np.full((self.n_obs, self.n_treat), np.nan)
        par_dict['_score_b'] = np.full((self.n_obs, self.n_treat), np.nan)
        
        return par_dict
    
    def _fit_double_ml_pl(self, X, y, d, z=None, export_scores=True):
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
        
        self.n_treat = d.shape[1]
        self.n_obs = X.shape[0]
        
        n_cols_X = X.shape[1]
        
        coef_ = np.full(self.n_treat, np.nan)
        se_ = np.full(self.n_treat, np.nan)
        if export_scores:
            par_dict = self._initialize_arrays()
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        # perform sample splitting
        self._split_samples(Xd)
        
        for i_d in range(self.n_treat):
            this_Xd = np.delete(Xd, n_cols_X + i_d, axis=1)
            # ml estimation of nuisance models
            if z is None:
                self._ml_nuisance(this_Xd, y, d[:, i_d])
            else:
                self._ml_nuisance(this_Xd, y, d[:, i_d], z)
            self._compute_score_elements()
            
            # estimate the causal parameter(s)
            self._est_causal_pars()
            
            coef_[i_d] =self.coef_
            se_[i_d] =self.se_
            if export_scores:
                par_dict['_score'][:, i_d] = self._score
                par_dict['_score_a'][:, i_d] = self._score_a
                par_dict['_score_b'][:, i_d] = self._score_b
        
        # setting final estimates and scores
        self.coef_ = coef_
        self.se_ = se_
        if export_scores:
            self._score = par_dict['_score']
            self._score_a = par_dict['_score_a']
            self._score_b = par_dict['_score_b']
        else:
            self._score = None
            self._score_a = None
            self._score_b = None
        
        t = self.coef_ / self.se_
        pval = 2 * norm.cdf(-np.abs(t))
        self.t_ = t
        self.pval_ = pval
        
        return
        
    def bootstrap(self, method = 'normal', n_rep = 500):
        if self.coef_ is None:
            raise ValueError('apply fit() before bootstrap()')
        
        # can be asserted here 
        #n_cols_d = len(self.coef_)
        #n_obs = self._score.shape[0]
        
        boot_coef = np.full((self.n_treat, n_rep), np.nan)
        
        for i_d in range(self.n_treat):
            
            score = self._score[:, i_d]
            J = np.mean(self._score_a[:, i_d])
            se = self.se_[i_d]
            
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
            
            boot_coef[i_d, :] = np.matmul(weights, score) / (self.n_obs * se * J)
            
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

