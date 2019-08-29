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
    
    #@abstractmethod
    #def fit(self, X, y, d):
    #    pass
    #
    #def inference(self, X, y, d):
        
    
    
class DoubleMLPLR(DoubleML):
    """
    Double Machine Learning for Partially Linear Regression
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
    
    def _ml_nuisance(self, X, y, d):
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        smpls = self._smpls
        
        # nuisance g
        self.g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        self.m_hat = cross_val_predict(ml_m, X, d, cv = smpls)
        
        # compute residuals
        self._u_hat = y - self.g_hat
        self._v_hat = d - self.m_hat
        self._v_hatd = np.multiply(self._v_hat, d)
    
    def _compute_score_elements(self):
        inf_model = self.inf_model
        
        u_hat = self._u_hat
        v_hat = self._v_hat
        v_hatd = self._v_hatd
        
        if inf_model == 'IV-type':
            self._score_a = -v_hatd
        elif inf_model == 'DML2018':
            self._score_a = -np.multiply(v_hat,v_hat)
        else:
            raise ValueError('invalid inf_model')
        self._score_b = np.multiply(v_hat,u_hat)
    
    def _compute_score(self):
        self._score = self._score_a * self.coef_ + self._score_b
    
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
    
    def fit(self, X, y, d):
        """
        Fit doubleML model for PLR
        Parameters
        ----------
        X : 
        y : 
        d : 
        Returns
        -------
        self: resturns an instance of DoubleMLPLR
        """
        
        dml_procedure = self.dml_procedure
        inf_model = self.inf_model
        resampling = self.resampling
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        self._smpls = smpls
        
        # ml estimation of nuisance models 
        self._ml_nuisance(X, y, d)
        self._compute_score_elements()
        
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
            se = np.sqrt(np.mean(vars))
            
        elif dml_procedure == 'dml2':
            theta_hat = self._orth_est()
            self.coef_ = theta_hat
            self._compute_score()
            
            se = np.sqrt(self._var_est())
            
        else:
            raise ValueError('invalid dml_procedure')
        
        t = theta_hat / se
        pval = 2 * norm.cdf(-np.abs(t))
        
        self.se_ = se
        self.t_ = t
        self.pval_ = pval
        return self
    
