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
    
    def _orth_est(self, inds = None):
        """
        Estimate the structural parameter in a partially linear regression model (PLR).
        Parameters
        """
        u_hat = self._u_hat
        v_hat = self._v_hat
        v_hatd = self._v_hatd
        
        if inds is not None:
            u_hat = u_hat[inds]
            v_hat = v_hat[inds]
            v_hatd = v_hatd[inds]
        
        inf_model = self.inf_model
        
        if inf_model == 'IV-type':
            theta = np.mean(np.multiply(v_hat,u_hat))/np.mean(v_hatd)
        elif inf_model == 'DML2018':
            ols = LinearRegression(fit_intercept=False)
            results = ols.fit(v_hat.reshape(-1, 1), u_hat)
            theta = results.coef_
        else:
            raise ValueError('invalid inf_model')
        
        return theta
        
    
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
        
        #orth_dml_plr_obj = OrthDmlPlr(inf_model)
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        self._smpls = smpls
        
        # ml estimation of nuisance models 
        self._ml_nuisance(X, y, d)
        
        m_hat = self.m_hat
        g_hat = self.g_hat
        u_hat = self._u_hat
        v_hat = self._v_hat
        v_hatd = self._v_hatd
        
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = self._orth_est(test_index)
            theta_hat = np.mean(thetas)
            
            ses = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                ses[idx] = var_plr(theta_hat, d[test_index],
                                   u_hat[test_index], v_hat[test_index],
                                   v_hatd[test_index],
                                   se_type)
            se = np.sqrt(np.mean(ses))
            
        elif dml_procedure == 'dml2':
            theta_hat = self._orth_est()
            
            # comute standard errors
            u_hat = y - g_hat
            v_hat = d - m_hat
            se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, v_hatd, se_type))
            
        else:
            raise ValueError('invalid dml_procedure')
        
        t = theta_hat / se
        pval = 2 * norm.cdf(-np.abs(t))
        
        self.coef_ = theta_hat
        self.se_ = se
        self.t_ = t
        self.pval_ = pval
        return self


def var_plr(theta, d, u_hat, v_hat, v_hatd, se_type):
    n_obs = len(u_hat)
    
    if se_type == 'DML2018':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    elif se_type == 'IV-type':
        var = 1/n_obs * 1/np.power(np.mean(v_hatd), 2) * \
              np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))
    else:
        raise ValueError('invalid se_type')
    
    return var
     
    
