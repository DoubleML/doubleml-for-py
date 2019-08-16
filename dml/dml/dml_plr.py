import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

from scipy.stats import norm


class DoubleMLPLR(object):
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
        ml_m = self.ml_learners['ml_m']
        ml_g = self.ml_learners['ml_g']
        resampling = self.resampling
        
        # TODO: se_type hard-coded to match inf_model
        se_type = inf_model
        
        orth_dml_plr_obj = OrthDmlPlr(inf_model)
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        # nuisance g
        g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
        
        # nuisance m
        m_hat = cross_val_predict(ml_m, X, d, cv = smpls)
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(resampling.get_n_splits())        
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = orth_dml_plr_obj.fit(y[test_index], d[test_index],
                                                   g_hat[test_index], m_hat[test_index]).coef_
            theta_hat = np.mean(thetas)
            
            se = np.nan 
            t = np.nan 
            pval = np.nan 
            
        elif dml_procedure == 'dml2':
            theta_hat = orth_dml_plr_obj.fit(y, d, g_hat, m_hat).coef_
            
            # comute standard errors
            u_hat = y - g_hat
            v_hat = d - m_hat
            se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, se_type))
            
            t = theta_hat / se
            pval = 2 * norm.cdf(-np.abs(t))
        else:
            raise ValueError('invalid dml_procedure')
        
        self.coef_ = theta_hat
        self.se_ = se
        self.t_ = t
        self.pval_ = pval
        return self

class OrthDmlPlr(object):
    """
    Orthogonalized Estimation of Coefficient in PLR
    """
    def __init__(self,
                 inf_model):
        self.inf_model = inf_model
    
    def fit(self, y, d, g_hat, m_hat):
        """
        Estimate the structural parameter in a partially linear regression model (PLR).
        Parameters
        ----------
        y : 
        d : 
        g_hat : 
        m_hat : 
        Returns
        -------
        self: resturns an instance of OrthDmlPlr
        """
        inf_model = self.inf_model
        
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.dot(v_hat, d)
        
        if inf_model == 'IV-type':
            theta = np.mean(np.dot(v_hat,u_hat))/np.mean(v_hatd)
        elif inf_model == 'DML2018':
            ols = LinearRegression(fit_intercept=False)
            results = ols.fit(v_hat.reshape(-1, 1), u_hat)
            theta = results.coef_
        else:
            raise ValueError('invalid inf_model')
        
        self.coef_ = theta
        return self


def var_plr(theta, d, u_hat, v_hat, se_type):
    v_hatd = np.dot(v_hat, d)
    n_obs = len(u_hat)
    
    if se_type == 'DML2018':
        var = 1/n_obs * 1/np.power(np.mean(np.dot(v_hat, v_hat)), 2) * \
              np.mean(np.power(np.dot(u_hat - v_hat*theta, v_hat), 2))
    elif se_type == 'IV-type':
        var = 1/n_obs * 1/np.power(np.mean(v_hatd), 2) * \
              np.mean(np.power(np.dot(u_hat - d*theta, v_hat), 2))
    else:
        raise ValueError('invalid se_type')
    
    return var
     
    
