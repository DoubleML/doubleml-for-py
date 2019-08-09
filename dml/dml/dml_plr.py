import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_predict


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
        
        orth_dml_plr_obj = OrthDmlPlr(inf_model)
        
        smpls = [(train, test) for train, test in resampling.split(X)]
        
        X, y = check_X_y(X, y)
        X, d = check_X_y(X, d)
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(resampling.get_n_splits())
            
            # nuisance g
            g_hat = cross_val_predict(ml_g, X, y, cv = smpls)
            
            # nuisance m
            m_hat = cross_val_predict(ml_m, X, d, cv = smpls)
            
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = orth_dml_plr_obj.fit(y[test_index], d[test_index],
                                                   g_hat[test_index], m_hat[test_index]).coef_
            theta_hat = np.mean(thetas)
        else:
            raise ValueError('invalid dml_procedure')
        
        self.coef_ = theta_hat
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
            results = ols.fit(u_hat, v_hat.reshape(-1, 1))
            theta = results.coef_
        else:
            raise ValueError('invalid inf_model')
        
        self.coef_ = theta
        return self
