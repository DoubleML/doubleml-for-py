from sklear.utils import check_X_y

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
        X, y = check_X_y(X, y)
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(smpl_splitter.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpl_splitter.split(X)):
                
                # nuisance g
                g_hat = ml_g.fit(X[train_index],
                                 y[train_index]).predict(X[test_index])
                u_hat = y[test_index] - g_hat
                
                # nuisance m
                m_hat = ml_m.fit(X[train_index],
                                 d[train_index]).predict(X[test_index])
                v_hat = d[test_index] - m_hat
                
                thetas[idx] = np.mean(np.dot(vhat, (Y[test_index] - ghat)))/np.mean(np.dot(vhat, D[test_index]))
            theta_hat = np.mean(thetas)
        
        return self
    
    def _orth_dml_plr(u_hat, v_hat, v_hatd, inf_model):
        
        return theta
    
