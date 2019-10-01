import numpy as np

from scipy.stats import norm

from abc import ABC, abstractmethod

from .double_ml_data import DoubleMLData


class DoubleML(ABC):
    """
    Double Machine Learning
    """
    def __init__(self,
                 resampling,
                 ml_learners,
                 dml_procedure,
                 inf_model,
                 n_rep_cross_fit=1):
        self.resampling = resampling
        self.ml_learners = ml_learners
        self.dml_procedure = dml_procedure
        self.inf_model = self._check_inf_method(inf_model)
        self.n_rep_cross_fit = n_rep_cross_fit
    
    @property 
    def score(self):
        return self._score

    @property
    def n_obs(self):
        return self.score.shape[0]

    @property
    def n_treat(self):
        return self.score.shape[1]
    
    @score.setter
    def score(self, value):
        self._score = value
    
    @property 
    def score_a(self):
        return self._score_a
    
    @score_a.setter
    def score_a(self, value):
        self._score_a[:, self._i_d] = value
    
    @property 
    def score_b(self):
        return self._score_b
    
    @score_b.setter
    def score_b(self, value):
        self._score_b[:, self._i_d] = value
    
    @property 
    def coef(self):
        return self._coef
    
    @coef.setter
    def coef(self, value):
        self._coef[self._i_d] = value
    
    @property 
    def se(self):
        return self._se
    
    @se.setter
    def se(self, value):
        self._se[self._i_d] = value

    @property
    def t_stat(self):
        t_stat = self.coef / self.se
        return t_stat

    @property
    def pval(self):
        pval = 2 * norm.cdf(-np.abs(self.t_stat))
        return pval
    
    @property 
    def boot_coef(self):
        return self._boot_coef
    
    @boot_coef.setter
    def boot_coef(self, value):
        self._boot_coef[self._i_d, :] = value
    
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
        return self._coef[self._i_d]
    
    @property 
    def __se_(self):
        return self._se[self._i_d]
    
    def fit(self, obj_dml_data, keep_scores=True):
        """
        Fit doubleML model
        Parameters
        ----------
        obj_dml_data : 
        Returns
        -------
        
        """
        
        assert isinstance(obj_dml_data, DoubleMLData)
        self._check_data(obj_dml_data)
        
        self._initialize_arrays(obj_dml_data.n_obs,
                                obj_dml_data.n_treat)

        all_coef = np.full((obj_dml_data.n_treat,
                            self.n_rep_cross_fit),
                           np.nan)

        all_se = np.full((obj_dml_data.n_treat,
                          self.n_rep_cross_fit),
                         np.nan)

        for i_rep in range(self.n_rep_cross_fit):
            # perform sample splitting
            self._split_samples(obj_dml_data.x)

            for i_d in range(self.n_treat):
                self._i_d = i_d

                # this step could be skipped for the single treatment variable case
                if self.n_treat > 1:
                    obj_dml_data._set_x_d(obj_dml_data.d_cols[i_d])

                # ml estimation of nuisance models and computation of score elements
                self._ml_nuisance_and_score_elements(obj_dml_data)

                # estimate the causal parameter(s)
                self._est_causal_pars()
            all_coef[:, i_rep] = self.coef
            all_se[:, i_rep] = self.se

        # don't use the setter (always for one treatment variable), but the private variable
        self._coef = np.median(all_coef, 1)
        self._se = np.median(all_se, 1)
        if not keep_scores:
            self._clean_scores()

    def bootstrap(self, method='normal', n_rep=500):
        """
        Bootstrap doubleML model
        Parameters
        ----------
        method : 
        n_rep : 
        Returns
        -------
        
        """
        if (not hasattr(self, 'coef')) or (self.coef is None):
            raise ValueError('apply fit() before bootstrap()')
        
        self._initialize_boot_arrays(n_rep)
        
        for i_d in range(self.n_treat):
            self._i_d = i_d
            
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
            
            J = np.mean(self.__score_a)
            self.boot_coef = np.matmul(weights, self.__score) / (self.n_obs * self.__se_ * J)

    @abstractmethod
    def _check_inf_method(self, inf_method):
        pass

    @abstractmethod
    def _check_data(self, obj_dml_data):
        pass

    @abstractmethod
    def _ml_nuisance_and_score_elements(self, obj_dml_data):
        pass

    def _initialize_arrays(self, n_obs, n_treat):
        self._score = np.full((n_obs, n_treat), np.nan)
        self._score_a = np.full((n_obs, n_treat), np.nan)
        self._score_b = np.full((n_obs, n_treat), np.nan)
        
        self._coef = np.full(n_treat, np.nan)
        self._se = np.full(n_treat, np.nan)

    def _initialize_boot_arrays(self, n_rep):
        self._boot_coef = np.full((self.n_treat, n_rep), np.nan)

    def _split_samples(self, x):
        resampling = self.resampling
        
        smpls = [(train, test) for train, test in resampling.split(x)]
        self._smpls = smpls
    
    def _est_causal_pars(self):
        dml_procedure = self.dml_procedure
        resampling = self.resampling
        smpls = self._smpls
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = self._orth_est(test_index)
            theta_hat = np.mean(thetas)
            self.coef = theta_hat
            self._compute_score()
            
            variances = np.zeros(resampling.get_n_splits())
            for idx, (train_index, test_index) in enumerate(smpls):
                variances[idx] = self._var_est(test_index)
            self.se = np.sqrt(np.mean(variances))
            
        elif dml_procedure == 'dml2':
            theta_hat = self._orth_est()
            self.coef = theta_hat
            self._compute_score()
            
            self.se = np.sqrt(self._var_est())
            
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
        n_obs = self.n_obs
        J = np.mean(score_a)
        sigma2_hat = 1/n_obs * np.mean(np.power(score, 2)) / np.power(J, 2)
        
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
        self.score = self.score_a * self.coef + self.score_b

    def _clean_scores(self):
        del self._score
        del self._score_a
        del self._score_b

