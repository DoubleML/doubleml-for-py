import numpy as np
import pandas as pd

from scipy.stats import norm

from abc import ABC, abstractmethod

import warnings

from .double_ml_data import DoubleMLData
from .double_ml_resampling import DoubleMLResampling


class DoubleML(ABC):
    """
    Double Machine Learning
    """
    def __init__(self,
                 obj_dml_data,
                 n_folds,
                 ml_learners,
                 dml_procedure,
                 inf_model,
                 n_rep_cross_fit=1):
        # check and pick up obj_dml_data
        assert isinstance(obj_dml_data, DoubleMLData)
        self._check_data(obj_dml_data)
        self._dml_data = obj_dml_data

        self.n_folds = n_folds
        self.smpls = None
        self.ml_learners = ml_learners
        self.dml_procedure = dml_procedure
        self.inf_model = self._check_inf_method(inf_model)
        self.n_rep_cross_fit = n_rep_cross_fit
        self._ml_nuiscance_params = None

        # initialize arrays according to obj_dml_data and the resampling settings
        self._initialize_arrays()

    @property
    def n_obs(self):
        return self._dml_data.n_obs

    @property
    def n_treat(self):
        return self._dml_data.n_treat

    @property
    def d_cols(self):
        return self._dml_data.d_cols

    @property
    def smpls(self):
        return self._smpls

    @smpls.setter
    def smpls(self, value):
        self._smpls = value

    @property
    def score(self):
        return self._score
    
    @property 
    def score_a(self):
        return self._score_a
    
    @property 
    def score_b(self):
        return self._score_b
    
    @property 
    def coef(self):
        return self._coef
    
    @coef.setter
    def coef(self, value):
        self._coef = value
    
    @property 
    def se(self):
        return self._se
    
    @se.setter
    def se(self, value):
        self._se = value

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

    @property
    def summary(self):
        col_names = ['coef', 'std err', 't', 'P>|t|']
        if self.d_cols is None:
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = np.transpose(np.vstack(
                [self.coef, self.se,
                 self.t_stat, self.pval]))
            df_summary = pd.DataFrame(summary_stats,
                                      columns=col_names,
                                      index=self.d_cols)
            ci = self.confint()
            df_summary = df_summary.join(ci)
        return df_summary
    
    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.
    @property
    def __smpls(self):
        return self.smpls[self._i_rep]

    @property 
    def __score(self):
        return self._score[:, self._i_rep, self._i_treat]

    @__score.setter
    def __score(self, value):
        self._score[:, self._i_rep, self._i_treat] = value

    @property
    def __score_a(self):
        return self._score_a[:, self._i_rep, self._i_treat]

    @__score_a.setter
    def __score_a(self, value):
        self._score_a[:, self._i_rep, self._i_treat] = value
    
    @property 
    def __score_b(self):
        return self._score_b[:, self._i_rep, self._i_treat]

    @__score_b.setter
    def __score_b(self, value):
        self._score_b[:, self._i_rep, self._i_treat] = value

    @property
    def __boot_coef(self):
        ind_start = self._i_rep * self.n_rep_boot
        ind_end = (self._i_rep + 1) * self.n_rep_boot
        return self._boot_coef[self._i_treat, ind_start:ind_end]

    @__boot_coef.setter
    def __boot_coef(self, value):
        ind_start = self._i_rep * self.n_rep_boot
        ind_end = (self._i_rep + 1) * self.n_rep_boot
        self._boot_coef[self._i_treat, ind_start:ind_end] = value

    @property
    def __all_coef(self):
        return self._all_coef[self._i_treat, self._i_rep]

    @__all_coef.setter
    def __all_coef(self, value):
        self._all_coef[self._i_treat, self._i_rep] = value

    @property
    def __all_se(self):
        return self._all_se[self._i_treat, self._i_rep]

    @__all_se.setter
    def __all_se(self, value):
        self._all_se[self._i_treat, self._i_rep] = value
    
    def fit(self, se_reestimate=False, n_jobs_cv=None, keep_scores=True):
        """
        Fit doubleML model
        Parameters
        ----------
        obj_dml_data : 
        Returns
        -------
        
        """

        # TODO: Check whether this check is still needed
        #if self.n_rep_cross_fit > 1:
        #    # externally transferred samples not supported for repeated cross-fitting
        #    assert self.smpls is None, 'externally transferred samples not supported for repeated cross-fitting'

        # perform sample splitting
        if self.smpls is None:
            self._split_samples()

        for i_rep in range(self.n_rep_cross_fit):
            self._i_rep = i_rep
            for i_d in range(self.n_treat):
                self._i_treat = i_d

                if self._ml_nuiscance_params is not None:
                    self._set_ml_nuisance_params(self._ml_nuiscance_params[i_rep][i_d])

                # this step could be skipped for the single treatment variable case
                if self.n_treat > 1:
                    self._dml_data._set_x_d(self.d_cols[i_d])

                # ml estimation of nuisance models and computation of score elements
                self.__score_a, self.__score_b = self._ml_nuisance_and_score_elements(self._dml_data, self.__smpls, n_jobs_cv)

                # estimate the causal parameter
                self.__all_coef = self._est_causal_pars()

                # compute score (depends on estimated causal parameter)
                self._compute_score()

                # compute standard errors for causal parameter
                self.__all_se = self._se_causal_pars(se_reestimate)

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self._agg_cross_fit()

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

        dml_procedure = self.dml_procedure
        
        self._initialize_boot_arrays(n_rep, self.n_rep_cross_fit)

        for i_rep in range(self.n_rep_cross_fit):
            self._i_rep = i_rep
            for i_d in range(self.n_treat):
                self._i_treat = i_d

                self.__boot_coef = self._compute_bootstrap(method, n_rep)


    def confint(self, joint=False, level=0.95):
        a = (1 - level)
        ab = np.array([a/2, 1. - a/2])
        if joint:
            sim = np.amax(np.abs(self.boot_coef), 0)
            hatc = np.quantile(sim, 1-a)
            hatc_two_sided = np.array([-hatc, hatc])
            ci = self.coef + self.se * hatc_two_sided
        else:
            fac = norm.ppf(ab)
            ci = self.coef + self.se * fac

        df_ci = pd.DataFrame([ci],
                             columns=['{:.1f} %'.format(i * 100) for i in ab],
                             index=self.d_cols)
        return df_ci

    def tune(self,
             param_grids,
             scoring_methods=None, # if None the estimator's score method is used
             n_folds_tune=5,
             n_jobs_cv=None,
             set_as_params=True):

        self._ml_nuiscance_params = [[None] * self.n_treat] * self.n_rep_cross_fit
        tuning_res = [[None] * self.n_treat] * self.n_rep_cross_fit

        # TODO: Check whether this check is still needed
        if self.n_rep_cross_fit > 1:
            # externally transferred samples not supported for repeated cross-fitting
            assert self.smpls is None, 'externally transferred samples not supported for repeated cross-fitting'

        # perform sample splitting
        if self.smpls is None:
            self._split_samples()

        for i_rep in range(self.n_rep_cross_fit):
            self._i_rep = i_rep
            for i_d in range(self.n_treat):
                self._i_treat = i_d

                # this step could be skipped for the single treatment variable case
                if self.n_treat > 1:
                    self._dml_data._set_x_d(self.d_cols[i_d])

                # ml estimation of nuisance models and computation of score elements
                res = self._ml_nuisance_tuning(self._dml_data, self.__smpls,
                                               param_grids, scoring_methods,
                                               n_folds_tune,
                                               n_jobs_cv)

                tuning_res[i_rep][i_d] = res
                self._ml_nuiscance_params[i_rep][i_d] = res['params']

        return tuning_res

    def set_ml_nuisance_params(self, params):
        if isinstance(params, dict):
            warnings.warn("Using the same (hyper-)parameters for all repeated cross-fits and treatment variables")
            self._ml_nuiscance_params = [[params] * self.n_treat] * self.n_rep_cross_fit
        else:
            assert len(params) == self.n_rep_cross_fit
            assert np.all(np.array([len(x) for x in params]) == self.n_treat)
            self._ml_nuiscance_params = params

    @abstractmethod
    def _check_inf_method(self, inf_method):
        pass

    @abstractmethod
    def _check_data(self, obj_dml_data):
        pass

    @abstractmethod
    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        pass

    @abstractmethod
    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        pass

    @abstractmethod
    def _set_ml_nuisance_params(self, params):
        pass

    def _initialize_arrays(self):
        self._score = np.full((self.n_obs, self.n_rep_cross_fit, self.n_treat), np.nan)
        self._score_a = np.full((self.n_obs, self.n_rep_cross_fit, self.n_treat), np.nan)
        self._score_b = np.full((self.n_obs, self.n_rep_cross_fit, self.n_treat), np.nan)
        
        self._coef = np.full(self.n_treat, np.nan)
        self._se = np.full(self.n_treat, np.nan)

        self._all_coef = np.full((self.n_treat, self.n_rep_cross_fit), np.nan)
        self._all_se = np.full((self.n_treat, self.n_rep_cross_fit), np.nan)

    def _initialize_boot_arrays(self, n_rep, n_rep_cross_fit):
        self.n_rep_boot = n_rep
        self._boot_coef = np.full((self.n_treat, n_rep * n_rep_cross_fit), np.nan)

    def _split_samples(self):
        obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                n_rep_cross_fit=self.n_rep_cross_fit,
                                                n_obs=self.n_obs)
        self.smpls = obj_dml_resampling.split_samples()

    def set_samples(self, all_smpls):
        # TODO warn if n_rep_cross_fit or n_folds is overwritten with different number induced by the transferred
        # TODO external samples?
        self.n_rep_cross_fit = len(all_smpls)
        n_folds_each_smpl = np.array([len(smpl) for smpl in all_smpls])
        assert np.all(n_folds_each_smpl == n_folds_each_smpl[0]), 'Different number of folds for repeated cross-fitting'
        self.n_folds = n_folds_each_smpl[0]
        self.smpls = all_smpls
        self._initialize_arrays()
    
    def _est_causal_pars(self):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls
        
        if dml_procedure == 'dml1':
            thetas = np.zeros(self.n_folds)
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = self._orth_est(test_index)
            theta_hat = np.mean(thetas)
            coef = theta_hat
            
        elif dml_procedure == 'dml2':
            theta_hat = self._orth_est()
            coef = theta_hat
            
        else:
            raise ValueError('invalid dml_procedure')

        return coef

    def _se_causal_pars(self, se_reestimate):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls

        if dml_procedure == 'dml1':
            if se_reestimate:
                se = np.sqrt(self._var_est())
            else:
                variances = np.zeros(self.n_folds)
                for idx, (train_index, test_index) in enumerate(smpls):
                    variances[idx] = self._var_est(test_index)
                se = np.sqrt(np.mean(variances))

        elif dml_procedure == 'dml2':
            se = np.sqrt(self._var_est())

        else:
            raise ValueError('invalid dml_procedure')

        return se

    def _agg_cross_fit(self):
        # aggregate parameters from the repeated cross-fitting
        # don't use the getter (always for one treatment variable and one sample), but the private variable
        self.coef = np.median(self._all_coef, 1)
        xx = np.tile(self.coef.reshape(-1, 1), self.n_rep_cross_fit)
        self.se = np.sqrt(np.median(np.power(self._all_se, 2) - np.power(self._all_coef - xx, 2), 1))

    def _compute_bootstrap(self, method, n_rep):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls

        if method == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=(n_rep, self.n_obs)) - 1.
        elif method == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
        elif method == 'wild':
            xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
            yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, self.n_obs))
            weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
        else:
            raise ValueError('invalid boot method')

        if dml_procedure == 'dml1':
            boot_coefs = np.full((n_rep, self.n_folds), np.nan)
            for idx, (_, test_index) in enumerate(smpls):
                J = np.mean(self.__score_a[test_index])
                boot_coefs[:, idx] = np.matmul(weights[:, test_index], self.__score[test_index]) / (
                            len(test_index) * self.__all_se * J)
            boot_coef = np.mean(boot_coefs, axis=1)

        elif dml_procedure == 'dml2':
            J = np.mean(self.__score_a)
            boot_coef = np.matmul(weights, self.__score) / (self.n_obs * self.__all_se * J)

        else:
            raise ValueError('invalid dml_procedure')

        return boot_coef

    
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
        self.__score = self.__score_a * self.__all_coef + self.__score_b

    def _clean_scores(self):
        del self._score
        del self._score_a
        del self._score_b

