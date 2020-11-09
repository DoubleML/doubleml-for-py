import numpy as np
import pandas as pd
import warnings

from sklearn.base import is_regressor, is_classifier

from scipy.stats import norm

from statsmodels.stats.multitest import multipletests

from abc import ABC, abstractmethod

from .double_ml_data import DoubleMLData
from .double_ml_resampling import DoubleMLResampling


class DoubleML(ABC):
    """
    Double Machine Learning
    """

    def __init__(self,
                 obj_dml_data,
                 n_folds,
                 n_rep,
                 score,
                 dml_procedure,
                 draw_sample_splitting,
                 apply_cross_fitting):
        # check and pick up obj_dml_data
        assert isinstance(obj_dml_data, DoubleMLData)
        self._check_data(obj_dml_data)
        self._dml_data = obj_dml_data

        # initialize learners and parameters which are set model specific
        self._learner = None
        self._params = None

        # check resampling specifications
        if not isinstance(n_folds, int):
            raise TypeError('The number of folds must be of int type. '
                            f'{str(n_folds)} of type {str(type(n_folds))} was passed.')
        if n_folds < 1:
            raise ValueError('The number of folds must be positive. '
                             f'{str(n_folds)} was passed.')

        if not isinstance(n_rep, int):
            raise TypeError('The number of repetitions for the sample splitting must be of int type. '
                            f'{str(n_rep)} of type {str(type(n_rep))} was passed.')
        if n_rep < 1:
            raise ValueError('The number of repetitions for the sample splitting must be positive. '
                             f'{str(n_rep)} was passed.')

        if not isinstance(apply_cross_fitting, bool):
            raise TypeError('apply_cross_fitting must be True or False. '
                            f'got {str(apply_cross_fitting)}')
        if not isinstance(draw_sample_splitting, bool):
            raise TypeError('draw_sample_splitting must be True or False. '
                            f'got {str(draw_sample_splitting)}')

        # set resampling specifications
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.apply_cross_fitting = apply_cross_fitting

        # check and set dml_procedure and score
        if (not isinstance(dml_procedure, str)) | (dml_procedure not in ['dml1', 'dml2']):
            raise ValueError('dml_procedure must be "dml1" or "dml2" '
                             f' got {str(dml_procedure)}')
        self.dml_procedure = dml_procedure
        self.score = self._check_score(score)

        if (self.n_folds == 1) & self.apply_cross_fitting:
            warnings.warn('apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.')
            self.apply_cross_fitting = False

        if not self.apply_cross_fitting:
            assert self.n_folds <= 2, 'Estimation without cross-fitting not supported for n_folds > 2.'
            if self.dml_procedure == 'dml2':
                # redirect to dml1 which works out-of-the-box; dml_procedure is of no relevance without cross-fitting
                self.dml_procedure = 'dml1'

        # perform sample splitting
        if draw_sample_splitting:
            self.draw_sample_splitting()
        else:
            self.smpls = None

        # initialize arrays according to obj_dml_data and the resampling settings
        self._initialize_arrays()

        # initialize instance attributes which are later used for iterating
        self._i_rep = None
        self._i_treat = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        data_info = f'Outcome variable: {self._dml_data.y_col}\n' \
                    f'Treatment variable(s): {self._dml_data.d_cols}\n' \
                    f'Covariates: {self._dml_data.x_cols}\n' \
                    f'Instrument variable(s): {self._dml_data.z_cols}\n' \
                    f'No. Observations: {self._dml_data.n_obs}\n'
        score_info = f'Score function: {str(self.score)}\n' \
                     f'DML algorithm: {self.dml_procedure}\n'
        learner_info = ''
        for key, value in self.learner.items():
            learner_info += f'Learner {key}: {str(value)}\n'
        resampling_info = f'No. folds: {self.n_folds}\n' \
                          f'No. repeated sample splits: {self.n_rep}\n' \
                          f'Apply cross-fitting: {self.apply_cross_fitting}\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Data summary      ------------------\n' + data_info + \
            '\n------------------ Score & algorithm ------------------\n' + score_info + \
            '\n------------------ Machine learner   ------------------\n' + learner_info + \
            '\n------------------ Resampling        ------------------\n' + resampling_info + \
            '\n------------------ Fit summary       ------------------\n' + fit_summary
        return res

    @property
    def learner(self):
        """
        The machine learners for the nuisance functions.
        """
        return self._learner

    @property
    def learner_names(self):
        """
        The names of the learners.
        """
        return list(self._learner.keys())

    @property
    def params(self):
        """
        The hyperparameters of the learners.
        """
        return self._params

    @property
    def params_names(self):
        """
        The names of the nuisance models with hyperparameters.
        """
        return list(self._params.keys())

    def get_params(self, learner):
        """
        Get hyperparameters for the nuisance model of DoubleML models.

        Parameters
        ----------
        learner : str
            The nuisance model / learner (see attribute ``params_names``).

        Returns
        -------
        params : dict
            Parameters for the nuisance model / learner.
        """
        valid_learner = self.params_names
        if (not isinstance(learner, str)) | (learner not in valid_learner):
            raise ValueError('invalid nuisance learner ' + str(learner) +
                             '\n valid nuisance learner ' + ' or '.join(valid_learner))
        return self._params[learner]

    # The private function _get_params delivers the single treatment, single (cross-fitting) sample subselection.
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.

    def _get_params(self, learner):
        return self._params[learner][self._dml_data.d_cols[self._i_treat]][self._i_rep]

    @property
    def smpls(self):
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            raise ValueError('sample splitting not specified\nEither draw samples via .draw_sample splitting()' +
                             'or set external samples via .set_sample_splitting().')
        return self._smpls

    @smpls.setter
    def smpls(self, value):
        # TODO add checks of dimensions vs properties
        self._smpls = value

    @property
    def psi(self):
        """
        Values of the score function :math:`\\psi(W; \\theta, \\eta) = \\psi_a(W; \\eta) \\theta + \\psi_b(W; \\eta)`
        after calling :meth:`fit`.
        """
        return self._psi

    @property
    def psi_a(self):
        """
        Values of the score function component :math:`\\psi_a(W; \\eta)` after calling :meth:`fit`.
        """
        return self._psi_a

    @property
    def psi_b(self):
        """
        Values of the score function component :math:`\\psi_b(W; \\eta)` after calling :meth:`fit`.
        """
        return self._psi_b

    @property
    def coef(self):
        """
        Estimates for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._coef

    @coef.setter
    def coef(self, value):
        self._coef = value

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._se

    @se.setter
    def se(self, value):
        self._se = value

    @property
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit`.
        """
        t_stat = self.coef / self.se
        return t_stat

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) after calling :meth:`fit`.
        """
        pval = 2 * norm.cdf(-np.abs(self.t_stat))
        return pval

    @property
    def boot_coef(self):
        """
        Bootstrapped coefficients for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`.
        """
        return self._boot_coef

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`.
        """
        return self._boot_t_stat

    @property
    def all_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`.
        """
        return self._all_coef

    @property
    def all_se(self):
        """
        Standard errors of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`.
        """
        return self._all_se

    @property
    def all_dml1_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` x ``n_folds`` different folds after calling :meth:`fit` with ``dml_procedure='dml1'``.
        """
        return self._all_dml1_coef

    @property
    def all_dml1_se(self):
        """
        Standard errors of the causal parameter(s) for the ``n_rep`` x ``n_folds`` different folds after calling :meth:`fit` with ``dml_procedure='dml1'`` and ``se_reestimate=False``.
        """
        return self._all_dml1_se

    @property
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        col_names = ['coef', 'std err', 't', 'P>|t|']
        if self._dml_data.d_cols is None:
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = np.transpose(np.vstack(
                [self.coef, self.se,
                 self.t_stat, self.pval]))
            df_summary = pd.DataFrame(summary_stats,
                                      columns=col_names,
                                      index=self._dml_data.d_cols)
            ci = self.confint()
            df_summary = df_summary.join(ci)
        return df_summary

    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection.
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.

    @property
    def __smpls(self):
        return self.smpls[self._i_rep]

    @property
    def __psi(self):
        return self._psi[:, self._i_rep, self._i_treat]

    @__psi.setter
    def __psi(self, value):
        self._psi[:, self._i_rep, self._i_treat] = value

    @property
    def __psi_a(self):
        return self._psi_a[:, self._i_rep, self._i_treat]

    @__psi_a.setter
    def __psi_a(self, value):
        self._psi_a[:, self._i_rep, self._i_treat] = value

    @property
    def __psi_b(self):
        return self._psi_b[:, self._i_rep, self._i_treat]

    @__psi_b.setter
    def __psi_b(self, value):
        self._psi_b[:, self._i_rep, self._i_treat] = value

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
    def __boot_t_stat(self):
        ind_start = self._i_rep * self.n_rep_boot
        ind_end = (self._i_rep + 1) * self.n_rep_boot
        return self._boot_t_stat[self._i_treat, ind_start:ind_end]

    @__boot_t_stat.setter
    def __boot_t_stat(self, value):
        ind_start = self._i_rep * self.n_rep_boot
        ind_end = (self._i_rep + 1) * self.n_rep_boot
        self._boot_t_stat[self._i_treat, ind_start:ind_end] = value

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

    @property
    def __all_dml1_coef(self):
        assert self.dml_procedure == 'dml1', 'only available for dml_procedure `dml1`'
        return self._all_dml1_coef[self._i_treat, self._i_rep, :]

    @__all_dml1_coef.setter
    def __all_dml1_coef(self, value):
        assert self.dml_procedure == 'dml1', 'only available for dml_procedure `dml1`'
        self._all_dml1_coef[self._i_treat, self._i_rep, :] = value

    @property
    def __all_dml1_se(self):
        assert self.dml_procedure == 'dml1', 'only available for dml_procedure `dml1`'
        return self._all_dml1_se[self._i_treat, self._i_rep, :]

    @__all_dml1_se.setter
    def __all_dml1_se(self, value):
        assert self.dml_procedure == 'dml1', 'only available for dml_procedure `dml1`'
        self._all_dml1_se[self._i_treat, self._i_rep, :] = value

    def fit(self, se_reestimate=False, n_jobs_cv=None, keep_scores=True):
        """
        Estimate DoubleML models.

        Parameters
        ----------
        se_reestimate : bool
            Indicates whether standard errors should be reestimated (only relevant for ``dml_procedure='dml1'``.
            Default is ``False``.

        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        keep_scores : bool
            Indicates whether the score function evaluations should be stored in ``psi``, ``psi_a`` and ``psi_b``.
            Default is ``True``.

        Returns
        -------
        self : object
        """

        if n_jobs_cv is not None:
            if not isinstance(n_jobs_cv, int):
                raise TypeError('The number of CPUs used to fit the learners must be of int type. '
                                f'{str(n_jobs_cv)} of type {str(type(n_jobs_cv))} was passed.')

        if not isinstance(keep_scores, bool):
            raise TypeError('keep_scores must be True or False. '
                            f'got {str(keep_scores)}')

        if not self.apply_cross_fitting:
            if se_reestimate:
                # redirect to se_reestimate = False; se_reestimate is of no relevance without cross-fitting
                se_reestimate = False

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                # if self._ml_nuiscance_params is not None:
                #    self._set_ml_nuisance_params(self._ml_nuiscance_params[i_rep][i_d])

                # this step could be skipped for the single treatment variable case
                if self._dml_data.n_treat > 1:
                    self._dml_data._set_x_d(self._dml_data.d_cols[i_d])

                # ml estimation of nuisance models and computation of score elements
                self.__psi_a, self.__psi_b = self._ml_nuisance_and_score_elements(self.__smpls, n_jobs_cv)

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

        return self

    def bootstrap(self, method='normal', n_boot_rep=500):
        """
        Bootstrap for DoubleML models.

        Parameters
        ----------
        method : str
            A str (``'Bayes'``, ``'normal'`` or ``'wild'``) specifying the bootstrap method.
            Default is ``'normal'``

        n_boot_rep : int
            The number of bootstrap replications.

        Returns
        -------
        self : object
        """
        if (not hasattr(self, 'coef')) or (self.coef is None):
            raise ValueError('apply fit() before bootstrap()')

        dml_procedure = self.dml_procedure

        self._initialize_boot_arrays(n_boot_rep)

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                self.__boot_coef, self.__boot_t_stat = self._compute_bootstrap(method, n_boot_rep)

        return self

    def confint(self, joint=False, level=0.95):
        """
        Confidence intervals for DoubleML models.

        Parameters
        ----------
        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``False``

        level : float
            The confidence level.
            Default is ``0.95``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s).
        """
        a = (1 - level)
        ab = np.array([a / 2, 1. - a / 2])
        if joint:
            sim = np.amax(np.abs(self.boot_t_stat), 0)
            hatc = np.quantile(sim, 1 - a)
            ci = np.vstack((self.coef - self.se * hatc, self.coef + self.se * hatc)).T
        else:
            fac = norm.ppf(ab)
            ci = np.vstack((self.coef + self.se * fac[0], self.coef + self.se * fac[1])).T

        df_ci = pd.DataFrame(ci,
                             columns=['{:.1f} %'.format(i * 100) for i in ab],
                             index=self._dml_data.d_cols)
        return df_ci

    def p_adjust(self, method='romano-wolf'):
        """
        Multiple testing adjustment for DoubleML models.

        Parameters
        ----------
        method : str
            A str (``'romano-wolf''``, ``'bonferroni'``, ``'holm'``, etc) specifying the adjustment method.
            In addition to ``'romano-wolf''``, all methods implemented in
            :py:func:`statsmodels.stats.multitest.multipletests` can be applied.
            Default is ``'romano-wolf'``.

        Returns
        -------
        p_val : np.array
            An array of adjusted p-values.
        """
        if (not hasattr(self, 'coef')) or (self.coef is None):
            raise ValueError('apply fit() before p_adjust()')

        if method.lower() in ['rw', 'romano-wolf']:
            if (not hasattr(self, 'boot_coef')) or (self.boot_coef is None):
                raise ValueError(f'apply fit() & bootstrap() before p_adjust("{method}")')

            pinit = np.full_like(self.pval, np.nan)
            p_val_corrected = np.full_like(self.pval, np.nan)

            boot_t_stats = self.boot_t_stat
            t_stat = self.t_stat
            stepdown_ind = np.argsort(t_stat)[::-1]
            ro = np.argsort(stepdown_ind)

            for i_d in range(self._dml_data.n_treat):
                if i_d == 0:
                    sim = np.max(boot_t_stats, axis=0)
                    pinit[i_d] = np.minimum(1, np.mean(sim >= np.abs(t_stat[stepdown_ind][i_d])))
                else:
                    sim = np.max(np.delete(boot_t_stats, stepdown_ind[:i_d], axis=0),
                                 axis=0)
                    pinit[i_d] = np.minimum(1, np.mean(sim >= np.abs(t_stat[stepdown_ind][i_d])))

            for i_d in range(self._dml_data.n_treat):
                if i_d == 0:
                    p_val_corrected[i_d] = pinit[i_d]
                else:
                    p_val_corrected[i_d] = np.maximum(pinit[i_d], p_val_corrected[i_d - 1])

            p_val = p_val_corrected[ro]
        else:
            _, p_val, _, _ = multipletests(self.pval, method=method)

        p_val = pd.DataFrame(np.vstack((self.coef, p_val)).T,
                             columns=['coef', 'pval'],
                             index=self._dml_data.d_cols)

        return p_val

    def tune(self,
             param_grids,
             tune_on_folds=False,
             scoring_methods=None,  # if None the estimator's score method is used
             n_folds_tune=5,
             search_mode='grid_search',
             n_iter_randomized_search=100,
             n_jobs_cv=None,
             set_as_params=True,
             return_tune_res=False):
        """
        Hyperparameter-tuning for DoubleML models.

        The hyperparameter-tuning is performed using either an exhaustive search over specified parameter values
        implemented in :class:`sklearn.model_selection.GridSearchCV` or via a randomized search implemented in
        :class:`sklearn.model_selection.RandomizedSearchCV`.

        Parameters
        ----------
        param_grids : dict
            A dict with a parameter grid for each nuisance model / learner (see attribute ``learner_names``).

        tune_on_folds : bool
            Indicates whether the tuning should be done fold-specific or globally.
            Default is ``False``.

        scoring_methods : None or dict
            The scoring method used to evaluate the predictions. The scoring method must be set per nuisance model via
            a dict (see attribute ``learner_names`` for the keys).
            If None, the estimator’s score method is used.
            Default is ``None``.

        n_folds_tune : int
            Number of folds used for tuning.
            Default is ``5``.

        search_mode : str
            A str (``'grid_search'`` or ``'randomized_search'``) specifying whether hyperparameters are optimized via
            :class:`sklearn.model_selection.GridSearchCV` or :class:`sklearn.model_selection.RandomizedSearchCV`.
            Default is ``'grid_search'``.

        n_iter_randomized_search : int
            If ``search_mode == 'randomized_search'``. The number of parameter settings that are sampled.
            Default is ``100``.

        n_jobs_cv : None or int
            The number of CPUs to use to tune the learners. ``None`` means ``1``.
            Default is ``None``.

        set_as_params : bool
            Indicates whether the hyperparameters should be set in order to be used when :meth:`fit` is called.
            Default is ``True``.

        return_tune_res : bool
            Indicates whether detailed tuning results should be returned.
            Default is ``False``.

        Returns
        -------
        self : object
            Returned if ``return_tune_res`` is ``False``.

        tune_res: list
            A list containing detailed tuning results and the proposed hyperparameters.
            Returned if ``return_tune_res`` is ``False``.
        """

        # check param_grids input
        if not isinstance(param_grids, dict) | (not all(k in param_grids for k in self.learner_names)):
            raise ValueError('invalid param_grids ' + str(param_grids) +
                             '\n param_grids must be a dictionary with keys ' + ' and '.join(self.learner_names))

        if scoring_methods is not None:
            if not isinstance(scoring_methods, dict) | (not all(k in self.learner_names for k in scoring_methods)):
                raise ValueError('invalid scoring_methods ' + str(scoring_methods) +
                                 '\n scoring_methods must be a dictionary.' +
                                 '\n Valid keys are ' + ' or '.join(self.learner_names))

        if tune_on_folds:
            tuning_res = [[None] * self.n_rep] * self._dml_data.n_treat
        else:
            tuning_res = [None] * self._dml_data.n_treat

        for i_d in range(self._dml_data.n_treat):
            self._i_treat = i_d
            # this step could be skipped for the single treatment variable case
            if self._dml_data.n_treat > 1:
                self._dml_data._set_x_d(self._dml_data.d_cols[i_d])

            if tune_on_folds:
                nuiscance_params = [None] * self.n_rep
                for i_rep in range(self.n_rep):
                    self._i_rep = i_rep

                    # tune hyperparameters
                    res = self._ml_nuisance_tuning(self.__smpls,
                                                   param_grids, scoring_methods,
                                                   n_folds_tune,
                                                   n_jobs_cv,
                                                   search_mode, n_iter_randomized_search)

                    tuning_res[i_rep][i_d] = res
                    nuiscance_params[i_rep] = res['params']

                if set_as_params:
                    for nuisance_model in nuiscance_params[0].keys():
                        params = [x[nuisance_model] for x in nuiscance_params]
                        self.set_ml_nuisance_params(nuisance_model, self._dml_data.d_cols[i_d], params)

            else:
                smpls = [(np.arange(self._dml_data.n_obs), np.arange(self._dml_data.n_obs))]
                # tune hyperparameters
                res = self._ml_nuisance_tuning(smpls,
                                               param_grids, scoring_methods,
                                               n_folds_tune,
                                               n_jobs_cv,
                                               search_mode, n_iter_randomized_search)
                tuning_res[i_d] = res

                if set_as_params:
                    for nuisance_model in res['params'].keys():
                        params = res['params'][nuisance_model]
                        self.set_ml_nuisance_params(nuisance_model, self._dml_data.d_cols[i_d], params[0])

        if return_tune_res:
            return tuning_res
        else:
            return self

    def set_ml_nuisance_params(self, learner, treat_var, params):
        """
        Set hyperparameters for the nuisance models of DoubleML models.

        Parameters
        ----------
        learner : str
            The nuisance model / learner (see attribute ``params_names``).

        treat_var : str
            The treatment variable (hyperparameters can be set treatment-variable specific).

        params : dict or list
            A dict with estimator parameters (used for all folds) or a nested list with fold specific parameters. The
            outer list needs to be of length ``n_rep`` and the inner list of length ``n_folds``.

        Returns
        -------
        self : object
        """
        valid_learner = self.params_names
        if learner not in valid_learner:
            raise ValueError('invalid nuisance learner ' + learner +
                             '\n valid nuisance learner ' + ' or '.join(valid_learner))

        if treat_var not in self._dml_data.d_cols:
            raise ValueError('invalid treatment variable' + treat_var +
                             '\n valid treatment variable ' + ' or '.join(self._dml_data.d_cols))

        if isinstance(params, dict):
            if self.apply_cross_fitting:
                all_params = [[params] * self.n_folds] * self.n_rep
            else:
                all_params = [[params] * 1] * self.n_rep
        else:
            assert len(params) == self.n_rep
            if self.apply_cross_fitting:
                assert np.all(np.array([len(x) for x in params]) == self.n_folds)
            else:
                assert np.all(np.array([len(x) for x in params]) == 1)
            all_params = params

        self._params[learner][treat_var] = all_params

        return self

    @abstractmethod
    def _initialize_ml_nuisance_params(self, params):
        pass

    @abstractmethod
    def _check_score(self, score):
        pass

    @abstractmethod
    def _check_data(self, obj_dml_data):
        pass

    @abstractmethod
    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        pass

    @abstractmethod
    def _ml_nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                            search_mode, n_iter_randomized_search):
        pass

    @staticmethod
    def _check_learner(learner, learner_name, classifier=False):
        err_msg_prefix = f'invalid learner provided for {learner_name}: '
        warn_msg_prefix = f'learner provided for {learner_name} is probably invalid: '

        assert not isinstance(learner, type), err_msg_prefix + f'provide an instance of a learner instead of a class'

        assert hasattr(learner, 'fit'), err_msg_prefix + f'{str(learner)} has no method .fit()'
        assert hasattr(learner, 'set_params'), err_msg_prefix + f'{str(learner)} has no method .set_params()'
        assert hasattr(learner, 'get_params'), err_msg_prefix + f'{str(learner)} has no method .get_params()'

        if classifier:
            assert hasattr(learner, 'predict_proba'), err_msg_prefix + f'{str(learner)} has no method .predict_proba()'
            if not is_classifier(learner):
                warnings.warn(warn_msg_prefix + f'{str(learner)} is (probably) no classifier')
        else:
            assert hasattr(learner, 'predict'), err_msg_prefix + f'{str(learner)} has no method .predict()'
            if not is_regressor(learner):
                warnings.warn(warn_msg_prefix + f'{str(learner)} is (probably) no regressor')
        return learner

    def _initialize_arrays(self):
        self._psi = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)
        self._psi_a = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)
        self._psi_b = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)

        self._coef = np.full(self._dml_data.n_treat, np.nan)
        self._se = np.full(self._dml_data.n_treat, np.nan)

        self._all_coef = np.full((self._dml_data.n_treat, self.n_rep), np.nan)
        self._all_se = np.full((self._dml_data.n_treat, self.n_rep), np.nan)

        if self.dml_procedure == 'dml1':
            if self.apply_cross_fitting:
                self._all_dml1_coef = np.full((self._dml_data.n_treat, self.n_rep, self.n_folds), np.nan)
                self._all_dml1_se = np.full((self._dml_data.n_treat, self.n_rep, self.n_folds), np.nan)
            else:
                self._all_dml1_coef = np.full((self._dml_data.n_treat, self.n_rep, 1), np.nan)
                self._all_dml1_se = np.full((self._dml_data.n_treat, self.n_rep, 1), np.nan)

    def _initialize_boot_arrays(self, n_rep):
        self.n_rep_boot = n_rep
        self._boot_coef = np.full((self._dml_data.n_treat, n_rep * self.n_rep), np.nan)
        self._boot_t_stat = np.full((self._dml_data.n_treat, n_rep * self.n_rep), np.nan)

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds``, ``n_rep`` and ``apply_cross_fitting``.

        Returns
        -------
        self : object
        """
        obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                n_rep=self.n_rep,
                                                n_obs=self._dml_data.n_obs,
                                                apply_cross_fitting=self.apply_cross_fitting)
        self.smpls = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self, all_smpls):
        """
        Set the sample splitting for DoubleML models.

        The  attributes ``n_folds`` and ``n_rep`` are derived from the provided partition.

        Parameters
        ----------
        all_smpls : list
            A nested list of train and test sets.
            The outer list needs to provide an entry per repeated sample splitting (length of list is set as ``n_rep``).
            The inner list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as ``n_folds``).

        Returns
        -------
        self : object
        """
        # TODO add an example to the documentation (maybe with only 5 observations)
        # TODO warn if n_rep or n_folds is overwritten with different number induced by the transferred external samples?
        # TODO check whether the provided samples are a partition --> set apply_cross_fitting accordingly
        self.n_rep = len(all_smpls)
        n_folds_each_smpl = np.array([len(smpl) for smpl in all_smpls])
        assert np.all(n_folds_each_smpl == n_folds_each_smpl[0]), 'Different number of folds for repeated cross-fitting'
        self.n_folds = n_folds_each_smpl[0]
        self.smpls = all_smpls
        self._initialize_arrays()
        self._initialize_ml_nuisance_params()

        return self

    def _est_causal_pars(self):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls

        if dml_procedure == 'dml1':
            # Note that len(smpls) is only not equal to self.n_folds if self.apply_cross_fitting = False
            thetas = np.zeros(len(smpls))
            for idx, (train_index, test_index) in enumerate(smpls):
                thetas[idx] = self._orth_est(test_index)
            theta_hat = np.mean(thetas)
            coef = theta_hat

            self.__all_dml1_coef = thetas

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
                # Note that len(smpls) is only not equal to self.n_folds if self.apply_cross_fitting = False
                variances = np.zeros(len(smpls))
                for idx, (train_index, test_index) in enumerate(smpls):
                    variances[idx] = self._var_est(test_index)
                se = np.sqrt(np.mean(variances))

                self.__all_dml1_se = np.sqrt(variances)

        elif dml_procedure == 'dml2':
            se = np.sqrt(self._var_est())

        else:
            raise ValueError('invalid dml_procedure')

        return se

    def _agg_cross_fit(self):
        # aggregate parameters from the repeated cross-fitting
        # don't use the getter (always for one treatment variable and one sample), but the private variable
        self.coef = np.median(self._all_coef, 1)

        # TODO: In the documentation of standard errors we need to cleary state what we return here, i.e.,
        # the asymptotic variance sigma_hat/N and not sigma_hat (which sometimes is also called the asympt var)!
        if self.apply_cross_fitting:
            n_obs = self._dml_data.n_obs
        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            smpls = self.__smpls
            test_index = smpls[0][1]
            n_obs = len(test_index)
        xx = np.tile(self.coef.reshape(-1, 1), self.n_rep)
        self.se = np.sqrt(np.divide(np.median(np.multiply(np.power(self._all_se, 2), n_obs) +
                                              np.power(self._all_coef - xx, 2), 1), n_obs))

    def _compute_bootstrap(self, method, n_boot_rep):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls
        if self.apply_cross_fitting:
            n_obs = self._dml_data.n_obs
        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            test_index = smpls[0][1]
            n_obs = len(test_index)

        if method == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=(n_boot_rep, n_obs)) - 1.
        elif method == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=(n_boot_rep, n_obs))
        elif method == 'wild':
            xx = np.random.normal(loc=0.0, scale=1.0, size=(n_boot_rep, n_obs))
            yy = np.random.normal(loc=0.0, scale=1.0, size=(n_boot_rep, n_obs))
            weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
        else:
            raise ValueError('invalid boot method')

        if self.apply_cross_fitting:
            if dml_procedure == 'dml1':
                boot_coefs = np.full((n_boot_rep, self.n_folds), np.nan)
                boot_t_stats = np.full((n_boot_rep, self.n_folds), np.nan)
                for idx, (_, test_index) in enumerate(smpls):
                    J = np.mean(self.__psi_a[test_index])
                    boot_coefs[:, idx] = np.matmul(weights[:, test_index], self.__psi[test_index]) / (
                            len(test_index) * J)
                    boot_t_stats[:, idx] = np.matmul(weights[:, test_index], self.__psi[test_index]) / (
                            len(test_index) * self.__all_se * J)
                boot_coef = np.mean(boot_coefs, axis=1)
                boot_t_stat = np.mean(boot_t_stats, axis=1)

            elif dml_procedure == 'dml2':
                J = np.mean(self.__psi_a)
                boot_coef = np.matmul(weights, self.__psi) / (self._dml_data.n_obs * J)
                boot_t_stat = np.matmul(weights, self.__psi) / (self._dml_data.n_obs * self.__all_se * J)

            else:
                raise ValueError('invalid dml_procedure')
        else:
            J = np.mean(self.__psi_a[test_index])
            boot_coef = np.matmul(weights, self.__psi[test_index]) / (len(test_index) * J)
            boot_t_stat = np.matmul(weights, self.__psi[test_index]) / (len(test_index) * self.__all_se * J)

        return boot_coef, boot_t_stat

    def _var_est(self, inds=None):
        """
        Estimate the standard errors of the structural parameter
        """
        psi_a = self.__psi_a
        psi = self.__psi

        if inds is not None:
            psi_a = psi_a[inds]
            psi = psi[inds]

        # TODO: In the documentation of standard errors we need to cleary state what we return here, i.e.,
        # the asymptotic variance sigma_hat/N and not sigma_hat (which sometimes is also called the asympt var)!
        if self.apply_cross_fitting:
            n_obs = self._dml_data.n_obs
        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            smpls = self.__smpls
            test_index = smpls[0][1]
            n_obs = len(test_index)
        J = np.mean(psi_a)
        sigma2_hat = 1 / n_obs * np.mean(np.power(psi, 2)) / np.power(J, 2)

        return sigma2_hat

    def _orth_est(self, inds=None):
        """
        Estimate the structural parameter
        """
        psi_a = self.__psi_a
        psi_b = self.__psi_b

        if inds is not None:
            psi_a = psi_a[inds]
            psi_b = psi_b[inds]

        theta = -np.mean(psi_b) / np.mean(psi_a)

        return theta

    def _compute_score(self):
        self.__psi = self.__psi_a * self.__all_coef + self.__psi_b

    def _clean_scores(self):
        del self._psi
        del self._psi_a
        del self._psi_b
