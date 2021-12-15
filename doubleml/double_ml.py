import numpy as np
import pandas as pd
import warnings

from sklearn.base import is_regressor, is_classifier

from scipy.stats import norm

from statsmodels.stats.multitest import multipletests

from abc import ABC, abstractmethod

from .double_ml_data import DoubleMLData, DoubleMLClusterData
from ._utils_resampling import DoubleMLResampling, DoubleMLClusterResampling
from ._utils import _check_is_partition, _check_all_smpls, _check_smpl_split, _check_smpl_split_tpl, _draw_weights


class DoubleML(ABC):
    """Double Machine Learning.
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
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            if obj_dml_data.n_cluster_vars > 2:
                raise NotImplementedError('Multi-way (n_ways > 2) clustering not yet implemented.')
            self._is_cluster_data = True
        self._dml_data = obj_dml_data

        # initialize learners and parameters which are set model specific
        self._learner = None
        self._params = None

        # initialize predictions to None which are only stored if method fit is called with store_predictions=True
        self._predictions = None

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
                            f'Got {str(apply_cross_fitting)}.')
        if not isinstance(draw_sample_splitting, bool):
            raise TypeError('draw_sample_splitting must be True or False. '
                            f'Got {str(draw_sample_splitting)}.')

        # set resampling specifications
        if self._is_cluster_data:
            if (n_folds == 1) | (not apply_cross_fitting):
                raise NotImplementedError('No cross-fitting (`apply_cross_fitting = False`) '
                                          'is not yet implemented with clustering.')
            self._n_folds_per_cluster = n_folds
            self._n_folds = n_folds ** self._dml_data.n_cluster_vars
        else:
            self._n_folds = n_folds
        self._n_rep = n_rep
        self._apply_cross_fitting = apply_cross_fitting

        # check and set dml_procedure and score
        if (not isinstance(dml_procedure, str)) | (dml_procedure not in ['dml1', 'dml2']):
            raise ValueError('dml_procedure must be "dml1" or "dml2". '
                             f'Got {str(dml_procedure)}.')
        self._dml_procedure = dml_procedure
        self._score = score

        if (self.n_folds == 1) & self.apply_cross_fitting:
            warnings.warn('apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.')
            self._apply_cross_fitting = False

        if not self.apply_cross_fitting:
            assert self.n_folds <= 2, 'Estimation without cross-fitting not supported for n_folds > 2.'
            if self.dml_procedure == 'dml2':
                # redirect to dml1 which works out-of-the-box; dml_procedure is of no relevance without cross-fitting
                self._dml_procedure = 'dml1'

        # perform sample splitting
        self._smpls = None
        self._smpls_cluster = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

        # initialize arrays according to obj_dml_data and the resampling settings
        self._psi, self._psi_a, self._psi_b,\
            self._coef, self._se, self._all_coef, self._all_se, self._all_dml1_coef = self._initialize_arrays()

        # also initialize bootstrap arrays with the default number of bootstrap replications
        self._n_rep_boot, self._boot_coef, self._boot_t_stat = self._initialize_boot_arrays(n_rep_boot=500)

        # initialize instance attributes which are later used for iterating
        self._i_rep = None
        self._i_treat = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        if self._is_cluster_data:
            cluster_info = f'Cluster variable(s): {self._dml_data.cluster_cols}\n'
        else:
            cluster_info = ''
        data_info = f'Outcome variable: {self._dml_data.y_col}\n' \
                    f'Treatment variable(s): {self._dml_data.d_cols}\n' \
                    f'Covariates: {self._dml_data.x_cols}\n' \
                    f'Instrument variable(s): {self._dml_data.z_cols}\n' \
                    + cluster_info +\
                    f'No. Observations: {self._dml_data.n_obs}\n'
        score_info = f'Score function: {str(self.score)}\n' \
                     f'DML algorithm: {self.dml_procedure}\n'
        learner_info = ''
        for key, value in self.learner.items():
            learner_info += f'Learner {key}: {str(value)}\n'
        if self._is_cluster_data:
            resampling_info = f'No. folds per cluster: {self._n_folds_per_cluster}\n' \
                              f'No. folds: {self.n_folds}\n' \
                              f'No. repeated sample splits: {self.n_rep}\n' \
                              f'Apply cross-fitting: {self.apply_cross_fitting}\n'
        else:
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
    def n_folds(self):
        """
        Number of folds.
        """
        return self._n_folds

    @property
    def n_rep(self):
        """
        Number of repetitions for the sample splitting.
        """
        return self._n_rep

    @property
    def apply_cross_fitting(self):
        """
        Indicates whether cross-fitting should be applied.
        """
        return self._apply_cross_fitting

    @property
    def dml_procedure(self):
        """
        The double machine learning algorithm.
        """
        return self._dml_procedure

    @property
    def n_rep_boot(self):
        """
        The number of bootstrap replications.
        """
        return self._n_rep_boot

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

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

    @property
    def predictions(self):
        """
        The predictions of the nuisance models.
        """
        return self._predictions

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
            raise ValueError('Invalid nuisance learner ' + str(learner) + '. ' +
                             'Valid nuisance learner ' + ' or '.join(valid_learner) + '.')
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
            if self._is_cluster_data:
                err_msg = 'Sample splitting not specified. Draw samples via .draw_sample splitting().'
            else:
                err_msg = ('Sample splitting not specified. Either draw samples via .draw_sample splitting() ' +
                           'or set external samples via .set_sample_splitting().')
            raise ValueError(err_msg)
        return self._smpls

    @property
    def smpls_cluster(self):
        """
        The partition of clusters used for cross-fitting.
        """
        if self._is_cluster_data:
            if self._smpls_cluster is None:
                raise ValueError('Sample splitting not specified. Draw samples via .draw_sample splitting().')
        return self._smpls_cluster

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
        Estimates of the causal parameter(s) for the ``n_rep`` x ``n_folds`` different folds after calling :meth:`fit`
        with ``dml_procedure='dml1'``.
        """
        return self._all_dml1_coef

    @property
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        col_names = ['coef', 'std err', 't', 'P>|t|']
        if np.isnan(self.coef).all():
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
        return self._smpls[self._i_rep]

    @property
    def __smpls_cluster(self):
        return self._smpls_cluster[self._i_rep]

    @property
    def __psi(self):
        return self._psi[:, self._i_rep, self._i_treat]

    @property
    def __psi_a(self):
        return self._psi_a[:, self._i_rep, self._i_treat]

    @property
    def __psi_b(self):
        return self._psi_b[:, self._i_rep, self._i_treat]

    @property
    def __all_coef(self):
        return self._all_coef[self._i_treat, self._i_rep]

    @property
    def __all_se(self):
        return self._all_se[self._i_treat, self._i_rep]

    def fit(self, n_jobs_cv=None, keep_scores=True, store_predictions=False):
        """
        Estimate DoubleML models.

        Parameters
        ----------
        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        keep_scores : bool
            Indicates whether the score function evaluations should be stored in ``psi``, ``psi_a`` and ``psi_b``.
            Default is ``True``.

        store_predictions : bool
            Indicates whether the predictions for the nuisance functions should be be stored in ``predictions``.
            Default is ``False``.

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
                            f'Got {str(keep_scores)}.')

        if not isinstance(store_predictions, bool):
            raise TypeError('store_predictions must be True or False. '
                            f'Got {str(store_predictions)}.')

        if store_predictions:
            self._initialize_predictions()

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                # this step could be skipped for the single treatment variable case
                if self._dml_data.n_treat > 1:
                    self._dml_data.set_x_d(self._dml_data.d_cols[i_d])

                # ml estimation of nuisance models and computation of score elements
                self._psi_a[:, self._i_rep, self._i_treat], self._psi_b[:, self._i_rep, self._i_treat], preds =\
                    self._nuisance_est(self.__smpls, n_jobs_cv)

                if store_predictions:
                    self._store_predictions(preds)

                # estimate the causal parameter
                self._all_coef[self._i_treat, self._i_rep] = self._est_causal_pars()

                # compute score (depends on estimated causal parameter)
                self._psi[:, self._i_rep, self._i_treat] = self._compute_score()

                # compute standard errors for causal parameter
                self._all_se[self._i_treat, self._i_rep] = self._se_causal_pars()

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self._agg_cross_fit()

        if not keep_scores:
            self._clean_scores()

        return self

    def bootstrap(self, method='normal', n_rep_boot=500):
        """
        Multiplier bootstrap for DoubleML models.

        Parameters
        ----------
        method : str
            A str (``'Bayes'``, ``'normal'`` or ``'wild'``) specifying the multiplier bootstrap method.
            Default is ``'normal'``

        n_rep_boot : int
            The number of bootstrap replications.

        Returns
        -------
        self : object
        """
        if np.isnan(self.coef).all():
            raise ValueError('Apply fit() before bootstrap().')

        if (not isinstance(method, str)) | (method not in ['Bayes', 'normal', 'wild']):
            raise ValueError('Method must be "Bayes", "normal" or "wild". '
                             f'Got {str(method)}.')

        if not isinstance(n_rep_boot, int):
            raise TypeError('The number of bootstrap replications must be of int type. '
                            f'{str(n_rep_boot)} of type {str(type(n_rep_boot))} was passed.')
        if n_rep_boot < 1:
            raise ValueError('The number of bootstrap replications must be positive. '
                             f'{str(n_rep_boot)} was passed.')
        if self._is_cluster_data:
            raise NotImplementedError('bootstrap not yet implemented with clustering.')

        self._n_rep_boot, self._boot_coef, self._boot_t_stat = self._initialize_boot_arrays(n_rep_boot)

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            # draw weights for the bootstrap
            if self.apply_cross_fitting:
                n_obs = self._dml_data.n_obs
            else:
                # be prepared for the case of test sets of different size in repeated no-cross-fitting
                smpls = self.__smpls
                test_index = smpls[0][1]
                n_obs = len(test_index)
            weights = _draw_weights(method, n_rep_boot, n_obs)

            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d
                i_start = self._i_rep * self.n_rep_boot
                i_end = (self._i_rep + 1) * self.n_rep_boot
                self._boot_coef[self._i_treat, i_start:i_end], self._boot_t_stat[self._i_treat, i_start:i_end] =\
                    self._compute_bootstrap(weights)

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

        if not isinstance(joint, bool):
            raise TypeError('joint must be True or False. '
                            f'Got {str(joint)}.')

        if not isinstance(level, float):
            raise TypeError('The confidence level must be of float type. '
                            f'{str(level)} of type {str(type(level))} was passed.')
        if (level <= 0) | (level >= 1):
            raise ValueError('The confidence level must be in (0,1). '
                             f'{str(level)} was passed.')

        a = (1 - level)
        ab = np.array([a / 2, 1. - a / 2])
        if joint:
            if np.isnan(self.boot_coef).all():
                raise ValueError('Apply fit() & bootstrap() before confint(joint=True).')
            sim = np.amax(np.abs(self.boot_t_stat), 0)
            hatc = np.quantile(sim, 1 - a)
            ci = np.vstack((self.coef - self.se * hatc, self.coef + self.se * hatc)).T
        else:
            if np.isnan(self.coef).all():
                raise ValueError('Apply fit() before confint().')
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
        p_val : pd.DataFrame
            A data frame with adjusted p-values.
        """
        if np.isnan(self.coef).all():
            raise ValueError('Apply fit() before p_adjust().')

        if not isinstance(method, str):
            raise TypeError('The p_adjust method must be of str type. '
                            f'{str(method)} of type {str(type(method))} was passed.')

        if method.lower() in ['rw', 'romano-wolf']:
            if np.isnan(self.boot_coef).all():
                raise ValueError(f'Apply fit() & bootstrap() before p_adjust("{method}").')

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
            Returned if ``return_tune_res`` is ``True``.
        """

        if (not isinstance(param_grids, dict)) | (not all(k in param_grids for k in self.learner_names)):
            raise ValueError('Invalid param_grids ' + str(param_grids) + '. '
                             'param_grids must be a dictionary with keys ' + ' and '.join(self.learner_names) + '.')

        if scoring_methods is not None:
            if (not isinstance(scoring_methods, dict)) | (not all(k in self.learner_names for k in scoring_methods)):
                raise ValueError('Invalid scoring_methods ' + str(scoring_methods) + '. ' +
                                 'scoring_methods must be a dictionary. ' +
                                 'Valid keys are ' + ' and '.join(self.learner_names) + '.')
            if not all(k in scoring_methods for k in self.learner_names):
                # if there are learners for which no scoring_method was set, we fall back to None, i.e., default scoring
                for learner in self.learner_names:
                    if learner not in scoring_methods:
                        scoring_methods[learner] = None

        if not isinstance(tune_on_folds, bool):
            raise TypeError('tune_on_folds must be True or False. '
                            f'Got {str(tune_on_folds)}.')

        if not isinstance(n_folds_tune, int):
            raise TypeError('The number of folds used for tuning must be of int type. '
                            f'{str(n_folds_tune)} of type {str(type(n_folds_tune))} was passed.')
        if n_folds_tune < 2:
            raise ValueError('The number of folds used for tuning must be at least two. '
                             f'{str(n_folds_tune)} was passed.')

        if (not isinstance(search_mode, str)) | (search_mode not in ['grid_search', 'randomized_search']):
            raise ValueError('search_mode must be "grid_search" or "randomized_search". '
                             f'Got {str(search_mode)}.')

        if not isinstance(n_iter_randomized_search, int):
            raise TypeError('The number of parameter settings sampled for the randomized search must be of int type. '
                            f'{str(n_iter_randomized_search)} of type '
                            f'{str(type(n_iter_randomized_search))} was passed.')
        if n_iter_randomized_search < 2:
            raise ValueError('The number of parameter settings sampled for the randomized search must be at least two. '
                             f'{str(n_iter_randomized_search)} was passed.')

        if n_jobs_cv is not None:
            if not isinstance(n_jobs_cv, int):
                raise TypeError('The number of CPUs used to fit the learners must be of int type. '
                                f'{str(n_jobs_cv)} of type {str(type(n_jobs_cv))} was passed.')

        if not isinstance(set_as_params, bool):
            raise TypeError('set_as_params must be True or False. '
                            f'Got {str(set_as_params)}.')

        if not isinstance(return_tune_res, bool):
            raise TypeError('return_tune_res must be True or False. '
                            f'Got {str(return_tune_res)}.')

        if tune_on_folds:
            tuning_res = [[None] * self.n_rep] * self._dml_data.n_treat
        else:
            tuning_res = [None] * self._dml_data.n_treat

        for i_d in range(self._dml_data.n_treat):
            self._i_treat = i_d
            # this step could be skipped for the single treatment variable case
            if self._dml_data.n_treat > 1:
                self._dml_data.set_x_d(self._dml_data.d_cols[i_d])

            if tune_on_folds:
                nuisance_params = list()
                for i_rep in range(self.n_rep):
                    self._i_rep = i_rep

                    # tune hyperparameters
                    res = self._nuisance_tuning(self.__smpls,
                                                param_grids, scoring_methods,
                                                n_folds_tune,
                                                n_jobs_cv,
                                                search_mode, n_iter_randomized_search)

                    tuning_res[i_rep][i_d] = res
                    nuisance_params.append(res['params'])

                if set_as_params:
                    for nuisance_model in nuisance_params[0].keys():
                        params = [x[nuisance_model] for x in nuisance_params]
                        self.set_ml_nuisance_params(nuisance_model, self._dml_data.d_cols[i_d], params)

            else:
                smpls = [(np.arange(self._dml_data.n_obs), np.arange(self._dml_data.n_obs))]
                # tune hyperparameters
                res = self._nuisance_tuning(smpls,
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
            raise ValueError('Invalid nuisance learner ' + learner + '. ' +
                             'Valid nuisance learner ' + ' or '.join(valid_learner) + '.')

        if treat_var not in self._dml_data.d_cols:
            raise ValueError('Invalid treatment variable ' + treat_var + '. ' +
                             'Valid treatment variable ' + ' or '.join(self._dml_data.d_cols) + '.')

        if params is None:
            all_params = [None] * self.n_rep
        elif isinstance(params, dict):
            if self.apply_cross_fitting:
                all_params = [[params] * self.n_folds] * self.n_rep
            else:
                all_params = [[params] * 1] * self.n_rep
        else:
            # ToDo: Add meaningful error message for asserts and corresponding uni tests
            assert len(params) == self.n_rep
            if self.apply_cross_fitting:
                assert np.all(np.array([len(x) for x in params]) == self.n_folds)
            else:
                assert np.all(np.array([len(x) for x in params]) == 1)
            all_params = params

        self._params[learner][treat_var] = all_params

        return self

    @abstractmethod
    def _initialize_ml_nuisance_params(self):
        pass

    @abstractmethod
    def _nuisance_est(self, smpls, n_jobs_cv):
        pass

    @abstractmethod
    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        pass

    @staticmethod
    def _check_learner(learner, learner_name, regressor, classifier):
        err_msg_prefix = f'Invalid learner provided for {learner_name}: '
        warn_msg_prefix = f'Learner provided for {learner_name} is probably invalid: '

        if isinstance(learner, type):
            raise TypeError(err_msg_prefix + 'provide an instance of a learner instead of a class.')

        if not hasattr(learner, 'fit'):
            raise TypeError(err_msg_prefix + f'{str(learner)} has no method .fit().')
        if not hasattr(learner, 'set_params'):
            raise TypeError(err_msg_prefix + f'{str(learner)} has no method .set_params().')
        if not hasattr(learner, 'get_params'):
            raise TypeError(err_msg_prefix + f'{str(learner)} has no method .get_params().')

        if regressor & classifier:
            if is_classifier(learner):
                learner_is_classifier = True
            elif is_regressor(learner):
                learner_is_classifier = False
            else:
                warnings.warn(warn_msg_prefix + f'{str(learner)} is (probably) neither a regressor nor a classifier. ' +
                              'Method predict is used for prediction.')
                learner_is_classifier = False
        elif classifier:
            if not is_classifier(learner):
                warnings.warn(warn_msg_prefix + f'{str(learner)} is (probably) no classifier.')
            learner_is_classifier = True
        else:
            assert regressor  # classifier, regressor or both must be True
            if not is_regressor(learner):
                warnings.warn(warn_msg_prefix + f'{str(learner)} is (probably) no regressor.')
            learner_is_classifier = False

        # check existence of the prediction method
        if learner_is_classifier:
            if not hasattr(learner, 'predict_proba'):
                raise TypeError(err_msg_prefix + f'{str(learner)} has no method .predict_proba().')
        else:
            if not hasattr(learner, 'predict'):
                raise TypeError(err_msg_prefix + f'{str(learner)} has no method .predict().')

        return learner_is_classifier

    def _initialize_arrays(self):
        psi = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)
        psi_a = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)
        psi_b = np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)

        coef = np.full(self._dml_data.n_treat, np.nan)
        se = np.full(self._dml_data.n_treat, np.nan)

        all_coef = np.full((self._dml_data.n_treat, self.n_rep), np.nan)
        all_se = np.full((self._dml_data.n_treat, self.n_rep), np.nan)

        if self.dml_procedure == 'dml1':
            if self.apply_cross_fitting:
                all_dml1_coef = np.full((self._dml_data.n_treat, self.n_rep, self.n_folds), np.nan)
            else:
                all_dml1_coef = np.full((self._dml_data.n_treat, self.n_rep, 1), np.nan)
        else:
            all_dml1_coef = None

        return psi, psi_a, psi_b, coef, se, all_coef, all_se, all_dml1_coef

    def _initialize_boot_arrays(self, n_rep_boot):
        boot_coef = np.full((self._dml_data.n_treat, n_rep_boot * self.n_rep), np.nan)
        boot_t_stat = np.full((self._dml_data.n_treat, n_rep_boot * self.n_rep), np.nan)
        return n_rep_boot, boot_coef, boot_t_stat

    def _initialize_predictions(self):
        self._predictions = {learner: np.full((self._dml_data.n_obs, self.n_rep, self._dml_data.n_treat), np.nan)
                             for learner in self.params_names}

    def _store_predictions(self, preds):
        for learner in self.params_names:
            self._predictions[learner][:, self._i_rep, self._i_treat] = preds[learner]

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds``, ``n_rep`` and ``apply_cross_fitting``.

        Returns
        -------
        self : object
        """
        if self._is_cluster_data:
            obj_dml_resampling = DoubleMLClusterResampling(n_folds=self._n_folds_per_cluster,
                                                           n_rep=self.n_rep,
                                                           n_obs=self._dml_data.n_obs,
                                                           apply_cross_fitting=self.apply_cross_fitting,
                                                           n_cluster_vars=self._dml_data.n_cluster_vars,
                                                           cluster_vars=self._dml_data.cluster_vars)
            self._smpls, self._smpls_cluster = obj_dml_resampling.split_samples()
        else:
            obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                    n_rep=self.n_rep,
                                                    n_obs=self._dml_data.n_obs,
                                                    apply_cross_fitting=self.apply_cross_fitting)
            self._smpls = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self, all_smpls):
        """
        Set the sample splitting for DoubleML models.

        The  attributes ``n_folds`` and ``n_rep`` are derived from the provided partition.

        Parameters
        ----------
        all_smpls : list or tuple
            If nested list of lists of tuples:
                The outer list needs to provide an entry per repeated sample splitting (length of list is set as
                ``n_rep``).
                The inner list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as
                ``n_folds``). If tuples for more than one fold are provided, it must form a partition and
                ``apply_cross_fitting`` is set to True. Otherwise ``apply_cross_fitting`` is set to False and
                ``n_folds=2``.
            If list of tuples:
                The list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as
                ``n_folds``). If tuples for more than one fold are provided, it must form a partition and
                ``apply_cross_fitting`` is set to True. Otherwise ``apply_cross_fitting`` is set to False and
                ``n_folds=2``.
                ``n_rep=1`` is always set.
            If tuple:
                Must be a tuple with two elements train_ind and test_ind. No sample splitting is achieved if train_ind
                and test_ind are range(n_rep). Otherwise ``n_folds=2``.
                ``apply_cross_fitting=False`` and ``n_rep=1`` is always set.

        Returns
        -------
        self : object

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from doubleml.datasets import make_plr_CCDDHNR2018
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.base import clone
        >>> np.random.seed(3141)
        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_g = learner
        >>> ml_m = learner
        >>> obj_dml_data = make_plr_CCDDHNR2018(n_obs=10, alpha=0.5)
        >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        >>> # simple sample splitting with two folds and without cross-fitting
        >>> smpls = ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        >>> dml_plr_obj.set_sample_splitting(smpls)
        >>> # sample splitting with two folds and cross-fitting
        >>> smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        >>>          ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])]
        >>> dml_plr_obj.set_sample_splitting(smpls)
        >>> # sample splitting with two folds and repeated cross-fitting with n_rep = 2
        >>> smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        >>>           ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
        >>>          [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
        >>>           ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
        >>> dml_plr_obj.set_sample_splitting(smpls)
        """
        if self._is_cluster_data:
            raise NotImplementedError('Externally setting the sample splitting for DoubleML is '
                                      'not yet implemented with clustering.')
        if isinstance(all_smpls, tuple):
            if not len(all_smpls) == 2:
                raise ValueError('Invalid partition provided. '
                                 'Tuple for train_ind and test_ind must consist of exactly two elements.')
            all_smpls = _check_smpl_split_tpl(all_smpls, self._dml_data.n_obs)
            if (_check_is_partition([all_smpls], self._dml_data.n_obs) &
                    _check_is_partition([(all_smpls[1], all_smpls[0])], self._dml_data.n_obs)):
                self._n_rep = 1
                self._n_folds = 1
                self._apply_cross_fitting = False
                self._smpls = [[all_smpls]]
            else:
                self._n_rep = 1
                self._n_folds = 2
                self._apply_cross_fitting = False
                self._smpls = _check_all_smpls([[all_smpls]], self._dml_data.n_obs, check_intersect=True)
        else:
            if not isinstance(all_smpls, list):
                raise TypeError('all_smpls must be of list or tuple type. '
                                f'{str(all_smpls)} of type {str(type(all_smpls))} was passed.')
            all_tuple = all([isinstance(tpl, tuple) for tpl in all_smpls])
            if all_tuple:
                if not all([len(tpl) == 2 for tpl in all_smpls]):
                    raise ValueError('Invalid partition provided. '
                                     'All tuples for train_ind and test_ind must consist of exactly two elements.')
                self._n_rep = 1
                all_smpls = _check_smpl_split(all_smpls, self._dml_data.n_obs)
                if _check_is_partition(all_smpls, self._dml_data.n_obs):
                    if ((len(all_smpls) == 1) &
                            _check_is_partition([(all_smpls[0][1], all_smpls[0][0])], self._dml_data.n_obs)):
                        self._n_folds = 1
                        self._apply_cross_fitting = False
                        self._smpls = [all_smpls]
                    else:
                        self._n_folds = len(all_smpls)
                        self._apply_cross_fitting = True
                        self._smpls = _check_all_smpls([all_smpls], self._dml_data.n_obs, check_intersect=True)
                else:
                    if not len(all_smpls) == 1:
                        raise ValueError('Invalid partition provided. '
                                         'Tuples for more than one fold provided that don\'t form a partition.')
                    self._n_folds = 2
                    self._apply_cross_fitting = False
                    self._smpls = _check_all_smpls([all_smpls], self._dml_data.n_obs, check_intersect=True)
            else:
                all_list = all([isinstance(smpl, list) for smpl in all_smpls])
                if not all_list:
                    raise ValueError('Invalid partition provided. '
                                     'all_smpls is a list where neither all elements are tuples '
                                     'nor all elements are lists.')
                all_tuple = all([all([isinstance(tpl, tuple) for tpl in smpl]) for smpl in all_smpls])
                if not all_tuple:
                    raise TypeError('For repeated sample splitting all_smpls must be list of lists of tuples.')
                all_pairs = all([all([len(tpl) == 2 for tpl in smpl]) for smpl in all_smpls])
                if not all_pairs:
                    raise ValueError('Invalid partition provided. '
                                     'All tuples for train_ind and test_ind must consist of exactly two elements.')
                n_folds_each_smpl = np.array([len(smpl) for smpl in all_smpls])
                if not np.all(n_folds_each_smpl == n_folds_each_smpl[0]):
                    raise ValueError('Invalid partition provided. '
                                     'Different number of folds for repeated sample splitting.')
                all_smpls = _check_all_smpls(all_smpls, self._dml_data.n_obs)
                smpls_are_partitions = [_check_is_partition(smpl, self._dml_data.n_obs) for smpl in all_smpls]

                if all(smpls_are_partitions):
                    if ((len(all_smpls) == 1) & (len(all_smpls[0]) == 1) &
                            _check_is_partition([(all_smpls[0][0][1], all_smpls[0][0][0])], self._dml_data.n_obs)):
                        self._n_rep = 1
                        self._n_folds = 1
                        self._apply_cross_fitting = False
                        self._smpls = all_smpls
                    else:
                        self._n_rep = len(all_smpls)
                        self._n_folds = n_folds_each_smpl[0]
                        self._apply_cross_fitting = True
                        self._smpls = _check_all_smpls(all_smpls, self._dml_data.n_obs, check_intersect=True)
                else:
                    if not n_folds_each_smpl[0] == 1:
                        raise ValueError('Invalid partition provided. '
                                         'Tuples for more than one fold provided '
                                         'but at least one does not form a partition.')
                    self._n_rep = len(all_smpls)
                    self._n_folds = 2
                    self._apply_cross_fitting = False
                    self._smpls = _check_all_smpls(all_smpls, self._dml_data.n_obs, check_intersect=True)

        self._psi, self._psi_a, self._psi_b, \
            self._coef, self._se, self._all_coef, self._all_se, self._all_dml1_coef = self._initialize_arrays()
        self._initialize_ml_nuisance_params()

        return self

    def _est_causal_pars(self):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls

        if not self._is_cluster_data:
            if dml_procedure == 'dml1':
                # Note that len(smpls) is only not equal to self.n_folds if self.apply_cross_fitting = False
                thetas = np.zeros(len(smpls))
                for idx, (_, test_index) in enumerate(smpls):
                    thetas[idx] = self._orth_est(test_index)
                theta_hat = np.mean(thetas)
                coef = theta_hat

                self._all_dml1_coef[self._i_treat, self._i_rep, :] = thetas
            else:
                assert dml_procedure == 'dml2'
                theta_hat = self._orth_est()
                coef = theta_hat
        else:
            coef = self._orth_est_cluster_data()

        return coef

    def _se_causal_pars(self):
        if not self._is_cluster_data:
            se = np.sqrt(self._var_est())
        else:
            se = np.sqrt(self._var_est_cluster_data())

        return se

    def _agg_cross_fit(self):
        # aggregate parameters from the repeated cross-fitting
        # don't use the getter (always for one treatment variable and one sample), but the private variable
        self.coef = np.median(self._all_coef, 1)

        # TODO: In the documentation of standard errors we need to cleary state what we return here, i.e.,
        #  the asymptotic variance sigma_hat/N and not sigma_hat (which sometimes is also called the asympt var)!
        # TODO: In the edge case of repeated no-cross-fitting, the test sets might have different size and therefore
        #  it would note be valid to always use the same self._var_scaling_factor
        xx = np.tile(self.coef.reshape(-1, 1), self.n_rep)
        self.se = np.sqrt(np.divide(np.median(np.multiply(np.power(self._all_se, 2), self._var_scaling_factor) +
                                              np.power(self._all_coef - xx, 2), 1), self._var_scaling_factor))

    def _est_causal_pars_and_se(self):
        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                # estimate the causal parameter
                self._all_coef[self._i_treat, self._i_rep] = self._est_causal_pars()

                # compute score (depends on estimated causal parameter)
                self._psi[:, self._i_rep, self._i_treat] = self._compute_score()

                # compute standard errors for causal parameter
                self._all_se[self._i_treat, self._i_rep] = self._se_causal_pars()

            # aggregated parameter estimates and standard errors from repeated cross-fitting
        self._agg_cross_fit()

    def _compute_bootstrap(self, weights):
        if self.apply_cross_fitting:
            J = np.mean(self.__psi_a)
            boot_coef = np.matmul(weights, self.__psi) / (self._dml_data.n_obs * J)
            boot_t_stat = np.matmul(weights, self.__psi) / (self._dml_data.n_obs * self.__all_se * J)

        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            smpls = self.__smpls
            test_index = smpls[0][1]
            J = np.mean(self.__psi_a[test_index])
            boot_coef = np.matmul(weights, self.__psi[test_index]) / (len(test_index) * J)
            boot_t_stat = np.matmul(weights, self.__psi[test_index]) / (len(test_index) * self.__all_se * J)

        return boot_coef, boot_t_stat

    def _var_est(self):
        """
        Estimate the standard errors of the structural parameter
        """
        psi_a = self.__psi_a
        psi = self.__psi

        if self.apply_cross_fitting:
            self._var_scaling_factor = self._dml_data.n_obs
        else:
            # In case of no-cross-fitting, the score function was only evaluated on the test data set
            smpls = self.__smpls
            test_index = smpls[0][1]
            psi_a = psi_a[test_index]
            psi = psi[test_index]
            self._var_scaling_factor = len(test_index)

        J = np.mean(psi_a)
        sigma2_hat = 1 / self._var_scaling_factor * np.mean(np.power(psi, 2)) / np.power(J, 2)

        return sigma2_hat

    def _var_est_cluster_data(self):
        psi_a = self.__psi_a
        psi = self.__psi

        if self._dml_data.n_cluster_vars == 1:
            this_cluster_var = self._dml_data.cluster_vars[:, 0]
            clusters = np.unique(this_cluster_var)
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(self.n_folds):
                test_inds = self.__smpls[i_fold][1]
                test_cluster_inds = self.__smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                const = 1 / len(I_k)
                for cluster_value in I_k:
                    ind_cluster = (this_cluster_var == cluster_value)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_a[test_inds]) / len(I_k)

            gamma_hat = gamma_hat / self._n_folds_per_cluster
            j_hat = j_hat / self._n_folds_per_cluster
            self._var_scaling_factor = len(clusters)
            sigma2_hat = gamma_hat / (j_hat ** 2) / self._var_scaling_factor
        else:
            assert self._dml_data.n_cluster_vars == 2
            first_cluster_var = self._dml_data.cluster_vars[:, 0]
            second_cluster_var = self._dml_data.cluster_vars[:, 1]
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(self.n_folds):
                test_inds = self.__smpls[i_fold][1]
                test_cluster_inds = self.__smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                J_l = test_cluster_inds[1]
                const = min(len(I_k), len(J_l)) / ((len(I_k) * len(J_l)) ** 2)
                for cluster_value in I_k:
                    ind_cluster = (first_cluster_var == cluster_value) & np.in1d(second_cluster_var, J_l)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                for cluster_value in J_l:
                    ind_cluster = (second_cluster_var == cluster_value) & np.in1d(first_cluster_var, I_k)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_a[test_inds]) / (len(I_k) * len(J_l))
            gamma_hat = gamma_hat / (self._n_folds_per_cluster ** 2)
            j_hat = j_hat / (self._n_folds_per_cluster ** 2)
            n_first_clusters = len(np.unique(first_cluster_var))
            n_second_clusters = len(np.unique(second_cluster_var))
            self._var_scaling_factor = min(n_first_clusters, n_second_clusters)
            sigma2_hat = gamma_hat / (j_hat ** 2) / self._var_scaling_factor

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

    def _orth_est_cluster_data(self):
        dml_procedure = self.dml_procedure
        smpls = self.__smpls
        psi_a = self.__psi_a
        psi_b = self.__psi_b

        if dml_procedure == 'dml1':
            # note that in the dml1 case we could also simply apply the standard function without cluster adjustment
            thetas = np.zeros(len(smpls))
            for i_fold, (_, test_index) in enumerate(smpls):
                test_cluster_inds = self.__smpls_cluster[i_fold][1]
                scaling_factor = 1./np.prod(np.array([len(inds) for inds in test_cluster_inds]))
                thetas[i_fold] = - (scaling_factor * np.sum(psi_b[test_index])) / \
                    (scaling_factor * np.sum(psi_a[test_index]))
            theta = np.mean(thetas)
            self._all_dml1_coef[self._i_treat, self._i_rep, :] = thetas
        else:
            assert dml_procedure == 'dml2'
            # See Chiang et al. (2021) Algorithm 1
            psi_a_subsample_mean = 0.
            psi_b_subsample_mean = 0.
            for i_fold, (_, test_index) in enumerate(smpls):
                test_cluster_inds = self.__smpls_cluster[i_fold][1]
                scaling_factor = 1./np.prod(np.array([len(inds) for inds in test_cluster_inds]))
                psi_a_subsample_mean += scaling_factor * np.sum(psi_a[test_index])
                psi_b_subsample_mean += scaling_factor * np.sum(psi_b[test_index])
            theta = -psi_b_subsample_mean / psi_a_subsample_mean

        return theta

    def _compute_score(self):
        psi = self.__psi_a * self.__all_coef + self.__psi_b
        return psi

    def _clean_scores(self):
        del self._psi
        del self._psi_a
        del self._psi_b
