import numpy as np
import pandas as pd
import copy
from collections.abc import Iterable

from sklearn.base import clone

from joblib import Parallel, delayed

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData, DoubleMLClusterData
from .apo import DoubleMLAPO
from ..double_ml_framework import concat

from ..utils.resampling import DoubleMLResampling
from ..utils._descriptive import generate_summary
from ..utils._checks import _check_score, _check_trimming, _check_weights, _check_sample_splitting
from ..utils.gain_statistics import gain_statistics


class DoubleMLAPOS:
    """Double machine learning for interactive regression models with multiple discrete treatments.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment_levels,
                 n_folds=5,
                 n_rep=1,
                 score='APO',
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):

        self._dml_data = obj_dml_data
        self._is_cluster_data = isinstance(obj_dml_data, DoubleMLClusterData)
        self._check_data(self._dml_data)

        self._all_treatment_levels = np.unique(self._dml_data.d)

        self._treatment_levels = self._check_treatment_levels(treatment_levels)
        self._n_treatment_levels = len(self._treatment_levels)
        # Check if there are elements in self._all_treatments that are not in self.treatment_levels
        self._add_treatment_levels = [t for t in self._all_treatment_levels if t not in self._treatment_levels]

        self._normalize_ipw = normalize_ipw
        self._n_folds = n_folds
        self._n_rep = n_rep

        # check score
        self._score = score
        valid_scores = ['APO']
        _check_score(self.score, valid_scores, allow_callable=False)

        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

        ml_g_is_classifier = DoubleML._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = DoubleML._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}

        # APO weights
        _check_weights(weights, score="ATE", n_obs=obj_dml_data.n_obs, n_rep=self.n_rep)
        self._initialize_weights(weights)

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

            # initialize all models if splits are known
            self._modellist = self._initialize_models()

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Fit summary       ------------------\n' + fit_summary
        return res

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def n_treatment_levels(self):
        """
        The number of treatment levels.
        """
        return self._n_treatment_levels

    @property
    def treatment_levels(self):
        """
        The evaluated treatment levels.
        """
        return self._treatment_levels

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

    @property
    def weights(self):
        """
        Specifies the weights for a weighted average potential outcome.
        """
        return self._weights

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
    def n_rep_boot(self):
        """
        The number of bootstrap replications.
        """
        if self._framework is None:
            n_rep_boot = None
        else:
            n_rep_boot = self._framework.n_rep_boot
        return n_rep_boot

    @property
    def boot_method(self):
        """
        The method to construct the bootstrap replications.
        """
        if self._framework is None:
            method = None
        else:
            method = self._framework.boot_method
        return method

    @property
    def coef(self):
        """
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            coef = None
        else:
            coef = self.framework.thetas
        return coef

    @property
    def all_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`
         (shape (``n_treatment_levels``, ``n_rep``)).
        """
        if self._framework is None:
            all_coef = None
        else:
            all_coef = self.framework.all_thetas
        return all_coef

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            se = None
        else:
            se = self.framework.ses
        return se

    @property
    def all_se(self):
        """
        Standard errors of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`
         (shape (``n_treatment_levels``, ``n_rep``)).
        """
        if self._framework is None:
            all_se = None
        else:
            all_se = self.framework.all_ses
        return all_se

    @property
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            t_stats = None
        else:
            t_stats = self.framework.t_stats
        return t_stats

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            pvals = None
        else:
            pvals = self.framework.pvals
        return pvals

    @property
    def smpls(self):
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            err_msg = ('Sample splitting not specified. Draw samples via .draw_sample splitting(). ' +
                       'External samples not implemented yet.')
            raise ValueError(err_msg)
        return self._smpls

    @property
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`
         (shape (``n_rep_boot``, ``n_treatment_levels``, ``n_rep``)).
        """
        if self._framework is None:
            boot_t_stat = None
        else:
            boot_t_stat = self._framework.boot_t_stat
        return boot_t_stat

    @property
    def modellist(self):
        """
        The list of models for each level.
        """
        return self._modellist

    @property
    def sensitivity_elements(self):
        """
        Values of the sensitivity components after calling :meth:`fit`;
        If available (e.g., PLR, IRM) a dictionary with entries ``sigma2``, ``nu2``, ``psi_sigma2``, ``psi_nu2``
        and ``riesz_rep``.
        """
        if self._framework is None:
            sensitivity_elements = None
        else:
            sensitivity_elements = self._framework.sensitivity_elements
        return sensitivity_elements

    @property
    def sensitivity_params(self):
        """
        Values of the sensitivity parameters after calling :meth:`sesitivity_analysis`;
        If available (e.g., PLR, IRM) a dictionary with entries ``theta``, ``se``, ``ci``, ``rv``
        and ``rva``.
        """
        if self._framework is None:
            sensitivity_params = None
        else:
            sensitivity_params = self._framework.sensitivity_params
        return sensitivity_params

    @property
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        if self.framework is None:
            col_names = ['coef', 'std err', 't', 'P>|t|']
            df_summary = pd.DataFrame(columns=col_names)
        else:
            ci = self.confint()
            df_summary = generate_summary(self.coef, self.se, self.t_stat,
                                          self.pval, ci, self._treatment_levels)
        return df_summary

    @property
    def sensitivity_summary(self):
        """
        Returns a summary for the sensitivity analysis after calling :meth:`sensitivity_analysis`.

        Returns
        -------
        res : str
            Summary for the sensitivity analysis.
        """
        if self._framework is None:
            raise ValueError('Apply sensitivity_analysis() before sensitivity_summary.')
        else:
            sensitivity_summary = self._framework.sensitivity_summary
        return sensitivity_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        """
        Estimate DoubleMLAPOS models.

        Parameters
        ----------
        n_jobs_models : None or int
            The number of CPUs to use to fit the treatment_levels. ``None`` means ``1``.
            Default is ``None``.

        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Does not speed up computation for quantile models.
            Default is ``None``.

        store_predictions : bool
            Indicates whether the predictions for the nuisance functions should be stored in ``predictions``.
            Default is ``True``.

        store_models : bool
            Indicates whether the fitted models for the nuisance functions should be stored in ``models``. This allows
            to analyze the fitted models or extract information like variable importance.
            Default is ``False``.

        external_predictions : dict or None
            A nested dictionary where the keys correspond the the treatment levels and can contain predictions according to
            each treatment level. The values have to be dictionaries which can contain keys ``'ml_g0'``, ``'ml_g1'``
            and ``'ml_m'``.
            Default is `None`.

        Returns
        -------
        self : object
        """

        if external_predictions is not None:
            self._check_external_predictions(external_predictions)
            ext_pred_dict = self._rename_external_predictions(external_predictions)
        else:
            ext_pred_dict = None

        # parallel estimation of the models
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch='2*n_jobs')
        fitted_models = parallel(
            delayed(self._fit_model)(
                i_level,
                n_jobs_cv,
                store_predictions,
                store_models,
                ext_pred_dict)
            for i_level in range(self.n_treatment_levels)
        )

        # combine the estimates and scores
        framework_list = [None] * self.n_treatment_levels

        for i_level in range(self.n_treatment_levels):
            self._modellist[i_level] = fitted_models[i_level]
            framework_list[i_level] = self._modellist[i_level].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

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

        if self.framework is None:
            raise ValueError('Apply fit() before confint().')

        df_ci = self.framework.confint(joint=joint, level=level)
        df_ci.set_index(pd.Index(self._treatment_levels), inplace=True)

        return df_ci

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
        if self._framework is None:
            raise ValueError('Apply fit() before bootstrap().')
        self._framework.bootstrap(method=method, n_rep_boot=n_rep_boot)

        return self

    def sensitivity_analysis(self, cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0):
        """
        Performs a sensitivity analysis to account for unobserved confounders.

        The evaluated scenario is stored as a dictionary in the property ``sensitivity_params``.

        Parameters
        ----------
        cf_y : float
            Percentage of the residual variation of the outcome explained by latent/confounding variables.
            Default is ``0.03``.

        cf_d : float
            Percentage gains in the variation of the Riesz representer generated by latent/confounding variables.
            Default is ``0.03``.

        rho : float
            The correlation between the differences in short and long representations in the main regression and
            Riesz representer. Has to be in [-1,1]. The absolute value determines the adversarial strength of the
            confounding (maximizes at 1.0).
            Default is ``1.0``.

        level : float
            The confidence level.
            Default is ``0.95``.

        null_hypothesis : float or numpy.ndarray
            Null hypothesis for the effect. Determines the robustness values.
            If it is a single float uses the same null hypothesis for all estimated parameters.
            Else the array has to be of shape (n_coefs,).
            Default is ``0.0``.

        Returns
        -------
        self : object
        """

        if self._framework is None:
            raise ValueError('Apply fit() before sensitivity_analysis().')
        self._framework.sensitivity_analysis(
            cf_y=cf_y,
            cf_d=cf_d,
            rho=rho,
            level=level,
            null_hypothesis=null_hypothesis
        )

        return self

    def sensitivity_plot(self, idx_treatment=0, value='theta', rho=1.0, level=0.95, null_hypothesis=0.0,
                         include_scenario=True, benchmarks=None, fill=True, grid_bounds=(0.15, 0.15), grid_size=100):
        """
        Contour plot of the sensivity with respect to latent/confounding variables.

        Parameters
        ----------
        idx_treatment : int
            Index of the treatment to perform the sensitivity analysis.
            Default is ``0``.

        value : str
            Determines which contours to plot. Valid values are ``'theta'`` (refers to the bounds)
            and ``'ci'`` (refers to the bounds including statistical uncertainty).
            Default is ``'theta'``.

        rho: float
            The correlation between the differences in short and long representations in the main regression and
            Riesz representer. Has to be in [-1,1]. The absolute value determines the adversarial strength of the
            confounding (maximizes at 1.0).
            Default is ``1.0``.

        level : float
            The confidence level.
            Default is ``0.95``.

        null_hypothesis : float
            Null hypothesis for the effect. Determines the direction of the contour lines.

        include_scenario : bool
            Indicates whether to highlight the scenario from the call of :meth:`sensitivity_analysis`.
            Default is ``True``.

        benchmarks : dict or None
            Dictionary of benchmarks to be included in the plot. The keys are ``cf_y``, ``cf_d`` and ``name``.
            Default is ``None``.

        fill : bool
            Indicates whether to use a heatmap style or only contour lines.
            Default is ``True``.

        grid_bounds : tuple
            Determines the evaluation bounds of the grid for ``cf_d`` and ``cf_y``. Has to contain two floats in [0, 1).
            Default is ``(0.15, 0.15)``.

        grid_size : int
            Determines the number of evaluation points of the grid.
            Default is ``100``.

        Returns
        -------
        fig : object
            Plotly figure of the sensitivity contours.
        """
        if self._framework is None:
            raise ValueError('Apply fit() before sensitivity_plot().')
        fig = self._framework.sensitivity_plot(
            idx_treatment=idx_treatment,
            value=value,
            rho=rho,
            level=level,
            null_hypothesis=null_hypothesis,
            include_scenario=include_scenario,
            benchmarks=benchmarks,
            fill=fill,
            grid_bounds=grid_bounds,
            grid_size=grid_size
        )

        return fig

    def sensitivity_benchmark(self, benchmarking_set, fit_args=None):
        """
        Computes a benchmark for a given set of features.
        Returns a DataFrame containing the corresponding values for cf_y, cf_d, rho and the change in estimates.
        Returns
        -------
        benchmark_results : pandas.DataFrame
            Benchmark results.
        """
        x_list_long = self._dml_data.x_cols

        # input checks
        if self.sensitivity_elements is None:
            raise NotImplementedError(f'Sensitivity analysis not yet implemented for {self.__class__.__name__}.')
        if not isinstance(benchmarking_set, list):
            raise TypeError('benchmarking_set must be a list. '
                            f'{str(benchmarking_set)} of type {type(benchmarking_set)} was passed.')
        if len(benchmarking_set) == 0:
            raise ValueError('benchmarking_set must not be empty.')
        if not set(benchmarking_set) <= set(x_list_long):
            raise ValueError(f"benchmarking_set must be a subset of features {str(self._dml_data.x_cols)}. "
                             f'{str(benchmarking_set)} was passed.')
        if fit_args is not None and not isinstance(fit_args, dict):
            raise TypeError('fit_args must be a dict. '
                            f'{str(fit_args)} of type {type(fit_args)} was passed.')

        # refit short form of the model
        x_list_short = [x for x in x_list_long if x not in benchmarking_set]
        dml_short = copy.deepcopy(self)
        dml_short._dml_data.x_cols = x_list_short
        if fit_args is not None:
            dml_short.fit(**fit_args)
        else:
            dml_short.fit()

        benchmark_dict = gain_statistics(dml_long=self, dml_short=dml_short)
        df_benchmark = pd.DataFrame(benchmark_dict, index=self.treatment_levels)
        return df_benchmark

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """
        obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                n_rep=self.n_rep,
                                                n_obs=self._dml_data.n_obs,
                                                stratify=self._dml_data.d)
        self._smpls = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self, all_smpls, all_smpls_cluster=None):
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
                ``n_folds``). test_ind must form a partition for each inner list.
            If list of tuples:
                The list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as
                ``n_folds``). test_ind must form a partition. ``n_rep=1`` is always set.
            If tuple:
                Must be a tuple with two elements train_ind and test_ind. Only viable option is to set
                train_ind and test_ind to np.arange(n_obs), which corresponds to no sample splitting.
                ``n_folds=1`` and ``n_rep=1`` is always set.

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
        self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
            all_smpls, all_smpls_cluster, self._dml_data, self._is_cluster_data)

        self._modellist = self._initialize_models()

        return self

    def causal_contrast(self, reference_levels):
        """
        Average causal contrasts for DoubleMLAPOS models. Estimates the difference in
        average potential outcomes between the treatment levels and the reference levels.
        The reference levels have to be a subset of the treatment levels or a single
        treatment level.

        Parameters
        ----------
        reference_levels :
            The reference levels for the difference in average potential outcomes.
            Has to be an element of ``treatment_levels``.

        Returns
        -------
        acc : DoubleMLFramework
            A DoubleMLFramwork class for average causal contrast(s).
        """

        if self.framework is None:
            raise ValueError('Apply fit() before causal_contrast().')
        if self.n_treatment_levels == 1:
            raise ValueError('Only one treatment level. No causal contrast can be computed.')
        is_iterable = isinstance(reference_levels, Iterable)
        if not is_iterable:
            reference_levels = [reference_levels]
        is_treatment_level_subset = set(reference_levels).issubset(set(self.treatment_levels))
        if not is_treatment_level_subset:
            raise ValueError('Invalid reference_levels. reference_levels has to be an iterable subset of treatment_levels or '
                             'a single treatment level.')

        skip_index = []
        all_treatment_names = []
        all_acc_frameworks = []
        for ref_lvl in reference_levels:
            i_ref_lvl = self.treatment_levels.index(ref_lvl)
            ref_framework = self.modellist[i_ref_lvl].framework

            skip_index += [i_ref_lvl]
            all_acc_frameworks += [model.framework - ref_framework for i, model in
                                   enumerate(self.modellist) if i not in skip_index]
            all_treatment_names += [f"{self.treatment_levels[i]} vs {self.treatment_levels[i_ref_lvl]}" for
                                    i in range(self.n_treatment_levels) if i not in skip_index]

        acc = concat(all_acc_frameworks)
        acc.treatment_names = all_treatment_names
        return acc

    def _fit_model(self, i_level, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions_dict=None):

        model = self.modellist[i_level]
        if external_predictions_dict is not None:
            external_predictions = external_predictions_dict[self.treatment_levels[i_level]]
        else:
            external_predictions = None
        model.fit(n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models,
                  external_predictions=external_predictions)
        return model

    def _check_treatment_levels(self, treatment_levels):
        is_iterable = isinstance(treatment_levels, Iterable)
        if not is_iterable:
            treatment_level_list = [treatment_levels]
        else:
            treatment_level_list = [t_lvl for t_lvl in treatment_levels]
        is_d_subset = set(treatment_level_list).issubset(set(self._all_treatment_levels))
        if not is_d_subset:
            raise ValueError('Invalid reference_levels. reference_levels has to be an iterable subset or '
                             'a single element of the unique treatment levels in the data.')
        return treatment_level_list

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData or DoubleMLClusterData type.')
        if obj_dml_data.z is not None:
            raise ValueError('The data must not contain instrumental variables.')
        return

    def _check_external_predictions(self, external_predictions):
        expected_keys = self.treatment_levels
        if not isinstance(external_predictions, dict):
            raise TypeError('external_predictions must be a dictionary. ' +
                            f'Object of type {type(external_predictions)} passed.')

        if not set(external_predictions.keys()).issubset(set(expected_keys)):
            raise ValueError('external_predictions must be a subset of all treatment levels. ' +
                             f'Expected keys: {set(expected_keys)}. ' +
                             f'Passed keys: {set(external_predictions.keys())}.')

        expected_learner_keys = ['ml_g0', 'ml_g1', 'ml_m']
        for key, value in external_predictions.items():
            if not isinstance(value, dict):
                raise TypeError(f'external_predictions[{key}] must be a dictionary. ' +
                                f'Object of type {type(value)} passed.')
            if not set(value.keys()).issubset(set(expected_learner_keys)):
                raise ValueError(f'external_predictions[{key}] must be a subset of {set(expected_learner_keys)}. ' +
                                 f'Passed keys: {set(value.keys())}.')

        return

    def _rename_external_predictions(self, external_predictions):
        d_col = self._dml_data.d_cols[0]
        ext_pred_dict = {treatment_level: {d_col: {}} for treatment_level in self.treatment_levels}
        for treatment_level in self.treatment_levels:
            if "ml_g1" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]['ml_g1'] = external_predictions[treatment_level]['ml_g1']
            if "ml_m" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]['ml_m'] = external_predictions[treatment_level]['ml_m']
            if "ml_g0" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]['ml_g0'] = external_predictions[treatment_level]['ml_g0']

        return ext_pred_dict

    def _initialize_weights(self, weights):
        if weights is None:
            weights = np.ones(self._dml_data.n_obs)
        if isinstance(weights, np.ndarray):
            self._weights = weights
        else:
            assert isinstance(weights, dict)
            self._weights = weights

    def _initialize_models(self):
        modellist = [None] * self.n_treatment_levels
        kwargs = {
            'obj_dml_data': self._dml_data,
            'ml_g': self._learner['ml_g'],
            'ml_m': self._learner['ml_m'],
            'score': self.score,
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'weights': self.weights,
            'trimming_rule': self.trimming_rule,
            'trimming_threshold': self.trimming_threshold,
            'normalize_ipw': self.normalize_ipw,
            'draw_sample_splitting': False
        }
        for i_level in range(self.n_treatment_levels):
            # initialize models for all levels
            model = DoubleMLAPO(
                treatment_level=self._treatment_levels[i_level],
                **kwargs
            )

            # synchronize the sample splitting
            model.set_sample_splitting(all_smpls=self.smpls)
            modellist[i_level] = model

        return modellist
