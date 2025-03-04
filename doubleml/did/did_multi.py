import copy
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from doubleml.data import DoubleMLPanelData
from doubleml.did.did_binary import DoubleMLDIDBinary
from doubleml.double_ml import DoubleML
from doubleml.double_ml_framework import concat
from doubleml.utils._checks import _check_score, _check_trimming
from doubleml.utils._descriptive import generate_summary
from doubleml.utils._did_utils import (
    _check_control_group,
    _check_gt_combination,
    _check_gt_values,
    _construct_gt_combinations,
    _construct_gt_index,
    _get_never_treated_value,
)
from doubleml.utils.gain_statistics import gain_statistics


class DoubleMLDIDMulti:
    """Double machine learning for multi-period difference-in-differences models.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLPanelData` object
        The :class:`DoubleMLPanelData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(d,X) = E[Y_1-Y_0|D=d, X]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=1|X]`.
        Only relevant for ``score='observational'``. Default is ``None``.

    gt_combinations : array-like
        TODO: Add description

    control_group : str
        Specifies the control group. Either ``'never_treated'`` or ``'not_yet_treated'``.
        Default is ``'never_treated'``.

    n_folds : int
        Number of folds for cross-fitting.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'observational'`` or ``'experimental'``) specifying the score function.
        The ``'experimental'`` scores refers to an A/B setting, where the treatment is independent
        from the pretreatment covariates.
        Default is ``'observational'``.

    in_sample_normalization : bool
        Indicates whether to use in-sample normalization of weights.
        Default is ``True``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization.
        Default is ``True``.

    print_periods : bool
        Indicates whether to print information about the evaluated periods.
        Default is ``False``.

    Examples
    --------
    TODO: Add example
    """

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m=None,
        gt_combinations="standard",
        control_group="never_treated",
        n_folds=5,
        n_rep=1,
        score="observational",
        in_sample_normalization=True,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
        print_periods=False,
    ):

        self._dml_data = obj_dml_data
        self._is_cluster_data = False
        self._is_panel_data = isinstance(obj_dml_data, DoubleMLPanelData)
        self._check_data(self._dml_data)
        self._g_values = self._dml_data.g_values
        self._t_values = self._dml_data.t_values
        self._print_periods = print_periods

        self._control_group = _check_control_group(control_group)
        self._never_treated_value = _get_never_treated_value(self.g_values)

        self._gt_combinations = self._validate_gt_combinations(gt_combinations)
        self._gt_index = _construct_gt_index(self.gt_combinations, self.g_values, self.t_values)
        self._gt_labels = [f"ATT({g},{t_pre},{t_eval})" for g, t_pre, t_eval in self.gt_combinations]

        # TODO: Check what to export and what not
        self._in_sample_normalization = in_sample_normalization
        if not isinstance(self.in_sample_normalization, bool):
            raise TypeError(
                "in_sample_normalization indicator has to be boolean. "
                + f"Object of type {str(type(self.in_sample_normalization))} passed."
            )

        self._n_folds = n_folds
        self._n_rep = n_rep

        # check score
        self._score = score
        valid_scores = ["observational", "experimental"]
        _check_score(self.score, valid_scores, allow_callable=False)

        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        ml_g_is_classifier = DoubleML._check_learner(ml_g, "ml_g", regressor=True, classifier=True)
        _ = DoubleML._check_learner(ml_m, "ml_m", regressor=False, classifier=True)
        self._learner = {"ml_g": clone(ml_g), "ml_m": clone(ml_m)}
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {"ml_g": "predict_proba", "ml_m": "predict_proba"}
            else:
                raise ValueError(
                    f"The ml_g learner {str(ml_g)} was identified as classifier "
                    "but the outcome variable is not binary with values 0 and 1."
                )
        else:
            self._predict_method = {"ml_g": "predict", "ml_m": "predict_proba"}

        # perform sample splitting
        self._smpls = None

        # TODO: Check draw_sample_splitting here vs. DoubleMLDIDBINARY
        self._draw_sample_splitting = draw_sample_splitting

        # initialize all models if splits are known
        self._modellist = self._initialize_models()
        self._nuisance_loss = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        data_summary = self._dml_data._data_summary_str()
        score_info = f"Score function: {str(self.score)}\n"
        resampling_info = f"No. folds: {self.n_folds}\nNo. repeated sample splits: {self.n_rep}\n"
        learner_info = ""
        for key, value in self._learner.items():
            learner_info += f"Learner {key}: {str(value)}\n"
        if self.nuisance_loss is not None:
            learner_info += "Out-of-sample Performance:\n"
            is_classifier = [value for value in self.modellist[0]._is_classifier.values()]
            is_regressor = [not value for value in is_classifier]
            if any(is_regressor):
                learner_info += "Regression:\n"
                for learner in [key for key, value in self.modellist[0]._is_classifier.items() if value is False]:
                    learner_info += f"Learner {learner} RMSE: {self.nuisance_loss[learner]}\n"
            if any(is_classifier):
                learner_info += "Classification:\n"
                for learner in [key for key, value in self.modellist[0]._is_classifier.items() if value is True]:
                    learner_info += f"Learner {learner} Log Loss: {self.nuisance_loss[learner]}\n"
        fit_summary = str(self.summary)
        res = (
            header
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ Score & algorithm ------------------\n"
            + score_info
            + "\n------------------ Machine learner   ------------------\n"
            + learner_info
            + "\n------------------ Resampling        ------------------\n"
            + resampling_info
            + "\n------------------ Fit summary       ------------------\n"
            + fit_summary
        )
        return res

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def control_group(self):
        """
        The control group.
        """
        return self._control_group

    @property
    def gt_combinations(self):
        """
        The combinations of g and t values.
        """
        return self._gt_combinations

    @property
    def gt_index(self):
        """
        The index of the combinations of g and t values.
        """
        return self._gt_index

    @property
    def n_gt_atts(self):
        """
        The number of evaluated combinations of the treatment variable and the period.
        """
        return len(self.gt_combinations)

    @property
    def gt_labels(self):
        """
        The evaluated labels of the treatment effects 'ATT(g, t_pre, t_eval)' and the period.
        """
        return self._gt_labels

    @property
    def g_values(self):
        """
        The values of the treatment variable.
        """
        return self._g_values

    @property
    def t_values(self):
        """
        The values of the time periods.
        """
        return self._t_values

    @property
    def never_treated_value(self):
        """
        The value indicating that a unit was never treated.
        """
        return self._never_treated_value

    @property
    def in_sample_normalization(self):
        """
        Indicates whether the in sample normalization of weights are used.
        """
        return self._in_sample_normalization

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
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape (``n_gt_atts``,)).
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
         (shape (``n_gt_atts``, ``n_rep``)).
        """
        if self._framework is None:
            all_coef = None
        else:
            all_coef = self.framework.all_thetas
        return all_coef

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (``n_gt_atts``,)).
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
         (shape (``n_gt_atts``, ``n_rep``)).
        """
        if self._framework is None:
            all_se = None
        else:
            all_se = self.framework.all_ses
        return all_se

    @property
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit` (shape (``n_gt_atts``,)).
        """
        if self._framework is None:
            t_stats = None
        else:
            t_stats = self.framework.t_stats
        return t_stats

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) (shape (``n_gt_atts``,)).
        """
        if self._framework is None:
            pvals = None
        else:
            pvals = self.framework.pvals
        return pvals

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`
         (shape (``n_rep_boot``, ``n_gt_atts``, ``n_rep``)).
        """
        if self._framework is None:
            boot_t_stat = None
        else:
            boot_t_stat = self._framework.boot_t_stat
        return boot_t_stat

    @property
    def smpls(self):
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            err_msg = (
                "Sample splitting not specified. Draw samples via .draw_sample_splitting(). "
                + "External samples not implemented yet."
            )
            raise NotImplementedError(err_msg)
        return self._smpls

    @property
    def nuisance_loss(self):
        """
        The losses of the nuisance models (root-mean-squared-errors or logloss).
        """
        return self._nuisance_loss

    @property
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

    @property
    def modellist(self):
        """
        The list of DoubleMLDIDBinary models.
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
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_summary = pd.DataFrame(columns=col_names)
        else:
            ci = self.confint()
            df_summary = generate_summary(self.coef, self.se, self.t_stat, self.pval, ci, self.gt_labels)
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
            raise ValueError("Apply sensitivity_analysis() before sensitivity_summary.")
        else:
            sensitivity_summary = self._framework.sensitivity_summary
        return sensitivity_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        """
        Estimate DoubleMLDIDMulti models.

        Parameters
        ----------
        n_jobs_models : None or int
            The number of CPUs to use to fit the group-time ATTs. ``None`` means ``1``.
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
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")
        fitted_models = parallel(
            delayed(self._fit_model)(i_gt, n_jobs_cv, store_predictions, store_models, ext_pred_dict)
            for i_gt in range(self.n_gt_atts)
        )

        # combine the estimates and scores
        framework_list = [None] * self.n_gt_atts

        for i_gt in range(self.n_gt_atts):
            self._modellist[i_gt] = fitted_models[i_gt]
            framework_list[i_gt] = self._modellist[i_gt].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)
        self._framework.treatment_names = self._gt_labels

        # store the nuisance losses
        self._nuisance_loss = self._calc_nuisance_loss()

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
            raise ValueError("Apply fit() before confint().")

        df_ci = self.framework.confint(joint=joint, level=level)
        df_ci.set_index(pd.Index(self.gt_labels), inplace=True)

        return df_ci

    def p_adjust(self, method="romano-wolf"):
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

        if self.framework is None:
            raise ValueError("Apply fit() before p_adjust().")

        p_val, _ = self.framework.p_adjust(method=method)
        # TODO: Update for different combinations
        p_val.set_index(pd.Index(self._dml_data.d_cols), inplace=True)

        return p_val

    def bootstrap(self, method="normal", n_rep_boot=500):
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
            raise ValueError("Apply fit() before bootstrap().")
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
            raise ValueError("Apply fit() before sensitivity_analysis().")
        self._framework.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level, null_hypothesis=null_hypothesis)

        return self

    def sensitivity_plot(
        self,
        idx_treatment=0,
        value="theta",
        rho=1.0,
        level=0.95,
        null_hypothesis=0.0,
        include_scenario=True,
        benchmarks=None,
        fill=True,
        grid_bounds=(0.15, 0.15),
        grid_size=100,
    ):
        """
        Contour plot of the sensivity with respect to latent/confounding variables.
        Parameters
        ----------
        idx_gt_atte : int
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
            raise ValueError("Apply fit() before sensitivity_plot().")
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
            grid_size=grid_size,
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
            raise NotImplementedError(f"Sensitivity analysis not yet implemented for {self.__class__.__name__}.")
        if not isinstance(benchmarking_set, list):
            raise TypeError(
                "benchmarking_set must be a list. " f"{str(benchmarking_set)} of type {type(benchmarking_set)} was passed."
            )
        if len(benchmarking_set) == 0:
            raise ValueError("benchmarking_set must not be empty.")
        if not set(benchmarking_set) <= set(x_list_long):
            raise ValueError(
                f"benchmarking_set must be a subset of features {str(self._dml_data.x_cols)}. "
                f"{str(benchmarking_set)} was passed."
            )
        if fit_args is not None and not isinstance(fit_args, dict):
            raise TypeError("fit_args must be a dict. " f"{str(fit_args)} of type {type(fit_args)} was passed.")

        # refit short form of the model
        x_list_short = [x for x in x_list_long if x not in benchmarking_set]
        dml_short = copy.deepcopy(self)
        dml_short._dml_data.x_cols = x_list_short
        if fit_args is not None:
            dml_short.fit(**fit_args)
        else:
            dml_short.fit()

        benchmark_dict = gain_statistics(dml_long=self, dml_short=dml_short)
        df_benchmark = pd.DataFrame(benchmark_dict, index=self.gt_labels)
        return df_benchmark

    def aggregate(self, aggregation="group"):
        if not isinstance(aggregation, str):
            raise TypeError("aggregation must be a string. " f"{str(aggregation)} of type {type(aggregation)} was passed.")
        valid_aggregations = ["group"]
        if aggregation not in valid_aggregations:
            raise ValueError(f"aggregation must be one of {valid_aggregations}. " f"{str(aggregation)} was passed.")
        if self.framework is None:
            raise ValueError("Apply fit() before aggregate().")

        all_agg_frameworks = []
        agg_names = []
        if aggregation == "group":
            group_weights = []
            for i_group, group in enumerate(self.g_values):
                # Get indices of non-masked values for this group
                group_indices = self.gt_index[i_group, :, :].compressed()
                if len(group_indices) > 0:
                    frameworks_for_group = [
                        self.modellist[idx].framework for idx in group_indices if self.modellist[idx].post_treatment
                    ]
                    if len(frameworks_for_group) > 1:
                        agg_framework = reduce(add, frameworks_for_group)
                        group_weights.append((self._dml_data.d == group).mean())
                        all_agg_frameworks.append(agg_framework)

                    elif len(frameworks_for_group) == 1:
                        agg_framework = frameworks_for_group[0]
                        group_weights.append((self._dml_data.d == group).mean())
                        all_agg_frameworks.append(agg_framework)
                        agg_names.append(str(group))

            rescaled_group_weights = [w / sum(group_weights) for w in group_weights]
            weighted_frameworks = [w * f for w, f in zip(rescaled_group_weights, frameworks_for_group)]
            if len(weighted_frameworks) > 1:
                overall_agg_framework = reduce(add, weighted_frameworks)
            elif len(weighted_frameworks) == 1:
                overall_agg_framework = weighted_frameworks[0]
            else:
                raise ValueError("No valid groups found for aggregation.")
            all_agg_frameworks.insert(0, overall_agg_framework)
            agg_names.insert(0, "Overall")

        agg_framework = concat(all_agg_frameworks)
        agg_framework.treatment_names = agg_names
        return agg_framework

    def _fit_model(self, i_gt, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions_dict=None):

        model = self.modellist[i_gt]
        if external_predictions_dict is not None:
            external_predictions = external_predictions_dict[self.gt_labels[i_gt]]
        else:
            external_predictions = None
        model.fit(
            n_jobs_cv=n_jobs_cv,
            store_predictions=store_predictions,
            store_models=store_models,
            external_predictions=external_predictions,
        )
        return model

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError(
                "The data has to be a DoubleMLPanelData object. "
                f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise NotImplementedError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "At the moment there are not DiD models with instruments implemented."
            )
        _check_gt_values(obj_dml_data.g_values, obj_dml_data.t_values)
        return

    def _validate_gt_combinations(self, gt_combinations):
        """Validate all treatment-time combinations."""

        if isinstance(gt_combinations, str):
            gt_combinations = _construct_gt_combinations(
                gt_combinations, self.g_values, self.t_values, self.never_treated_value
            )

        if not isinstance(gt_combinations, list):
            raise TypeError(
                "gt_combinations must be a list. " + f"{str(gt_combinations)} of type {type(gt_combinations)} was passed."
            )

        if len(gt_combinations) == 0:
            raise ValueError("gt_combinations must not be empty.")

        if not all(isinstance(gt_combination, tuple) for gt_combination in gt_combinations):
            raise TypeError("gt_combinations must be a list of tuples. At least one element is not a tuple.")

        if not all(len(gt_combination) == 3 for gt_combination in gt_combinations):
            raise ValueError(
                "gt_combinations must be a list of tuples with 3 elements. At least one tuple has not 3 elements."
            )

        for gt_combination in gt_combinations:
            _check_gt_combination(gt_combination, self.g_values, self.t_values, self.never_treated_value)

        return gt_combinations

    def _check_external_predictions(self, external_predictions):
        expected_keys = self.gt_labels
        if not isinstance(external_predictions, dict):
            raise TypeError(
                "external_predictions must be a dictionary. " + f"Object of type {type(external_predictions)} passed."
            )

        if not set(external_predictions.keys()).issubset(set(expected_keys)):
            raise ValueError(
                "external_predictions must be a subset of all gt_combinations. "
                + f"Expected keys: {set(expected_keys)}. "
                + f"Passed keys: {set(external_predictions.keys())}."
            )

        expected_learner_keys = ["ml_g0", "ml_g1", "ml_m"]
        for key, value in external_predictions.items():
            if not isinstance(value, dict):
                raise TypeError(
                    f"external_predictions[{key}] must be a dictionary. " + f"Object of type {type(value)} passed."
                )
            if not set(value.keys()).issubset(set(expected_learner_keys)):
                raise ValueError(
                    f"external_predictions[{key}] must be a subset of {set(expected_learner_keys)}. "
                    + f"Passed keys: {set(value.keys())}."
                )

        return

    def _rename_external_predictions(self, external_predictions):
        d_col = self._dml_data.d_cols[0]
        ext_pred_dict = {gt_combination: {d_col: {}} for gt_combination in self.gt_labels}
        for gt_combination in self.gt_labels:
            if "ml_g0" in external_predictions[gt_combination]:
                ext_pred_dict[gt_combination][d_col]["ml_g0"] = external_predictions[gt_combination]["ml_g0"]
            if "ml_g1" in external_predictions[gt_combination]:
                ext_pred_dict[gt_combination][d_col]["ml_g1"] = external_predictions[gt_combination]["ml_g1"]
            if "ml_m" in external_predictions[gt_combination]:
                ext_pred_dict[gt_combination][d_col]["ml_m"] = external_predictions[gt_combination]["ml_m"]

        return ext_pred_dict

    def _calc_nuisance_loss(self):
        nuisance_loss = {learner: np.full((self.n_rep, self.n_gt_atts), np.nan) for learner in self.modellist[0].params_names}
        for i_model, model in enumerate(self.modellist):
            for learner in self.modellist[0].params_names:
                for i_rep in range(self.n_rep):
                    nuisance_loss[learner][i_rep, i_model] = model.nuisance_loss[learner][i_rep]
                    nuisance_loss[learner][i_rep, i_model] = model.nuisance_loss[learner][i_rep]

        return nuisance_loss

    def _initialize_models(self):
        modellist = [None] * self.n_gt_atts
        kwargs = {
            "obj_dml_data": self._dml_data,
            "ml_g": self._learner["ml_g"],
            "ml_m": self._learner["ml_m"],
            "control_group": self._control_group,
            "score": self.score,
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "trimming_rule": self.trimming_rule,
            "trimming_threshold": self.trimming_threshold,
            "in_sample_normalization": self.in_sample_normalization,
            "draw_sample_splitting": True,
            "print_periods": self._print_periods,
        }
        for i_model, (g_value, t_value_pre, t_value_eval) in enumerate(self.gt_combinations):
            # initialize models for all levels
            model = DoubleMLDIDBinary(g_value=g_value, t_value_pre=t_value_pre, t_value_eval=t_value_eval, **kwargs)

            modellist[i_model] = model

        return modellist
