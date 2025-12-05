import copy
import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from doubleml.data import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_framework import DoubleMLCore, DoubleMLFramework, concat
from doubleml.double_ml_sampling_mixins import SampleSplittingMixin
from doubleml.irm.apo import DoubleMLAPO
from doubleml.utils._checks import _check_score, _check_weights
from doubleml.utils._descriptive import generate_summary
from doubleml.utils._sensitivity import _compute_sensitivity_bias
from doubleml.utils._tune_optuna import TUNE_ML_MODELS_DOC
from doubleml.utils.gain_statistics import gain_statistics
from doubleml.utils.propensity_score_processing import PSProcessorConfig, init_ps_processor


class DoubleMLAPOS(SampleSplittingMixin):
    """Double machine learning for interactive regression models with multiple discrete
    treatments.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(D, X) = E[Y | X, D]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D | X]`.

    treatment_levels : iterable of int or float
        The treatment levels for which average potential outcomes are evaluated. Each element must be present in the
        treatment variable ``d`` of ``obj_dml_data``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'APO'``) specifying the score function.
        Default is ``'APO'``.

    weights : array, dict or None
        A numpy array of weights for each individual observation. If ``None``, then the ``'APO'`` score
        is applied (corresponds to weights equal to 1).
        An array has to be of shape ``(n,)``, where ``n`` is the number of observations.
        A dictionary can be used to specify weights which depend on the treatment variable.
        In this case, the dictionary has to contain two keys ``weights`` and ``weights_bar``, where the values
        have to be arrays of shape ``(n,)`` and ``(n, n_rep)``.
        Default is ``None``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.

    trimming_rule : str, optional, deprecated
        (DEPRECATED) A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Use ``ps_processor_config`` instead. Will be removed in a future version.

    trimming_threshold : float, optional, deprecated
        (DEPRECATED) The threshold used for trimming.
        Use ``ps_processor_config`` instead. Will be removed in a future version.

    ps_processor_config : PSProcessorConfig, optional
        Configuration for propensity score processing (clipping, calibration, etc.).

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.
    """

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m,
        treatment_levels,
        n_folds=5,
        n_rep=1,
        score="APO",
        weights=None,
        normalize_ipw=False,
        trimming_rule="truncate",  # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        trimming_threshold=1e-2,  # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        ps_processor_config: Optional[PSProcessorConfig] = None,
        draw_sample_splitting=True,
    ):
        self._dml_data = obj_dml_data
        self._check_data(self._dml_data)
        self._is_cluster_data = self._dml_data.is_cluster_data

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
        valid_scores = ["APO"]
        _check_score(self.score, valid_scores, allow_callable=False)

        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        self._ps_processor_config, self._ps_processor = init_ps_processor(
            ps_processor_config, trimming_rule, trimming_threshold
        )
        self._trimming_rule = trimming_rule
        self._trimming_threshold = self._ps_processor.clipping_threshold

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError(
                "Normalization indicator has to be boolean. " + f"Object of type {str(type(self.normalize_ipw))} passed."
            )

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

        # APO weights
        _check_weights(weights, score="ATE", n_obs=obj_dml_data.n_obs, n_rep=self.n_rep)
        self._initialize_weights(weights)

        # perform sample splitting
        self._smpls = None
        self._n_obs_sample_splitting = self._dml_data.n_obs
        self._strata = self._dml_data.d
        if draw_sample_splitting:
            self.draw_sample_splitting()

            # initialize all models if splits are known
            self._initialize_dml_model()

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        fit_summary = str(self.summary)
        res = header + "\n------------------ Fit summary       ------------------\n" + fit_summary
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
    def ps_processor_config(self):
        """
        Configuration for propensity score processing (clipping, calibration, etc.).
        """
        return self._ps_processor_config

    @property
    def ps_processor(self):
        """
        Propensity score processor.
        """
        return self._ps_processor

    # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        warnings.warn(
            "'trimming_rule' is deprecated and will be removed in a future version. ", DeprecationWarning, stacklevel=2
        )
        return self._trimming_rule

    # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        warnings.warn(
            "'trimming_threshold' is deprecated and will be removed in a future version. "
            "Use 'ps_processor_config.clipping_threshold' or 'ps_processor.clipping_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ps_processor.clipping_threshold

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
            err_msg = (
                "Sample splitting not specified. Draw samples via .draw_sample_splitting(). "
                + "External samples not implemented yet."
            )
            raise NotImplementedError(err_msg)
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
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_summary = pd.DataFrame(columns=col_names)
        else:
            ci = self.confint()
            df_summary = generate_summary(self.coef, self.se, self.t_stat, self.pval, ci, self._treatment_levels)
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
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")
        fitted_models = parallel(
            delayed(self._fit_model)(i_level, n_jobs_cv, store_predictions, store_models, ext_pred_dict)
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
            raise ValueError("Apply fit() before confint().")

        df_ci = self.framework.confint(joint=joint, level=level)
        df_ci.set_index(pd.Index(self._treatment_levels), inplace=True)

        return df_ci

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
                f"benchmarking_set must be a list. {str(benchmarking_set)} of type {type(benchmarking_set)} was passed."
            )
        if len(benchmarking_set) == 0:
            raise ValueError("benchmarking_set must not be empty.")
        if not set(benchmarking_set) <= set(x_list_long):
            raise ValueError(
                f"benchmarking_set must be a subset of features {str(self._dml_data.x_cols)}. "
                f"{str(benchmarking_set)} was passed."
            )
        if fit_args is not None and not isinstance(fit_args, dict):
            raise TypeError(f"fit_args must be a dict. {str(fit_args)} of type {type(fit_args)} was passed.")

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

    def _initialize_dml_model(self):
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
            raise ValueError("Apply fit() before causal_contrast().")
        if self.n_treatment_levels == 1:
            raise ValueError("Only one treatment level. No causal contrast can be computed.")
        is_iterable = isinstance(reference_levels, Iterable)
        if not is_iterable:
            reference_levels = [reference_levels]
        is_treatment_level_subset = set(reference_levels).issubset(set(self.treatment_levels))
        if not is_treatment_level_subset:
            raise ValueError(
                "Invalid reference_levels. reference_levels has to be an iterable subset of treatment_levels or "
                "a single treatment level."
            )

        skip_index = set()
        all_treatment_names = []
        all_acc_frameworks = []

        for ref_lvl in reference_levels:
            i_ref_lvl = self.treatment_levels.index(ref_lvl)
            ref_model = self.modellist[i_ref_lvl]

            skip_index.add(i_ref_lvl)
            for i, model in enumerate(self.modellist):
                # only comparisons which are not yet computed
                if i in skip_index:
                    continue

                diff_framework = model.framework - ref_model.framework
                current_treatment_name = f"{self.treatment_levels[i]} vs {self.treatment_levels[i_ref_lvl]}"

                # update sensitivity elements with sharper bounds
                current_sensitivity_dict = self._compute_causal_contrast_sensitivity_dict(model=model, ref_model=ref_model)
                updated_dml_core = DoubleMLCore(
                    all_thetas=diff_framework.all_thetas,
                    all_ses=diff_framework.all_ses,
                    var_scaling_factors=diff_framework.var_scaling_factors,
                    scaled_psi=diff_framework.scaled_psi,
                    is_cluster_data=diff_framework.is_cluster_data,
                    cluster_dict=diff_framework.cluster_dict,
                    sensitivity_elements=current_sensitivity_dict["sensitivity_elements"],
                )
                current_framework = DoubleMLFramework(updated_dml_core, treatment_names=[current_treatment_name])

                all_acc_frameworks += [current_framework]
                all_treatment_names += [current_treatment_name]

        acc = concat(all_acc_frameworks)
        acc.treatment_names = all_treatment_names
        return acc

    def _fit_model(self, i_level, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions_dict=None):
        model = self.modellist[i_level]
        if external_predictions_dict is not None:
            external_predictions = external_predictions_dict[self.treatment_levels[i_level]]
        else:
            external_predictions = None
        model.fit(
            n_jobs_cv=n_jobs_cv,
            store_predictions=store_predictions,
            store_models=store_models,
            external_predictions=external_predictions,
        )
        return model

    def _compute_causal_contrast_sensitivity_dict(self, model, ref_model):
        # reshape sensitivity elements to (1 or n_obs, n_coefs, n_rep)
        model_sigma2 = np.transpose(model.sensitivity_elements["sigma2"], (0, 2, 1))
        model_nu2 = np.transpose(model.sensitivity_elements["nu2"], (0, 2, 1))
        model_psi_sigma2 = np.transpose(model.sensitivity_elements["psi_sigma2"], (0, 2, 1))
        model_psi_nu2 = np.transpose(model.sensitivity_elements["psi_nu2"], (0, 2, 1))

        ref_model_sigma2 = np.transpose(ref_model.sensitivity_elements["sigma2"], (0, 2, 1))
        ref_model_nu2 = np.transpose(ref_model.sensitivity_elements["nu2"], (0, 2, 1))
        ref_model_psi_sigma2 = np.transpose(ref_model.sensitivity_elements["psi_sigma2"], (0, 2, 1))
        ref_model_psi_nu2 = np.transpose(ref_model.sensitivity_elements["psi_nu2"], (0, 2, 1))

        combined_sensitivity_dict = {
            "sigma2": (model_sigma2 + ref_model_sigma2) / 2,
            "nu2": model_nu2 + ref_model_nu2,
            "psi_sigma2": (model_psi_sigma2 + ref_model_psi_sigma2) / 2,
            "psi_nu2": model_psi_nu2 + ref_model_psi_nu2,
        }

        max_bias, psi_max_bias = _compute_sensitivity_bias(**combined_sensitivity_dict)

        new_sensitvitiy_dict = {
            "sensitivity_elements": {
                "max_bias": max_bias,
                "psi_max_bias": psi_max_bias,
                "sigma2": combined_sensitivity_dict["sigma2"],
                "nu2": combined_sensitivity_dict["nu2"],
            }
        }

        return new_sensitvitiy_dict

    def _check_treatment_levels(self, treatment_levels):
        is_iterable = isinstance(treatment_levels, Iterable)
        if not is_iterable:
            treatment_level_list = [treatment_levels]
        else:
            treatment_level_list = [t_lvl for t_lvl in treatment_levels]
        is_d_subset = set(treatment_level_list).issubset(set(self._all_treatment_levels))
        if not is_d_subset:
            raise ValueError(
                "Invalid reference_levels. reference_levels has to be an iterable subset or "
                "a single element of the unique treatment levels in the data."
            )
        return treatment_level_list

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError("The data must be of DoubleMLData type.")
        if obj_dml_data.z is not None:
            raise ValueError("The data must not contain instrumental variables.")
        return

    def _check_external_predictions(self, external_predictions):
        expected_keys = self.treatment_levels
        if not isinstance(external_predictions, dict):
            raise TypeError(
                "external_predictions must be a dictionary. " + f"Object of type {type(external_predictions)} passed."
            )

        if not set(external_predictions.keys()).issubset(set(expected_keys)):
            raise ValueError(
                "external_predictions must be a subset of all treatment levels. "
                + f"Expected keys: {set(expected_keys)}. "
                + f"Passed keys: {set(external_predictions.keys())}."
            )

        expected_learner_keys = ["ml_g_d_lvl0", "ml_g_d_lvl1", "ml_m"]
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
        ext_pred_dict = {treatment_level: {d_col: {}} for treatment_level in self.treatment_levels}
        for treatment_level in self.treatment_levels:
            if "ml_g_d_lvl1" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]["ml_g_d_lvl1"] = external_predictions[treatment_level]["ml_g_d_lvl1"]
            if "ml_m" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]["ml_m"] = external_predictions[treatment_level]["ml_m"]
            if "ml_g_d_lvl0" in external_predictions[treatment_level]:
                ext_pred_dict[treatment_level][d_col]["ml_g_d_lvl0"] = external_predictions[treatment_level]["ml_g_d_lvl0"]

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
            "obj_dml_data": self._dml_data,
            "ml_g": self._learner["ml_g"],
            "ml_m": self._learner["ml_m"],
            "score": self.score,
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "weights": self.weights,
            "ps_processor_config": self.ps_processor_config,
            "normalize_ipw": self.normalize_ipw,
            "draw_sample_splitting": False,
        }
        for i_level in range(self.n_treatment_levels):
            # initialize models for all levels
            model = DoubleMLAPO(treatment_level=self._treatment_levels[i_level], **kwargs)

            # synchronize the sample splitting
            model.set_sample_splitting(all_smpls=self.smpls)
            modellist[i_level] = model

        return modellist

    def tune_ml_models(
        self,
        ml_param_space,
        scoring_methods=None,
        cv=5,
        set_as_params=True,
        return_tune_res=False,
        optuna_settings=None,
    ):
        """Hyperparameter-tuning for DoubleML models using Optuna."""

        tuning_kwargs = {
            "ml_param_space": ml_param_space,
            "scoring_methods": scoring_methods,
            "cv": cv,
            "set_as_params": set_as_params,
            "return_tune_res": return_tune_res,
            "optuna_settings": optuna_settings,
        }

        tune_res = [] if return_tune_res else None
        for model in self._modellist:
            res = model.tune_ml_models(**tuning_kwargs)
            if return_tune_res:
                tune_res.append(res[0])
        return tune_res if return_tune_res else self

    tune_ml_models.__doc__ = TUNE_ML_MODELS_DOC
