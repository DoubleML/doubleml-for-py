import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import is_classifier, is_regressor

from doubleml.data import DoubleMLClusterData, DoubleMLPanelData
from doubleml.data.base_data import DoubleMLBaseData
from doubleml.double_ml_framework import DoubleMLFramework
from doubleml.utils._checks import _check_external_predictions, _check_sample_splitting
from doubleml.utils._estimation import _aggregate_coefs_and_ses, _rmse, _set_external_predictions, _var_est
from doubleml.utils._sensitivity import _compute_sensitivity_bias
from doubleml.utils.gain_statistics import gain_statistics
from doubleml.utils.resampling import DoubleMLClusterResampling, DoubleMLResampling

_implemented_data_backends = ["DoubleMLData", "DoubleMLClusterData"]


class DoubleML(ABC):
    """Double Machine Learning."""

    def __init__(self, obj_dml_data, n_folds, n_rep, score, draw_sample_splitting):
        # check and pick up obj_dml_data
        if not isinstance(obj_dml_data, DoubleMLBaseData):
            raise TypeError(
                "The data must be of " + " or ".join(_implemented_data_backends) + " type. "
                f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            if obj_dml_data.n_cluster_vars > 2:
                raise NotImplementedError("Multi-way (n_ways > 2) clustering not yet implemented.")
            self._is_cluster_data = True
        self._is_panel_data = False
        if isinstance(obj_dml_data, DoubleMLPanelData):
            self._is_panel_data = True

        self._dml_data = obj_dml_data
        self._n_obs = self._dml_data.n_obs

        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # initialize learners and parameters which are set model specific
        self._learner = None
        self._params = None
        self._is_classifier = {}

        # initialize predictions and target to None which are only stored if method fit is called with store_predictions=True
        self._predictions = None
        self._nuisance_targets = None
        self._nuisance_loss = None

        # initialize models to None which are only stored if method fit is called with store_models=True
        self._models = None

        # initialize sensitivity elements to None (only available if implemented for the class
        self._sensitivity_implemented = False
        self._sensitivity_elements = None
        self._sensitivity_params = None

        # initialize external predictions
        self._external_predictions_implemented = False

        # check resampling specifications
        if not isinstance(n_folds, int):
            raise TypeError(
                f"The number of folds must be of int type. {str(n_folds)} of type {str(type(n_folds))} was passed."
            )
        if n_folds < 1:
            raise ValueError(f"The number of folds must be positive. {str(n_folds)} was passed.")

        if not isinstance(n_rep, int):
            raise TypeError(
                "The number of repetitions for the sample splitting must be of int type. "
                f"{str(n_rep)} of type {str(type(n_rep))} was passed."
            )
        if n_rep < 1:
            raise ValueError(f"The number of repetitions for the sample splitting must be positive. {str(n_rep)} was passed.")

        if not isinstance(draw_sample_splitting, bool):
            raise TypeError(f"draw_sample_splitting must be True or False. Got {str(draw_sample_splitting)}.")

        # set resampling specifications
        if self._is_cluster_data:
            self._n_folds_per_cluster = n_folds
            self._n_folds = n_folds**self._dml_data.n_cluster_vars
        else:
            self._n_folds = n_folds
        self._n_rep = n_rep
        self._score = score
        # default is no stratification
        self._strata = None

        # perform sample splitting
        self._smpls = None
        self._smpls_cluster = None
        self._n_obs_sample_splitting = self.n_obs
        if draw_sample_splitting:
            self.draw_sample_splitting()

        self._score_dim = (self._dml_data.n_obs, self.n_rep, self._dml_data.n_coefs)
        # initialize arrays according to obj_dml_data and the resampling settings
        self._initialize_arrays()

        # initialize instance attributes which are later used for iterating
        self._i_rep = None
        self._i_treat = None

    def _format_header_str(self):
        class_name = self.__class__.__name__
        return f"================== {class_name} Object =================="

    def _format_score_info_str(self):
        return f"Score function: {str(self.score)}"

    def _format_learner_info_str(self):
        learner_info = ""
        if self.learner is not None:
            for key, value in self.learner.items():
                learner_info += f"Learner {key}: {str(value)}\n"
        if self.nuisance_loss is not None:
            learner_info += "Out-of-sample Performance:\n"
            # Check if _is_classifier is populated, otherwise, it might be called before fit
            if self._is_classifier:
                is_classifier_any = any(self._is_classifier.values())
                is_regressor_any = any(not v for v in self._is_classifier.values())

                if is_regressor_any:
                    learner_info += "Regression:\n"
                    for learner_name in self.params_names:  # Iterate through known learners
                        if not self._is_classifier.get(learner_name, True):  # Default to not regressor if not found
                            loss_val = self.nuisance_loss.get(learner_name, "N/A")
                            learner_info += f"Learner {learner_name} RMSE: {loss_val}\n"
                if is_classifier_any:
                    learner_info += "Classification:\n"
                    for learner_name in self.params_names:  # Iterate through known learners
                        if self._is_classifier.get(learner_name, False):  # Default to not classifier if not found
                            loss_val = self.nuisance_loss.get(learner_name, "N/A")
                            learner_info += f"Learner {learner_name} Log Loss: {loss_val}\n"
            else:
                learner_info += " (Run .fit() to see out-of-sample performance)\n"
        return learner_info.strip()

    def _format_resampling_info_str(self):
        if self._is_cluster_data:
            return (
                f"No. folds per cluster: {self._n_folds_per_cluster}\n"
                f"No. folds: {self.n_folds}\n"
                f"No. repeated sample splits: {self.n_rep}"
            )
        else:
            return f"No. folds: {self.n_folds}\nNo. repeated sample splits: {self.n_rep}"

    def _format_additional_info_str(self):
        """
        Hook for subclasses to add additional information to the string representation.
        Returns an empty string by default.
        Subclasses should override this method to provide content.
        The content should not include the 'Additional Information' header itself.
        """
        return ""

    def __str__(self):
        header = self._format_header_str()
        # Assumes self._dml_data._data_summary_str() exists and is well-formed
        data_summary = self._dml_data._data_summary_str()
        score_info = self._format_score_info_str()
        learner_info = self._format_learner_info_str()
        resampling_info = self._format_resampling_info_str()
        fit_summary = str(self.summary)  # Assumes self.summary is well-formed

        representation = (
            f"{header}\n"
            f"\n------------------ Data Summary      ------------------\n"
            f"{data_summary}\n"
            f"\n------------------ Score & Algorithm ------------------\n"
            f"{score_info}\n"
            f"\n------------------ Machine Learner   ------------------\n"
            f"{learner_info}\n"
            f"\n------------------ Resampling        ------------------\n"
            f"{resampling_info}\n"
            f"\n------------------ Fit Summary       ------------------\n"
            f"{fit_summary}"
        )

        additional_info = self._format_additional_info_str()
        if additional_info:
            representation += f"\n\n------------------ Additional Information ------------------\n" f"{additional_info}"
        return representation

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
    def n_obs(self):
        """
        The number of observations used for estimation.
        """
        return self._n_obs

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
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

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
        The predictions of the nuisance models in form of a dictinary.
        Each key refers to a nuisance element with a array of values of shape ``(n_obs, n_rep, n_coefs)``.
        """
        return self._predictions

    @property
    def nuisance_targets(self):
        """
        The outcome of the nuisance models.
        """
        return self._nuisance_targets

    @property
    def nuisance_loss(self):
        """
        The losses of the nuisance models (root-mean-squared-errors or logloss).
        """
        return self._nuisance_loss

    @property
    def models(self):
        """
        The fitted nuisance models.
        """
        return self._models

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
            raise ValueError(
                "Invalid nuisance learner "
                + str(learner)
                + ". "
                + "Valid nuisance learner "
                + " or ".join(valid_learner)
                + "."
            )
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
            err_msg = (
                "Sample splitting not specified. Either draw samples via .draw_sample splitting() "
                + "or set external samples via .set_sample_splitting()."
            )
            raise ValueError(err_msg)
        return self._smpls

    @property
    def smpls_cluster(self):
        """
        The partition of clusters used for cross-fitting.
        """
        return self._smpls_cluster

    @property
    def psi(self):
        """
        Values of the score function after calling :meth:`fit`;
        For models (e.g., PLR, IRM, PLIV, IIVM) with linear score (in the parameter)
        :math:`\\psi(W; \\theta, \\eta) = \\psi_a(W; \\eta) \\theta + \\psi_b(W; \\eta)`.
        The shape is ``(n_obs, n_rep, n_coefs)``.
        """
        return self._psi

    @property
    def psi_deriv(self):
        """
        Values of the derivative of the score function with respect to the parameter :math:`\\theta`
        after calling :meth:`fit`;
        For models (e.g., PLR, IRM, PLIV, IIVM) with linear score (in the parameter)
        :math:`\\psi_a(W; \\eta)`.
        The shape is ``(n_obs, n_rep, n_coefs)``.
        """
        return self._psi_deriv

    @property
    def psi_elements(self):
        """
        Values of the score function components after calling :meth:`fit`;
        For models (e.g., PLR, IRM, PLIV, IIVM) with linear score (in the parameter) a dictionary with entries ``psi_a``
        and ``psi_b`` for :math:`\\psi_a(W; \\eta)` and :math:`\\psi_b(W; \\eta)`.
        """
        return self._psi_elements

    @property
    def sensitivity_elements(self):
        """
        Values of the sensitivity components after calling :meth:`fit`;
        If available (e.g., PLR, IRM) a dictionary with entries ``sigma2``, ``nu2``, ``psi_sigma2``, ``psi_nu2``
        and ``riesz_rep``.
        """
        return self._sensitivity_elements

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
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`.
        """
        if self._framework is None:
            boot_t_stat = None
        else:
            boot_t_stat = self._framework.boot_t_stat
        return boot_t_stat

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
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        col_names = ["coef", "std err", "t", "P>|t|"]
        if np.isnan(self.coef).all():
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = np.transpose(np.vstack([self.coef, self.se, self.t_stat, self.pval]))
            df_summary = pd.DataFrame(summary_stats, columns=col_names, index=self._dml_data.d_cols)
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
    def __psi_deriv(self):
        return self._psi_deriv[:, self._i_rep, self._i_treat]

    @property
    def __all_se(self):
        return self._all_se[self._i_treat, self._i_rep]

    def fit(self, n_jobs_cv=None, store_predictions=True, external_predictions=None, store_models=False):
        """
        Estimate DoubleML models.

        Parameters
        ----------
        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        store_predictions : bool
            Indicates whether the predictions for the nuisance functions should be stored in ``predictions``.
            Default is ``True``.

        store_models : bool
            Indicates whether the fitted models for the nuisance functions should be stored in ``models``. This allows
            to analyze the fitted models or extract information like variable importance.
            Default is ``False``.

        external_predictions : None or dict
            If `None` all models for the learners are fitted and evaluated. If a dictionary containing predictions
            for a specific learner is supplied, the model will use the supplied nuisance predictions instead. Has to
            be a nested dictionary where the keys refer to the treatment and the keys of the nested dictionarys refer to the
            corresponding learners.
            Default is `None`.

        Returns
        -------
        self : object
        """

        self._check_fit(n_jobs_cv, store_predictions, external_predictions, store_models)
        self._initalize_fit(store_predictions, store_models)

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                # this step could be skipped for the single treatment variable case
                if self._dml_data.n_treat > 1:
                    self._dml_data.set_x_d(self._dml_data.d_cols[i_d])

                # predictions have to be stored in loop for sensitivity analysis
                nuisance_predictions = self._fit_nuisance_and_score_elements(
                    n_jobs_cv, store_predictions, external_predictions, store_models
                )

                self._solve_score_and_estimate_se()

                # sensitivity elements can depend on the estimated parameter
                self._fit_sensitivity_elements(nuisance_predictions)

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self.coef, self.se = _aggregate_coefs_and_ses(self._all_coef, self._all_se)

        # validate sensitivity elements (e.g., re-estimate nu2 if negative)
        self._validate_sensitivity_elements()

        # construct framework for inference
        self._framework = self.construct_framework()

        return self

    def construct_framework(self):
        """
        Construct a :class:`doubleml.DoubleMLFramework` object. Can be used to construct e.g. confidence intervals.

        Parameters
        ----------

        Returns
        -------
        doubleml_framework : doubleml.DoubleMLFramework
        """
        # standardize the score function and reshape to (n_obs, n_coefs, n_rep)
        scaled_psi = np.divide(self.psi, np.mean(self.psi_deriv, axis=0))
        scaled_psi_reshape = np.transpose(scaled_psi, (0, 2, 1))

        doubleml_dict = {
            "thetas": self.coef,
            "all_thetas": self.all_coef,
            "ses": self.se,
            "all_ses": self.all_se,
            "var_scaling_factors": self._var_scaling_factors,
            "scaled_psi": scaled_psi_reshape,
            "is_cluster_data": self._is_cluster_data,
            "treatment_names": self._dml_data.d_cols,
        }

        if self._sensitivity_implemented:
            # reshape sensitivity elements to (1 or n_obs, n_coefs, n_rep)
            sensitivity_dict = {
                "sigma2": np.transpose(self.sensitivity_elements["sigma2"], (0, 2, 1)),
                "nu2": np.transpose(self.sensitivity_elements["nu2"], (0, 2, 1)),
                "psi_sigma2": np.transpose(self.sensitivity_elements["psi_sigma2"], (0, 2, 1)),
                "psi_nu2": np.transpose(self.sensitivity_elements["psi_nu2"], (0, 2, 1)),
            }

            max_bias, psi_max_bias = _compute_sensitivity_bias(**sensitivity_dict)

            doubleml_dict.update(
                {
                    "sensitivity_elements": {
                        "max_bias": max_bias,
                        "psi_max_bias": psi_max_bias,
                        "sigma2": sensitivity_dict["sigma2"],
                        "nu2": sensitivity_dict["nu2"],
                    }
                }
            )

        if self._is_cluster_data:
            doubleml_dict.update(
                {
                    "is_cluster_data": True,
                    "cluster_dict": {
                        "smpls": self._smpls,
                        "smpls_cluster": self._smpls_cluster,
                        "cluster_vars": self._dml_data.cluster_vars,
                        "n_folds_per_cluster": self._n_folds_per_cluster,
                    },
                }
            )

        doubleml_framework = DoubleMLFramework(doubleml_dict)
        return doubleml_framework

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
        df_ci.set_index(pd.Index(self._dml_data.d_cols), inplace=True)

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
        p_val.set_index(pd.Index(self._dml_data.d_cols), inplace=True)

        return p_val

    def tune(
        self,
        param_grids,
        tune_on_folds=False,
        scoring_methods=None,  # if None the estimator's score method is used
        n_folds_tune=5,
        search_mode="grid_search",
        n_iter_randomized_search=100,
        n_jobs_cv=None,
        set_as_params=True,
        return_tune_res=False,
    ):
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
            raise ValueError(
                "Invalid param_grids " + str(param_grids) + ". "
                "param_grids must be a dictionary with keys " + " and ".join(self.learner_names) + "."
            )

        if scoring_methods is not None:
            if (not isinstance(scoring_methods, dict)) | (not all(k in self.learner_names for k in scoring_methods)):
                raise ValueError(
                    "Invalid scoring_methods "
                    + str(scoring_methods)
                    + ". "
                    + "scoring_methods must be a dictionary. "
                    + "Valid keys are "
                    + " and ".join(self.learner_names)
                    + "."
                )
            if not all(k in scoring_methods for k in self.learner_names):
                # if there are learners for which no scoring_method was set, we fall back to None, i.e., default scoring
                for learner in self.learner_names:
                    if learner not in scoring_methods:
                        scoring_methods[learner] = None

        if not isinstance(tune_on_folds, bool):
            raise TypeError(f"tune_on_folds must be True or False. Got {str(tune_on_folds)}.")

        if not isinstance(n_folds_tune, int):
            raise TypeError(
                "The number of folds used for tuning must be of int type. "
                f"{str(n_folds_tune)} of type {str(type(n_folds_tune))} was passed."
            )
        if n_folds_tune < 2:
            raise ValueError(f"The number of folds used for tuning must be at least two. {str(n_folds_tune)} was passed.")

        if (not isinstance(search_mode, str)) | (search_mode not in ["grid_search", "randomized_search"]):
            raise ValueError(f'search_mode must be "grid_search" or "randomized_search". Got {str(search_mode)}.')

        if not isinstance(n_iter_randomized_search, int):
            raise TypeError(
                "The number of parameter settings sampled for the randomized search must be of int type. "
                f"{str(n_iter_randomized_search)} of type "
                f"{str(type(n_iter_randomized_search))} was passed."
            )
        if n_iter_randomized_search < 2:
            raise ValueError(
                "The number of parameter settings sampled for the randomized search must be at least two. "
                f"{str(n_iter_randomized_search)} was passed."
            )

        if n_jobs_cv is not None:
            if not isinstance(n_jobs_cv, int):
                raise TypeError(
                    "The number of CPUs used to fit the learners must be of int type. "
                    f"{str(n_jobs_cv)} of type {str(type(n_jobs_cv))} was passed."
                )

        if not isinstance(set_as_params, bool):
            raise TypeError(f"set_as_params must be True or False. Got {str(set_as_params)}.")

        if not isinstance(return_tune_res, bool):
            raise TypeError(f"return_tune_res must be True or False. Got {str(return_tune_res)}.")

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
                    res = self._nuisance_tuning(
                        self.__smpls,
                        param_grids,
                        scoring_methods,
                        n_folds_tune,
                        n_jobs_cv,
                        search_mode,
                        n_iter_randomized_search,
                    )

                    tuning_res[i_rep][i_d] = res
                    nuisance_params.append(res["params"])

                if set_as_params:
                    for nuisance_model in nuisance_params[0].keys():
                        params = [x[nuisance_model] for x in nuisance_params]
                        self.set_ml_nuisance_params(nuisance_model, self._dml_data.d_cols[i_d], params)

            else:
                smpls = [(np.arange(self.n_obs), np.arange(self.n_obs))]
                # tune hyperparameters
                res = self._nuisance_tuning(
                    smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
                )
                tuning_res[i_d] = res

                if set_as_params:
                    for nuisance_model in res["params"].keys():
                        params = res["params"][nuisance_model]
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
            raise ValueError(
                "Invalid nuisance learner " + learner + ". " + "Valid nuisance learner " + " or ".join(valid_learner) + "."
            )

        if treat_var not in self._dml_data.d_cols:
            raise ValueError(
                "Invalid treatment variable "
                + treat_var
                + ". "
                + "Valid treatment variable "
                + " or ".join(self._dml_data.d_cols)
                + "."
            )

        if params is None:
            all_params = [None] * self.n_rep
        elif isinstance(params, dict):
            all_params = [[params] * self.n_folds] * self.n_rep

        else:
            # ToDo: Add meaningful error message for asserts and corresponding uni tests
            assert len(params) == self.n_rep
            assert np.all(np.array([len(x) for x in params]) == self.n_folds)
            all_params = params

        self._params[learner][treat_var] = all_params

        return self

    @abstractmethod
    def _initialize_ml_nuisance_params(self):
        pass

    @abstractmethod
    def _nuisance_est(self, smpls, n_jobs_cv, return_models, external_predictions):
        pass

    @abstractmethod
    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        pass

    @staticmethod
    def _check_learner(learner, learner_name, regressor, classifier):
        err_msg_prefix = f"Invalid learner provided for {learner_name}: "
        warn_msg_prefix = f"Learner provided for {learner_name} is probably invalid: "

        if isinstance(learner, type):
            raise TypeError(err_msg_prefix + "provide an instance of a learner instead of a class.")

        if not hasattr(learner, "fit"):
            raise TypeError(err_msg_prefix + f"{str(learner)} has no method .fit().")
        if not hasattr(learner, "set_params"):
            raise TypeError(err_msg_prefix + f"{str(learner)} has no method .set_params().")
        if not hasattr(learner, "get_params"):
            raise TypeError(err_msg_prefix + f"{str(learner)} has no method .get_params().")

        if regressor & classifier:
            if is_classifier(learner):
                learner_is_classifier = True
            elif is_regressor(learner):
                learner_is_classifier = False
            else:
                warnings.warn(
                    warn_msg_prefix
                    + f"{str(learner)} is (probably) neither a regressor nor a classifier. "
                    + "Method predict is used for prediction."
                )
                learner_is_classifier = False
        elif classifier:
            if not is_classifier(learner):
                warnings.warn(warn_msg_prefix + f"{str(learner)} is (probably) no classifier.")
            learner_is_classifier = True
        else:
            assert regressor  # classifier, regressor or both must be True
            if not is_regressor(learner):
                warnings.warn(warn_msg_prefix + f"{str(learner)} is (probably) no regressor.")
            learner_is_classifier = False

        # check existence of the prediction method
        if learner_is_classifier:
            if not hasattr(learner, "predict_proba"):
                raise TypeError(err_msg_prefix + f"{str(learner)} has no method .predict_proba().")
        else:
            if not hasattr(learner, "predict"):
                raise TypeError(err_msg_prefix + f"{str(learner)} has no method .predict().")

        return learner_is_classifier

    def _check_fit(self, n_jobs_cv, store_predictions, external_predictions, store_models):
        if n_jobs_cv is not None:
            if not isinstance(n_jobs_cv, int):
                raise TypeError(
                    "The number of CPUs used to fit the learners must be of int type. "
                    f"{str(n_jobs_cv)} of type {str(type(n_jobs_cv))} was passed."
                )

        if not isinstance(store_predictions, bool):
            raise TypeError(f"store_predictions must be True or False. Got {str(store_predictions)}.")

        if not isinstance(store_models, bool):
            raise TypeError(f"store_models must be True or False. Got {str(store_models)}.")

        # check if external predictions are implemented
        if self._external_predictions_implemented:
            _check_external_predictions(
                external_predictions=external_predictions,
                valid_treatments=self._dml_data.d_cols,
                valid_learners=self.params_names,
                n_obs=self.n_obs,
                n_rep=self.n_rep,
            )
        elif not self._external_predictions_implemented and external_predictions is not None:
            raise NotImplementedError(f"External predictions not implemented for {self.__class__.__name__}.")

    def _initalize_fit(self, store_predictions, store_models):
        # initialize loss arrays for nuisance functions evaluation
        self._initialize_nuisance_loss()

        if store_predictions:
            self._initialize_predictions_and_targets()

        if store_models:
            self._initialize_models()

        if self._sensitivity_implemented:
            self._sensitivity_elements = self._initialize_sensitivity_elements(self._score_dim)

    def _fit_nuisance_and_score_elements(self, n_jobs_cv, store_predictions, external_predictions, store_models):
        ext_prediction_dict = _set_external_predictions(
            external_predictions, learners=self.params_names, treatment=self._dml_data.d_cols[self._i_treat], i_rep=self._i_rep
        )

        # ml estimation of nuisance models and computation of score elements
        score_elements, preds = self._nuisance_est(
            self.__smpls, n_jobs_cv, external_predictions=ext_prediction_dict, return_models=store_models
        )

        self._set_score_elements(score_elements, self._i_rep, self._i_treat)

        # calculate nuisance losses and store predictions and targets of the nuisance models
        self._calc_nuisance_loss(preds["predictions"], preds["targets"])
        if store_predictions:
            self._store_predictions_and_targets(preds["predictions"], preds["targets"])
        if store_models:
            self._store_models(preds["models"])

        return preds

    def _solve_score_and_estimate_se(self):
        # estimate the causal parameter
        self._all_coef[self._i_treat, self._i_rep] = self._est_causal_pars(
            self._get_score_elements(self._i_rep, self._i_treat)
        )

        # compute score (depends on the estimated causal parameter)
        self._psi[:, self._i_rep, self._i_treat] = self._compute_score(
            self._get_score_elements(self._i_rep, self._i_treat), self._all_coef[self._i_treat, self._i_rep]
        )

        # compute score derivative (can depend on the estimated causal parameter)
        self._psi_deriv[:, self._i_rep, self._i_treat] = self._compute_score_deriv(
            self._get_score_elements(self._i_rep, self._i_treat), self._all_coef[self._i_treat, self._i_rep]
        )

        # compute standard errors for causal parameter
        self._all_se[self._i_treat, self._i_rep], self._var_scaling_factors[self._i_treat] = self._se_causal_pars()

    def _fit_sensitivity_elements(self, nuisance_predictions):
        if self._sensitivity_implemented:
            if callable(self.score):
                warnings.warn("Sensitivity analysis not implemented for callable scores.")
            else:
                # compute sensitivity analysis elements
                element_dict = self._sensitivity_element_est(nuisance_predictions)
                self._set_sensitivity_elements(element_dict, self._i_rep, self._i_treat)

    def _initialize_arrays(self):
        # scores
        self._psi = np.full(self._score_dim, np.nan)
        self._psi_deriv = np.full(self._score_dim, np.nan)
        self._psi_elements = self._initialize_score_elements(self._score_dim)

        n_rep = self._score_dim[1]
        n_thetas = self._score_dim[2]

        self._var_scaling_factors = np.full(n_thetas, np.nan)
        # coefficients and ses
        self._coef = np.full(n_thetas, np.nan)
        self._se = np.full(n_thetas, np.nan)

        self._all_coef = np.full((n_thetas, n_rep), np.nan)
        self._all_se = np.full((n_thetas, n_rep), np.nan)

    def _initialize_predictions_and_targets(self):
        self._predictions = {learner: np.full(self._score_dim, np.nan) for learner in self.params_names}
        self._nuisance_targets = {learner: np.full(self._score_dim, np.nan) for learner in self.params_names}

    def _initialize_nuisance_loss(self):
        self._nuisance_loss = {learner: np.full((self.n_rep, self._dml_data.n_coefs), np.nan) for learner in self.params_names}

    def _initialize_models(self):
        self._models = {
            learner: {treat_var: [None] * self.n_rep for treat_var in self._dml_data.d_cols} for learner in self.params_names
        }

    def _store_predictions_and_targets(self, preds, targets):
        for learner in self.params_names:
            self._predictions[learner][:, self._i_rep, self._i_treat] = preds[learner]
            self._nuisance_targets[learner][:, self._i_rep, self._i_treat] = targets[learner]

    def _calc_nuisance_loss(self, preds, targets):
        self._is_classifier = {key: False for key in self.params_names}
        for learner in self.params_names:
            # check if the learner is a classifier
            learner_keys = [key for key in self._learner.keys() if key in learner]
            assert len(learner_keys) == 1
            self._is_classifier[learner] = self._check_learner(
                self._learner[learner_keys[0]], learner, regressor=True, classifier=True
            )

            if targets[learner] is None:
                self._nuisance_loss[learner][self._i_rep, self._i_treat] = np.nan
            else:
                learner_keys = [key for key in self._learner.keys() if key in learner]
                assert len(learner_keys) == 1

                if self._is_classifier[learner]:
                    predictions = np.clip(preds[learner], 1e-15, 1 - 1e-15)
                    logloss = targets[learner] * np.log(predictions) + (1 - targets[learner]) * np.log(1 - predictions)
                    loss = -np.nanmean(logloss, axis=0)
                else:
                    sq_error = np.power(targets[learner] - preds[learner], 2)
                    loss = np.sqrt(np.nanmean(sq_error, axis=0))

                self._nuisance_loss[learner][self._i_rep, self._i_treat] = loss

    def _store_models(self, models):
        for learner in self.params_names:
            self._models[learner][self._dml_data.d_cols[self._i_treat]][self._i_rep] = models[learner]

    def evaluate_learners(self, learners=None, metric=_rmse):
        """
        Evaluate fitted learners for DoubleML models on cross-validated predictions.

        Parameters
        ----------
        learners : list
            A list of strings which correspond to the nuisance functions of the model.

        metric : callable
            A callable function with inputs ``y_pred`` and ``y_true`` of shape ``(1, n)``,
            where ``n`` specifies the number of observations. Remark that some models like IRM are
            not able to provide all values for ``y_true`` for all learners and might contain
            some ``nan`` values in the target vector.
            Default is the root-mean-square error.

        Returns
        -------
        dist : dict
            A dictionary containing the evaluated metric for each learner.

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from sklearn.metrics import mean_absolute_error
        >>> from doubleml.datasets import make_irm_data
        >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        >>> np.random.seed(3141)
        >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
        >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
        >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
        >>> dml_irm_obj.fit()
        >>> def mae(y_true, y_pred):
        >>>     subset = np.logical_not(np.isnan(y_true))
        >>>     return mean_absolute_error(y_true[subset], y_pred[subset])
        >>> dml_irm_obj.evaluate_learners(metric=mae)
        {'ml_g0': array([[0.85974356]]),
         'ml_g1': array([[0.85280376]]),
         'ml_m': array([[0.35365143]])}
        """
        # if no learners are provided try to evaluate all learners
        if learners is None:
            learners = self.params_names

        # check metric
        if not callable(metric):
            raise TypeError("metric should be a callable. %r was passed." % metric)

        if all(learner in self.params_names for learner in learners):
            if self.nuisance_targets is None:
                raise ValueError("Apply fit() before evaluate_learners().")
            else:
                dist = {learner: np.full((self.n_rep, self._dml_data.n_coefs), np.nan) for learner in learners}
            for learner in learners:
                for rep in range(self.n_rep):
                    for coef_idx in range(self._dml_data.n_coefs):
                        res = metric(
                            y_pred=self.predictions[learner][:, rep, coef_idx].reshape(1, -1),
                            y_true=self.nuisance_targets[learner][:, rep, coef_idx].reshape(1, -1),
                        )
                        if not np.isfinite(res):
                            raise ValueError(f"Evaluation from learner {str(learner)} is not finite.")
                        dist[learner][rep, coef_idx] = res
            return dist
        else:
            raise ValueError(
                f"The learners have to be a subset of {str(self.params_names)}. Learners {str(learners)} provided."
            )

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """
        if self._is_cluster_data:
            obj_dml_resampling = DoubleMLClusterResampling(
                n_folds=self._n_folds_per_cluster,
                n_rep=self.n_rep,
                n_obs=self._n_obs_sample_splitting,
                n_cluster_vars=self._dml_data.n_cluster_vars,
                cluster_vars=self._dml_data.cluster_vars,
            )
            self._smpls, self._smpls_cluster = obj_dml_resampling.split_samples()
        else:
            obj_dml_resampling = DoubleMLResampling(
                n_folds=self.n_folds, n_rep=self.n_rep, n_obs=self._n_obs_sample_splitting, stratify=self._strata
            )
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

        all_smpls_cluster : list or None
            Nested list or ``None``. The first level of nesting corresponds to the number of repetitions. The second level
            of nesting corresponds to the number of folds. The third level of nesting contains a tuple of training and
            testing lists. Both training and testing contain an array for each cluster variable, which form a partition of
            the clusters.
            Default is ``None``.

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
        self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
            all_smpls, all_smpls_cluster, self._dml_data, self._is_cluster_data, n_obs=self._n_obs_sample_splitting
        )

        # set sample splitting can update the number of repetitions
        self._score_dim = (self._score_dim[0], self._n_rep, self._score_dim[2])
        self._initialize_arrays()
        self._initialize_ml_nuisance_params()

        return self

    def _est_causal_pars(self, psi_elements):
        smpls = self.__smpls

        if not self._is_cluster_data:
            coef = self._est_coef(psi_elements)
        else:
            scaling_factor = [1.0] * len(smpls)
            for i_fold, (_, _) in enumerate(smpls):
                test_cluster_inds = self.__smpls_cluster[i_fold][1]
                scaling_factor[i_fold] = 1.0 / np.prod(np.array([len(inds) for inds in test_cluster_inds]))
            coef = self._est_coef(psi_elements, smpls=smpls, scaling_factor=scaling_factor)

        return coef

    def _se_causal_pars(self):
        if not self._is_cluster_data:
            cluster_vars = None
            smpls_cluster = None
            n_folds_per_cluster = None

        else:
            cluster_vars = self._dml_data.cluster_vars
            smpls_cluster = self.__smpls_cluster
            n_folds_per_cluster = self._n_folds_per_cluster

        sigma2_hat, var_scaling_factor = _var_est(
            psi=self.__psi,
            psi_deriv=self.__psi_deriv,
            smpls=self.__smpls,
            is_cluster_data=self._is_cluster_data,
            cluster_vars=cluster_vars,
            smpls_cluster=smpls_cluster,
            n_folds_per_cluster=n_folds_per_cluster,
        )

        se = np.sqrt(sigma2_hat)
        return se, var_scaling_factor

    # to estimate causal parameters without predictions
    def _est_causal_pars_and_se(self):
        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            for i_d in range(self._dml_data.n_treat):
                self._i_treat = i_d

                # estimate the causal parameter
                self._all_coef[self._i_treat, self._i_rep] = self._est_causal_pars(
                    self._get_score_elements(self._i_rep, self._i_treat)
                )

                # compute score (depends on the estimated causal parameter)
                self._psi[:, self._i_rep, self._i_treat] = self._compute_score(
                    self._get_score_elements(self._i_rep, self._i_treat), self._all_coef[self._i_treat, self._i_rep]
                )

                # compute score (can depend on the estimated causal parameter)
                self._psi_deriv[:, self._i_rep, self._i_treat] = self._compute_score_deriv(
                    self._get_score_elements(self._i_rep, self._i_treat), self._all_coef[self._i_treat, self._i_rep]
                )

                # compute standard errors for causal parameter
                self._all_se[self._i_treat, self._i_rep], self._var_scaling_factors[self._i_treat] = self._se_causal_pars()

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self.coef, self.se = _aggregate_coefs_and_ses(self._all_coef, self._all_se)

    # Score estimation and elements
    @abstractmethod
    def _est_coef(self, psi_elements, smpls=None, scaling_factor=None, inds=None):
        pass

    @property
    @abstractmethod
    def _score_element_names(self):
        pass

    @abstractmethod
    def _compute_score(self, psi_elements, coef):
        pass

    @abstractmethod
    def _compute_score_deriv(self, psi_elements, coef):
        pass

    def _get_score_elements(self, i_rep, i_treat):
        psi_elements = {key: value[:, i_rep, i_treat] for key, value in self.psi_elements.items()}
        return psi_elements

    def _set_score_elements(self, psi_elements, i_rep, i_treat):
        if not isinstance(psi_elements, dict):
            raise TypeError(
                f"_ml_nuisance_and_score_elements must return score elements in a dict. Got type {str(type(psi_elements))}."
            )
        if not (set(self._score_element_names) == set(psi_elements.keys())):
            raise ValueError(
                "_ml_nuisance_and_score_elements returned incomplete score elements. "
                "Expected dict with keys: " + " and ".join(set(self._score_element_names)) + "."
                "Got dict with keys: " + " and ".join(set(psi_elements.keys())) + "."
            )
        for key in self._score_element_names:
            self.psi_elements[key][:, i_rep, i_treat] = psi_elements[key]
        return

    def _initialize_score_elements(self, score_dim):
        psi_elements = {key: np.full(score_dim, np.nan) for key in self._score_element_names}
        return psi_elements

    # Sensitivity estimation and elements
    @abstractmethod
    def _sensitivity_element_est(self, preds):
        pass

    @property
    def _sensitivity_element_names(self):
        return ["sigma2", "nu2", "psi_sigma2", "psi_nu2", "riesz_rep"]

    # the dimensions will usually be (n_obs, n_rep, n_coefs) to be equal to the score dimensions psi
    def _initialize_sensitivity_elements(self, score_dim):
        sensitivity_elements = {
            "sigma2": np.full((1, score_dim[1], score_dim[2]), np.nan),
            "nu2": np.full((1, score_dim[1], score_dim[2]), np.nan),
            "psi_sigma2": np.full(score_dim, np.nan),
            "psi_nu2": np.full(score_dim, np.nan),
            "riesz_rep": np.full(score_dim, np.nan),
        }
        return sensitivity_elements

    def _validate_sensitivity_elements(self):
        if self._sensitivity_implemented:
            for i_treat in range(self._dml_data.n_treat):
                nu2 = self.sensitivity_elements["nu2"][:, :, i_treat]
                riesz_rep = self.sensitivity_elements["riesz_rep"][:, :, i_treat]

                if np.any(nu2 <= 0):
                    treatment_name = self._dml_data.d_cols[i_treat]
                    msg = (
                        f"The estimated nu2 for {treatment_name} is not positive. "
                        "Re-estimation based on riesz representer (non-orthogonal)."
                    )
                    warnings.warn(msg, UserWarning)
                    psi_nu2 = np.power(riesz_rep, 2)
                    nu2 = np.mean(psi_nu2, axis=0, keepdims=True)

                    self.sensitivity_elements["nu2"][:, :, i_treat] = nu2
                    self.sensitivity_elements["psi_nu2"][:, :, i_treat] = psi_nu2

        return

    def _get_sensitivity_elements(self, i_rep, i_treat):
        sensitivity_elements = {key: value[:, i_rep, i_treat] for key, value in self.sensitivity_elements.items()}
        return sensitivity_elements

    def _set_sensitivity_elements(self, sensitivity_elements, i_rep, i_treat):
        if not isinstance(sensitivity_elements, dict):
            raise TypeError(
                "_sensitivity_element_est must return sensitivity elements in a dict. "
                f"Got type {str(type(sensitivity_elements))}."
            )
        if not (set(self._sensitivity_element_names) == set(sensitivity_elements.keys())):
            raise ValueError(
                "_sensitivity_element_est returned incomplete sensitivity elements. "
                "Expected dict with keys: " + " and ".join(set(self._sensitivity_element_names)) + ". "
                "Got dict with keys: " + " and ".join(set(sensitivity_elements.keys())) + "."
            )
        for key in self._sensitivity_element_names:
            self.sensitivity_elements[key][:, i_rep, i_treat] = sensitivity_elements[key]
        return

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

        Parameters
        ----------
        benchmarking_set : list
            List of features to be used for benchmarking.

        fit_args : dict, optional
            Additional arguments for the fit method.
            Default is None.

        Returns
        -------
        benchmark_results : pandas.DataFrame
            Benchmark results.
        """
        x_list_long = self._dml_data.x_cols

        # input checks
        if self._sensitivity_elements is None:
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
        df_benchmark = pd.DataFrame(benchmark_dict, index=self._dml_data.d_cols)
        return df_benchmark
