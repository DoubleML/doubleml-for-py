"""Abstract base class for multi-treatment DoubleML models (parameter vector estimation)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from .utils._tune_optuna import DMLOptunaResult

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data.base_data import DoubleMLData
from .double_ml_base import DoubleMLBase
from .double_ml_framework import concat
from .double_ml_scalar import DoubleMLScalar
from .utils._checks import _check_sample_splitting
from .utils._tune_optuna import TUNE_ML_MODELS_DOC
from .utils.gain_statistics import gain_statistics
from .utils.resampling import DoubleMLResampling


class DoubleMLVector(DoubleMLBase, ABC):
    """
    Abstract base class for multi-treatment DoubleML models.

    Orchestrates multiple :class:`~doubleml.DoubleMLScalar` instances — one per
    treatment column in ``d_cols`` — sharing a single set of sample splits and
    concatenating their :class:`~doubleml.DoubleMLFramework` objects into one
    unified result.

    This class is intentionally general: by overriding :meth:`_initialize_models`
    (and optionally :meth:`_get_data_for_model`), concrete subclasses can cover
    any scenario where multiple scalar models must be fitted and combined:

    * **Multiple treatment columns** (e.g., ``DoubleMLPLRVector``): each sub-model
      receives a single-column data view created by :meth:`_get_data_for_model`.
    * **Multiple treatment levels** (e.g., a future ``DoubleMLAPOSVector``): all
      sub-models share the same data; each scalar carries its own ``treatment_level``
      parameter.  Override :meth:`_get_data_for_model` to return ``self._dml_data``
      unchanged, or bypass it entirely inside :meth:`_initialize_models`.

    Parameters
    ----------
    obj_dml_data : DoubleMLBaseData
        The data object for the double machine learning model.
    score : str, optional
        The score function to use. Default is ``'default'``.

    Attributes
    ----------
    n_folds : int
        Number of cross-fitting folds.
    n_rep : int
        Number of sample-splitting repetitions.
    score : str
        The score function being used.
    modellist : list of DoubleMLScalar
        The scalar sub-models, one per treatment column (or model key).
    """

    def __init__(
        self,
        obj_dml_data: DoubleMLData,
        score: str = "default",
    ) -> None:
        super().__init__(obj_dml_data)
        self._dml_data: DoubleMLData = obj_dml_data  # narrow for attribute access
        self._score = score

        # Sample-splitting state
        self._n_folds: int | None = None
        self._n_folds_per_cluster: int | None = None
        self._n_rep: int | None = None
        self._smpls: list | None = None
        self._smpls_cluster: list | None = None

        # Sub-model list — populated by subclass via _initialize_models()
        self._modellist: list[DoubleMLScalar] | None = None

    # ==================== Properties ====================

    @property
    def n_rep(self) -> int:
        """
        Number of repetitions for sample splitting.

        Returns
        -------
        int
            Number of repetitions.

        Raises
        ------
        ValueError
            If sample splitting has not been drawn yet.
        """
        if self._n_rep is None:
            raise ValueError("n_rep not set. Call draw_sample_splitting() first.")
        return self._n_rep

    @property
    def n_folds(self) -> int:
        """
        Number of folds for cross-fitting.

        Returns
        -------
        int
            Number of folds.

        Raises
        ------
        ValueError
            If sample splitting has not been drawn yet.
        """
        if self._n_folds is None:
            raise ValueError("n_folds not set. Call draw_sample_splitting() first.")
        return self._n_folds

    @property
    def score(self) -> str:
        """
        The score function being used.

        Returns
        -------
        str
            Score function name.
        """
        return self._score

    @property
    def smpls(self) -> list:
        """
        Sample-splitting indices used for cross-fitting.

        Returns
        -------
        list
            List of sample-splitting indices for each repetition.

        Raises
        ------
        ValueError
            If sample splitting has not been drawn yet.
        """
        if self._smpls is None:
            raise ValueError("Sample splitting has not been performed. Call draw_sample_splitting() first.")
        return self._smpls

    @property
    def modellist(self) -> list[DoubleMLScalar] | None:
        """
        The scalar sub-models in the same order as ``d_cols``.

        Returns
        -------
        list of DoubleMLScalar or None
            ``None`` before :meth:`_initialize_models` has been called by the subclass.
        """
        return self._modellist

    @property
    def n_rep_boot(self) -> int | None:
        """
        The number of bootstrap replications, or ``None`` if not bootstrapped.

        Returns
        -------
        int or None
        """
        return None if self._framework is None else self._framework.n_rep_boot

    @property
    def boot_method(self) -> str | None:
        """
        The bootstrap method used, or ``None`` if not bootstrapped.

        Returns
        -------
        str or None
        """
        return None if self._framework is None else self._framework.boot_method

    @property
    def boot_t_stat(self) -> np.ndarray | None:
        """
        Bootstrapped t-statistics, or ``None`` if not bootstrapped.

        Returns
        -------
        np.ndarray or None
        """
        return None if self._framework is None else self._framework.boot_t_stat

    @property
    def sensitivity_elements(self) -> dict[str, np.ndarray] | None:
        """
        Raw sensitivity elements after :meth:`fit`, or ``None`` if unavailable.

        Returns
        -------
        dict or None
        """
        return None if self._framework is None else self._framework.sensitivity_elements

    @property
    def sensitivity_params(self) -> dict | None:
        """
        Sensitivity analysis parameters after :meth:`sensitivity_analysis`,
        or ``None`` if not yet computed.

        Returns
        -------
        dict or None
        """
        return None if self._framework is None else self._framework.sensitivity_params

    @property
    def sensitivity_summary(self) -> str:
        """
        Summary for the sensitivity analysis after :meth:`sensitivity_analysis`.

        Returns
        -------
        str

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.
        """
        if self._framework is None:
            raise ValueError("Apply fit() before accessing sensitivity_summary.")
        return self._framework.sensitivity_summary

    # ==================== Abstract Methods ====================

    @property
    @abstractmethod
    def required_learners(self) -> list[str]:
        """
        Names of the required learners for the current configuration.

        Returns
        -------
        list of str
            Ordered list of required learner names.
        """

    @abstractmethod
    def set_learners(self, **kwargs: object) -> Self:
        """
        Set the learners for nuisance estimation on all sub-models.

        Subclasses must implement this method with explicit keyword arguments
        matching their model's learners (e.g., ``ml_l``, ``ml_m`` for PLR).
        The same learners (cloned per sub-model) are applied to every treatment.

        Parameters
        ----------
        **kwargs
            Learner keyword arguments specific to the subclass.

        Returns
        -------
        self : Self
        """

    @abstractmethod
    def _initialize_models(self) -> list[DoubleMLScalar]:
        """
        Create and return one scalar sub-model per treatment column.

        Called once during ``__init__`` of concrete subclasses.  Use
        :meth:`_get_data_for_model` to obtain a single-treatment data view for
        each ``d_col``, or bypass it for scenarios where all sub-models share the
        same data (e.g., APOS-like treatment-level orchestration).

        Returns
        -------
        list of DoubleMLScalar
            One configured scalar model per element of ``self._dml_data.d_cols``.
        """

    # ==================== Protected Helpers ====================

    def _get_data_for_model(self, d_col: str) -> DoubleMLData:
        """
        Return a single-treatment :class:`~doubleml.data.DoubleMLData` for ``d_col``.

        Creates a new :class:`~doubleml.data.DoubleMLData` that **shares the
        underlying DataFrame** (zero additional memory for array data). Other
        treatment columns are appended to ``x_cols`` so that the
        :class:`DoubleMLScalar` single-treatment check passes.

        Override in subclasses for non-d_col scenarios. For example, an APOS-like
        class would override this to return ``self._dml_data`` unchanged (each APO
        scalar stores its treatment level internally).

        Parameters
        ----------
        d_col : str
            The treatment column to make active.

        Returns
        -------
        DoubleMLData
            A :class:`~doubleml.data.DoubleMLData` with ``d_cols=[d_col]``
            and all other treatment columns added to ``x_cols``.
        """
        other_d_cols = [c for c in self._dml_data.d_cols if c != d_col]
        x_cols = list(self._dml_data.x_cols) + other_d_cols

        return DoubleMLData(
            data=self._dml_data.data,  # Shared DataFrame — zero copy overhead
            y_col=self._dml_data.y_col,
            d_cols=d_col,
            x_cols=x_cols,
            z_cols=self._dml_data.z_cols,
            cluster_cols=self._dml_data.cluster_cols,
            use_other_treat_as_covariate=False,  # Already handled above
            force_all_x_finite=self._dml_data.force_all_x_finite,
            force_all_d_finite=self._dml_data.force_all_d_finite,
        )

    def _reset_fit_state(self) -> None:
        """Clear fit-dependent state when sample splitting changes."""
        self._framework = None
        if self._modellist is not None:
            for model in self._modellist:
                model._reset_fit_state()

    def _propagate_splits_to_models(self) -> None:
        """Push the vector's sample splits into each sub-model."""
        if self._modellist is None:
            raise ValueError("Sub-models are not initialized. Call _initialize_models() in the subclass __init__.")
        for model in self._modellist:
            model._smpls = self._smpls
            model._smpls_cluster = self._smpls_cluster
            model._n_folds = self._n_folds
            model._n_folds_per_cluster = self._n_folds_per_cluster
            model._n_rep = self._n_rep

    def _fit_single_model(
        self,
        i_d: int,
        n_jobs_cv: int | None,
        ext_preds: dict[str, np.ndarray] | None,
    ) -> DoubleMLScalar:
        """Fit nuisance models and estimate causal parameters for one sub-model."""
        if self._modellist is None:
            raise ValueError("Sub-models are not initialized.")
        model = self._modellist[i_d]
        model.fit(n_jobs_cv=n_jobs_cv, external_predictions=ext_preds)
        return model

    # ==================== Sample Splitting ====================

    def draw_sample_splitting(self, n_folds: int = 5, n_rep: int = 1) -> Self:
        """
        Draw sample splitting for cross-fitting.

        Splits are drawn once for the vector and shared across all sub-models via
        :meth:`_propagate_splits_to_models`.

        Parameters
        ----------
        n_folds : int, optional
            Number of folds. Default is ``5``.
        n_rep : int, optional
            Number of repetitions. Default is ``1``.

        Returns
        -------
        self : Self

        Raises
        ------
        ValueError
            If ``n_folds < 2`` or ``n_rep < 1``.
        """
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2. Got {n_folds}.")
        if not isinstance(n_rep, int) or n_rep < 1:
            raise ValueError(f"n_rep must be an integer >= 1. Got {n_rep}.")

        resampler = DoubleMLResampling(
            n_folds=n_folds,
            n_rep=n_rep,
            n_obs=self._n_obs,
        )
        self._smpls = resampler.split_samples()
        self._smpls_cluster = None
        self._n_folds = n_folds
        self._n_folds_per_cluster = None
        self._n_rep = n_rep

        self._reset_fit_state()
        return self

    def set_sample_splitting(self, all_smpls: list, all_smpls_cluster: list | None = None) -> Self:
        """
        Set pre-computed sample splitting for all sub-models.

        Parameters
        ----------
        all_smpls : list
            List of ``(train_ind, test_ind)`` tuples per fold, or a list of such
            lists for repeated sample splitting.
        all_smpls_cluster : list or None, optional
            Nested list for cluster sample splitting. Default is ``None``.

        Returns
        -------
        self : Self

        Raises
        ------
        TypeError
            If ``all_smpls`` is not a list.
        ValueError
            If the partition is invalid.
        """
        if isinstance(all_smpls, tuple):
            raise TypeError("all_smpls must be a list of folds; tuple shorthand is not supported for DoubleMLVector.")
        if not isinstance(all_smpls, list):
            raise TypeError(f"all_smpls must be of list type. " f"{str(all_smpls)} of type {str(type(all_smpls))} was passed.")

        smpls, smpls_cluster, n_rep, n_folds = _check_sample_splitting(
            all_smpls,
            all_smpls_cluster,
            self._dml_data,
            self._dml_data.is_cluster_data,
            n_obs=self._n_obs,
        )

        self._smpls = smpls
        self._smpls_cluster = smpls_cluster
        self._n_rep = n_rep
        self._n_folds = n_folds
        self._n_folds_per_cluster = None

        self._reset_fit_state()
        return self

    # ==================== Fit ====================

    def fit(
        self,
        n_folds: int = 5,
        n_rep: int = 1,
        n_jobs_models: int | None = None,
        n_jobs_cv: int | None = None,
        external_predictions: dict[str, dict[str, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Estimate all sub-models and combine their results.

        Calls :meth:`draw_sample_splitting` (if not yet done), fits each scalar
        sub-model (optionally in parallel via joblib), and concatenates their
        :class:`~doubleml.DoubleMLFramework` objects into one unified result.

        Parameters
        ----------
        n_folds : int, optional
            Number of cross-fitting folds. Default is ``5``.
            Only used if sample splitting has not been drawn yet.
        n_rep : int, optional
            Number of repetitions. Default is ``1``.
            Only used if sample splitting has not been drawn yet.
        n_jobs_models : int or None, optional
            Number of jobs for parallel sub-model fitting. ``None`` means
            sequential. Default is ``None``.
        n_jobs_cv : int or None, optional
            Number of jobs for cross-validation inside each sub-model.
            Default is ``None``.
        external_predictions : dict or None, optional
            Nested dictionary keyed by treatment column name. Each value is a dict
            of external predictions passed to the corresponding sub-model's
            :meth:`~doubleml.DoubleMLScalar.fit_nuisance_models`.
            Default is ``None``.

        Returns
        -------
        self : Self
        """
        if self._smpls is None:
            self.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)

        self._propagate_splits_to_models()

        fitted_models = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")(
            delayed(self._fit_single_model)(
                i_d,
                n_jobs_cv,
                external_predictions.get(d_col) if external_predictions is not None else None,
            )
            for i_d, d_col in enumerate(self._dml_data.d_cols)
        )

        self._modellist = list(fitted_models)

        # Concatenate scalar frameworks into one unified multi-treatment framework
        self._framework = concat([m.framework for m in self._modellist])
        self._framework.treatment_names = list(self._dml_data.d_cols)

        return self

    # ==================== Learner Access ====================

    def get_params(self, learner_name: str) -> list[dict]:
        """
        Get parameters of a learner across all sub-models.

        Parameters
        ----------
        learner_name : str
            Name of the learner.

        Returns
        -------
        list of dict
            One parameter dict per sub-model, in ``d_cols`` order.
        """
        if self._modellist is None:
            raise ValueError("Sub-models are not initialized. Call _initialize_models() in the subclass __init__.")
        return [model.get_params(learner_name) for model in self._modellist]

    def set_params(self, learner_name: str, **params: object) -> Self:
        """
        Set parameters of a learner on all sub-models.

        Parameters
        ----------
        learner_name : str
            Name of the learner.
        **params
            Parameters to set on each sub-model's learner.

        Returns
        -------
        self : Self
        """
        if self._modellist is None:
            raise ValueError("Sub-models are not initialized. Call _initialize_models() in the subclass __init__.")
        for model in self._modellist:
            model.set_params(learner_name, **params)
        return self

    # ==================== Hyperparameter Tuning ====================

    def tune_ml_models(
        self,
        ml_param_space: dict,
        scoring_methods: dict | None = None,
        cv: int = 5,
        set_as_params: bool = True,
        return_tune_res: bool = False,
        optuna_settings: dict | None = None,
    ) -> "Self | list[dict[str, DMLOptunaResult]]":
        """Hyperparameter-tuning for DoubleML models using Optuna."""
        if self._modellist is None:
            raise ValueError("Sub-models are not initialized. Call _initialize_models() in the subclass __init__.")
        tuning_kwargs: dict[str, Any] = {
            "ml_param_space": ml_param_space,
            "scoring_methods": scoring_methods,
            "cv": cv,
            "set_as_params": set_as_params,
            "return_tune_res": return_tune_res,
            "optuna_settings": optuna_settings,
        }

        tune_res: list = []
        for model in self._modellist:
            res = model.tune_ml_models(**tuning_kwargs)
            if return_tune_res:
                tune_res.append(res)

        return tune_res if return_tune_res else self

    tune_ml_models.__doc__ = TUNE_ML_MODELS_DOC

    # ==================== Sensitivity ====================

    def sensitivity_plot(
        self,
        idx_treatment: int = 0,
        value: str = "theta",
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: float = 0.0,
        include_scenario: bool = True,
        benchmarks: dict | None = None,
        fill: bool = True,
        grid_bounds: tuple[float, float] = (0.15, 0.15),
        grid_size: int = 100,
    ) -> object:
        """
        Contour plot of the sensitivity with respect to latent/confounding variables.

        Parameters
        ----------
        idx_treatment : int, optional
            Index of the treatment parameter to plot. Default is ``0``.
        value : str, optional
            Contour value: ``'theta'`` for bounds, ``'ci'`` for bounds including
            statistical uncertainty. Default is ``'theta'``.
        rho : float, optional
            Correlation between confounders in the main regression and Riesz
            representer. Default is ``1.0``.
        level : float, optional
            The confidence level. Default is ``0.95``.
        null_hypothesis : float, optional
            Null hypothesis for the direction of contour lines. Default is ``0.0``.
        include_scenario : bool, optional
            Whether to highlight the last :meth:`sensitivity_analysis` scenario.
            Default is ``True``.
        benchmarks : dict or None, optional
            Benchmark dictionary with keys ``'cf_y'``, ``'cf_d'``, ``'name'``.
            Default is ``None``.
        fill : bool, optional
            Heatmap style (``True``) vs. contour lines only (``False``).
            Default is ``True``.
        grid_bounds : tuple of float, optional
            Evaluation bounds ``(cf_d_max, cf_y_max)`` in ``[0, 1)``.
            Default is ``(0.15, 0.15)``.
        grid_size : int, optional
            Number of grid evaluation points. Default is ``100``.

        Returns
        -------
        fig : plotly figure
            Plotly figure of the sensitivity contours.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.
        """
        if self._framework is None:
            raise ValueError("Apply fit() before sensitivity_plot().")
        return self._framework.sensitivity_plot(
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

    def sensitivity_benchmark(self, benchmarking_set: list[str], fit_args: dict | None = None) -> pd.DataFrame:
        """
        Compute a benchmark for a given set of features.

        Refits a short-form model excluding ``benchmarking_set`` from ``x_cols``
        and computes gain statistics comparing long and short forms.

        Parameters
        ----------
        benchmarking_set : list of str
            Feature names to benchmark. Must be a non-empty subset of ``x_cols``.
        fit_args : dict or None, optional
            Additional keyword arguments passed to :meth:`fit` when refitting the
            short-form model. Default is ``None``.

        Returns
        -------
        pd.DataFrame
            Benchmark results indexed by treatment column names with columns
            ``'cf_y'``, ``'cf_d'``, ``'rho'``, and ``'delta_theta'``.

        Raises
        ------
        NotImplementedError
            If sensitivity analysis is not available for this model.
        TypeError
            If ``benchmarking_set`` or ``fit_args`` have the wrong type.
        ValueError
            If ``benchmarking_set`` is empty or not a subset of ``x_cols``.
        """
        if self._framework is None:
            raise ValueError("Apply fit() before sensitivity_benchmark().")

        x_list_long = self._dml_data.x_cols

        if self.sensitivity_elements is None:
            raise NotImplementedError(f"Sensitivity analysis not yet implemented for {self.__class__.__name__}.")
        if not isinstance(benchmarking_set, list):
            raise TypeError(
                f"benchmarking_set must be a list. " f"{str(benchmarking_set)} of type {type(benchmarking_set)} was passed."
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

        x_list_short = [x for x in x_list_long if x not in benchmarking_set]
        dml_short = copy.deepcopy(self)
        dml_short._dml_data.x_cols = x_list_short
        # Sub-models each hold their own DoubleMLData — rebuild them from the updated _dml_data
        # so that the short-form model actually uses the reduced feature set.
        dml_short._modellist = dml_short._initialize_models()
        dml_short._framework = None

        if fit_args is not None:
            dml_short.fit(**fit_args)
        else:
            dml_short.fit()

        benchmark_dict = gain_statistics(dml_long=self, dml_short=dml_short)
        df_benchmark = pd.DataFrame(benchmark_dict, index=self._dml_data.d_cols)
        return df_benchmark

    # ==================== String Representation ====================

    def __str__(self) -> str:
        """
        String representation of the DoubleMLVector object.

        Returns
        -------
        str
            A formatted string summary of the model.
        """
        class_name = self.__class__.__name__
        header = f"{'=' * 20} {class_name} Object {'=' * 20}"

        info = f"Score function: {self.score}\n"
        if self._n_folds is not None:
            info += f"Resampling: {self._n_folds}-fold CV, {self._n_rep} repetitions\n"
        info += f"Treatments: {list(self._dml_data.d_cols)}\n"

        if self._framework is not None:
            return f"{header}\n\n{info}\n{str(self.summary)}"
        else:
            return f"{header}\n\n{info}\nModel not yet fitted. Call fit() first."
