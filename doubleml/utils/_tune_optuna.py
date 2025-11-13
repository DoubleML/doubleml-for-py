"""
Optuna-based hyperparameter tuning utilities for DoubleML.

This module provides Optuna-specific functionality for hyperparameter optimization,
decoupled from sklearn-based grid/randomized search.

Logging
-------
This module uses Python's logging module. The logger is synchronized with Optuna's logging
system. You can control the verbosity by:
1. Setting the logging level for 'doubleml.utils._tune_optuna'
2. Passing 'verbosity' in optuna_settings (takes precedence)

Example:
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> # Now you'll see tuning progress and information
"""

import logging
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, Union

import numpy as np
import optuna
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import BaseCrossValidator, KFold, cross_val_score

logger = logging.getLogger(__name__)

_OPTUNA_DEFAULT_SETTINGS = {
    "n_trials": 100,
    "timeout": None,
    "direction": "maximize",
    "study_kwargs": {},
    "optimize_kwargs": {},
    "sampler": None,
    "callbacks": None,
    "catch": (),
    "show_progress_bar": False,
    "gc_after_trial": False,
    "study": None,
    "n_jobs_optuna": None,
    "verbosity": None,
}


@dataclass
class DMLOptunaResult:
    """
    Container for Optuna search results.
    Attributes
    ----------
    learner_name : str
        Name of the learner passed (e.g., 'ml_g').
    params_name : str
        Name of the nuisance parameter being tuned (e.g., 'ml_g0').
    best_estimator : object
        The estimator instance with the best found hyperparameters set (not fitted).
    best_params : dict
        The best hyperparameters found during tuning.
    best_score : float
        The best average cross-validation score achieved during tuning.
    scoring_method : str or callable
        The scoring method used during tuning.
    study : optuna.study.Study
        The Optuna study object containing the tuning history.
    tuned : bool
        Indicates whether tuning was performed (True) or skipped (False).
    """

    learner_name: str
    params_name: str
    best_estimator: object
    best_params: dict
    best_score: float
    scoring_method: Union[str, Callable]
    study: optuna.study.Study
    tuned: bool

    def __str__(self):
        core_summary = self._core_summary_str()
        params_summary = self._best_params_str()
        res = (
            "================== DMLOptunaResult ==================\n"
            + core_summary
            + "\n------------------ Best parameters    ------------------\n"
            + params_summary
        )
        return res

    def _core_summary_str(self):
        scoring_repr = (
            self.scoring_method.__name__
            if callable(self.scoring_method) and hasattr(self.scoring_method, "__name__")
            else str(self.scoring_method)
        )
        summary = (
            f"Learner name: {self.learner_name}\n"
            f"Params name: {self.params_name}\n"
            f"Tuned: {self.tuned}\n"
            f"Best score: {self.best_score}\n"
            f"Scoring method: {scoring_repr}\n"
        )
        return summary

    def _best_params_str(self):
        if not self.best_params:
            return "No best parameters available.\n"
        formatted = pformat(self.best_params, sort_dicts=True, compact=True)
        return f"{formatted}\n"


OPTUNA_GLOBAL_SETTING_KEYS = frozenset(_OPTUNA_DEFAULT_SETTINGS.keys())


def _default_optuna_settings():
    return deepcopy(_OPTUNA_DEFAULT_SETTINGS)


def _resolve_optuna_scoring(scoring_method, learner, learner_name):
    """Resolve the scoring argument for an Optuna-tuned learner.

    Parameters
    ----------
    scoring_method : str, callable or None
        Scoring argument supplied by the caller. ``None`` triggers automatic
        fallback selection.
    learner : estimator
        Estimator instance that will be tuned.
    learner_name : str
        Identifier used for logging and error messages.

    Returns
    -------
    tuple
    A pair consisting of the scoring argument to pass to
    :func:`sklearn.model_selection.cross_val_score` (``None`` means use the
    estimator's default ``score``) and a human-readable message describing
    the decision for logging purposes.
    """

    if scoring_method is not None:
        message = f"Using provided scoring method: {scoring_method} for learner '{learner_name}'"
        return scoring_method, message

    if is_regressor(learner):
        message = (
            "No scoring method provided, using 'neg_root_mean_squared_error' (RMSE) "
            f"for learner '{learner_name}'."
        )
        return "neg_root_mean_squared_error", message

    if is_classifier(learner):
        message = (
            f"No scoring method provided, using 'neg_log_loss' "
            f"for learner '{learner_name}'."
        )
        return "neg_log_loss", message


    raise RuntimeError(
        f"No scoring method provided and estimator type could not be inferred. Please provide a scoring_method for learner "
        f"'{learner_name}'."
    )


def resolve_optuna_cv(cv):
    """Normalize the ``cv`` argument for Optuna-based tuning."""

    if cv is None:
        cv = 5

    if isinstance(cv, int):
        if cv < 2:
            raise ValueError(f"The number of folds used for tuning must be at least two. {cv} was passed.")
        return KFold(n_splits=cv, shuffle=True, random_state=42)

    if isinstance(cv, BaseCrossValidator):
        return cv

    if isinstance(cv, str):
        raise TypeError("cv must not be provided as a string. Pass an integer or a cross-validation splitter.")

    split_attr = getattr(cv, "split", None)
    if callable(split_attr):
        return cv

    if isinstance(cv, Iterable):
        cv_list = list(cv)
        if not cv_list:
            raise ValueError("cv iterable must not be empty.")
        for split in cv_list:
            if not isinstance(split, (tuple, list)) or len(split) != 2:
                raise TypeError("cv iterable must yield (train_indices, test_indices) pairs.")
        return cv_list

    raise TypeError(
        "cv must be an integer >= 2, a scikit-learn cross-validation splitter, or an iterable of "
        "(train_indices, test_indices) pairs."
    )


def _check_tuning_inputs(
    y,
    x,
    learner,
    param_grid_func,
    scoring_method,
    cv,
    learner_name=None,
):
    """Validate Optuna tuning inputs and normalize the cross-validation splitter.

    Parameters
    ----------
    y : np.ndarray
        Target array used during tuning.
    x : np.ndarray
        Feature matrix used during tuning.
    learner : estimator
        Estimator that will be tuned.
    param_grid_func : callable or None
        Callback that samples hyperparameters from an Optuna trial.
    scoring_method : str, callable or None
    Scoring argument after applying :func:`doubleml.utils._tune_optuna._resolve_optuna_scoring`.
    cv : int, cross-validation splitter or iterable
        Cross-validation definition provided by the caller.
    learner_name : str or None
        Optional name used to contextualise error messages.

    Returns
    -------
    cross-validator or iterable
        Cross-validation splitter compatible with
        :func:`sklearn.model_selection.cross_val_score`.
    """

    learner_label = learner_name or learner.__class__.__name__

    if y.shape[0] != x.shape[0]:
        raise ValueError(f"Features and target must contain the same number of observations for learner '{learner_label}'.")
    if y.size == 0:
        raise ValueError(f"Empty target passed to Optuna tuner for learner '{learner_label}'.")

    if param_grid_func is not None and not callable(param_grid_func):
        raise TypeError(
            "param_grid must be a callable function that takes a trial and returns a dict. "
            f"Got {type(param_grid_func).__name__} for learner '{learner_label}'."
        )

    if scoring_method is not None and not callable(scoring_method) and not isinstance(scoring_method, str):
        if not isinstance(scoring_method, Iterable):
            raise TypeError(
                "scoring_method must be None, a string, a callable, or an iterable accepted by scikit-learn. "
                f"Got {type(scoring_method).__name__} for learner '{learner_label}'."
            )

    if not hasattr(learner, "fit") or not hasattr(learner, "set_params"):
        raise TypeError(f"Learner '{learner_label}' must implement fit and set_params to be tuned with Optuna.")

    return resolve_optuna_cv(cv)


def _get_optuna_settings(optuna_settings, params_name=None):
    """
    Get Optuna settings, considering defaults, user-provided values, and learner-specific overrides.

    Parameters
    ----------
    optuna_settings : dict or None
        User-provided Optuna settings.
    params_name : str
        Name of the learner to check for specific setting, e.g. `ml_g0` or `ml_g1` for `DoubleMLIRM`.
    default_learner_name : str or None
        A default learner name to use as a fallback.

    Returns
    -------
    dict
        Resolved settings dictionary.
    """
    default_settings = _default_optuna_settings()

    if optuna_settings is None:
        return default_settings

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    # Base settings are the user-provided settings filtered by default keys
    base_settings = {key: value for key, value in optuna_settings.items() if key in OPTUNA_GLOBAL_SETTING_KEYS}
    learner_or_params_keys = set(optuna_settings.keys()) - set(base_settings.keys())

    # Find matching learner-specific settings, handles the case to match ml_g to ml_g0, ml_g1, etc.
    learner_specific_settings = {}
    if any(params_name in key for key in learner_or_params_keys):
        for k in learner_or_params_keys:
            if params_name in k and params_name != k:
                learner_specific_settings = optuna_settings[k]

    # set params specific settings
    params_specific_settings = {}
    if params_name in learner_or_params_keys:
        params_specific_settings = optuna_settings[params_name]

    # Merge settings: defaults < base < learner-specific < params_specific
    resolved = default_settings.copy() | base_settings | learner_specific_settings | params_specific_settings

    # Validate types
    if not isinstance(resolved["study_kwargs"], dict):
        raise TypeError("study_kwargs must be a dict.")
    if not isinstance(resolved["optimize_kwargs"], dict):
        raise TypeError("optimize_kwargs must be a dict.")
    if resolved["callbacks"] is not None and not isinstance(resolved["callbacks"], (list, tuple)):
        raise TypeError("callbacks must be a sequence of callables or None.")

    return resolved


def _create_study(settings, learner_name):
    """
    Create or retrieve an Optuna :class:`optuna.study.Study` instance.

    Parameters
    ----------
    settings : dict
        Resolved Optuna settings containing study configuration.
    learner_name : str
        Identifier used for logging the resolved study configuration.

    Returns
    -------
    optuna.study.Study
        The Optuna study object ready for optimization.
    """

    # Check if a study instance is provided directly
    study_instance = settings.get("study")
    if study_instance is not None:
        return study_instance

    # Build study kwargs from settings
    study_kwargs = settings.get("study_kwargs", {}).copy()
    if "direction" not in study_kwargs:
        study_kwargs["direction"] = settings.get("direction", "maximize")
        logger.info(f"Optuna study direction set to '{study_kwargs['direction']}' for learner '{learner_name}'.")
    if settings.get("sampler") is not None:
        study_kwargs["sampler"] = settings["sampler"]
        logger.info(f"Using sampler {settings['sampler'].__class__.__name__} for learner '{learner_name}'.")

    return optuna.create_study(**study_kwargs, study_name=f"tune_{learner_name}")


def _create_objective(param_grid_func, learner, x, y, cv, scoring_method):
    """
    Create an Optuna objective function for hyperparameter optimization.

    Parameters
    ----------
    param_grid_func : callable
        Function that takes an Optuna trial and returns a parameter dictionary.
        Example: def params(trial): return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)}
    learner : estimator
        The machine learning model to tune.
    x : np.ndarray
        Features (full dataset).
    y : np.ndarray
        Target variable (full dataset).
    cv : cross-validation generator
        KFold or similar cross-validation splitter.
    scoring_method : str, callable or None
        Scoring argument for cross-validation. ``None`` delegates to the
        estimator's default ``score`` implementation.

    Returns
    -------
    callable
        Objective function for Optuna optimization.
    """

    def objective(trial):
        """Objective function for Optuna optimization."""
        # Get parameters from the user-provided function
        params = param_grid_func(trial)

        if not isinstance(params, dict):
            raise TypeError(
                f"param function must return a dict. Got {type(params).__name__}. "
                f"Example: def params(trial): return {{'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)}}"
            )

        # Clone learner and set parameters
        estimator = clone(learner).set_params(**params)

        # Perform cross-validation on full dataset
        scores = cross_val_score(
            estimator,
            x,
            y,
            cv=cv,
            scoring=scoring_method,
            error_score="raise",
        )

        # Return mean test score
        return np.nanmean(scores)

    return objective


def _dml_tune_optuna(
    y,
    x,
    learner,
    param_grid_func,
    scoring_method,
    cv,
    optuna_settings,
    learner_name,
    params_name,
):
    """
    Tune hyperparameters using Optuna on the whole dataset with cross-validation.

    Unlike the grid/randomized search which tunes separately for each fold, this function
    tunes once on the full dataset and returns a single tuning result per learner.

    Parameters
    ----------
    y : np.ndarray
        Target variable (full dataset).
    x : np.ndarray
        Features (full dataset).
    learner : estimator
        The machine learning model to tune.
    param_grid_func : callable
        Function that takes an Optuna trial and returns a parameter dictionary.
        Example: def params(trial): return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)}
    scoring_method : str, callable or None
        Scoring argument passed to cross-validation. ``None`` triggers an
        automatic fallback chosen by :func:`_resolve_optuna_scoring`.
    cv : int, cross-validation splitter, or iterable of (train_indices, test_indices)
        Cross-validation strategy used during tuning. If an integer is provided, a shuffled
        :class:`sklearn.model_selection.KFold` with the specified number of splits and ``random_state=42`` is used.
    optuna_settings : dict or None
        Optuna-specific settings.
    params_name : str or None
        Name of the nuisance parameter for settings selection.

    Returns
    -------
    DMLOptunaResult
        A tuning result containing the optuna.Study object and further information.
    """

    scoring_method, scoring_message = _resolve_optuna_scoring(scoring_method, learner, params_name)
    if scoring_message:
        logger.info(scoring_message)

    cv_splitter = _check_tuning_inputs(
        y,
        x,
        learner,
        param_grid_func,
        scoring_method,
        cv,
        learner_name=params_name,
    )

    if param_grid_func is None:
        estimator = clone(learner)
        best_params = estimator.get_params(deep=True)
        return DMLOptunaResult(
            params_name=params_name,
            learner_name=learner_name,
            estimator=estimator,
            best_params=best_params,
            best_score=np.nan,
            study=None,
            trials_dataframe=None,
            tuned=False,
        )

    settings = _get_optuna_settings(optuna_settings, params_name or learner_name)

    # Set Optuna logging verbosity if specified
    verbosity = settings.get("verbosity")
    if verbosity is not None:
        optuna.logging.set_verbosity(verbosity)

    # Create the study
    study = _create_study(settings, learner_name)
    study.set_metric_names([f"{scoring_method}_{learner_name}"])

    # Create the objective function
    objective = _create_objective(param_grid_func, learner, x, y, cv_splitter, scoring_method)

    # Build optimize kwargs
    optimize_kwargs = {
        "n_trials": settings["n_trials"],
        "timeout": settings["timeout"],
        "callbacks": settings["callbacks"],
        "catch": settings["catch"],
        "show_progress_bar": settings["show_progress_bar"],
        "gc_after_trial": settings["gc_after_trial"],
        "n_jobs": settings["n_jobs_optuna"],
    }
    optimize_kwargs.update(settings.get("optimize_kwargs", {}))

    # Filter out None values, but keep boolean flags
    final_optimize_kwargs = {
        k: v for k, v in optimize_kwargs.items() if v is not None or k in ["show_progress_bar", "gc_after_trial"]
    }

    # Run optimization once on the full dataset
    study.optimize(objective, **final_optimize_kwargs)

    # Validate optimization results
    if not study.trials or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("Optuna optimization failed to produce any complete trials.")

    # Extract best parameters and score
    # Since each learner is tuned independently, use all parameters from the study
    best_params = dict(study.best_trial.params)
    best_score = study.best_value

    # Cache trials dataframe (computed once and reused for all folds)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    # Fit the best estimator on the full dataset once
    best_estimator = clone(learner).set_params(**best_params)
    best_estimator.fit(x, y)

    return DMLOptunaResult(
        estimator=best_estimator,
        best_params=best_params,
        best_score=best_score,
        study=study,
        trials_dataframe=trials_df,
        tuned=True,
    )


def _join_param_spaces(param_space_global, param_space_local):
    def joined_param_space(trial):
        return param_space_global(trial) | param_space_local(trial)

    return joined_param_space
