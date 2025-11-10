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

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import BaseCrossValidator, KFold, cross_validate

logger = logging.getLogger(__name__)

_OPTUNA_DEFAULT_SETTINGS = {
    "n_trials": 100,
    "timeout": None,
    "direction": "maximize",
    "study_kwargs": {},
    "optimize_kwargs": {},
    "sampler": None,
    "pruner": None,
    "callbacks": None,
    "catch": (),
    "show_progress_bar": False,
    "gc_after_trial": False,
    "study_factory": None,
    "study": None,
    "n_jobs_optuna": None,
    "verbosity": None,
}


OPTUNA_GLOBAL_SETTING_KEYS = frozenset(_OPTUNA_DEFAULT_SETTINGS.keys())


def _default_optuna_settings():
    return deepcopy(_OPTUNA_DEFAULT_SETTINGS)


def _resolve_optuna_scoring(scoring_method, learner, learner_name):
    """Select a scoring method when Optuna tuning does not receive one explicitly."""

    if scoring_method is not None:
        message = f"Using provided scoring method: {scoring_method} for learner '{learner_name}'"
        return scoring_method, message

    criterion = getattr(learner, "criterion", None)
    if criterion is not None:
        message = f"No scoring method provided, using estimator criterion '{criterion}' for learner '{learner_name}'."
        return None, message

    if is_regressor(learner):
        message = (
            "No scoring method provided and estimator has no criterion; using 'neg_root_mean_squared_error' (RMSE) "
            f"for learner '{learner_name}'."
        )
        return "neg_root_mean_squared_error", message

    if is_classifier(learner):
        if hasattr(learner, "predict_proba"):
            metric = "neg_log_loss"
            readable = "log loss"
        else:
            metric = "accuracy"
            readable = "accuracy"
        message = (
            f"No scoring method provided and estimator has no criterion; using '{metric}' ({readable}) "
            f"for learner '{learner_name}'."
        )
        return metric, message

    message = (
        f"No scoring method provided and estimator type could not be inferred. Please provide a scoring_method for learner "
        f"'{learner_name}'."
    )
    return None, message


class _OptunaSearchResult:
    """Container for Optuna search results."""

    def __init__(self, estimator, best_params, best_score, study, trials_dataframe, tuned=True):
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.study_ = study
        self.trials_dataframe_ = trials_dataframe
        self.tuned_ = tuned

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("The wrapped estimator does not support predict_proba().")
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


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
    n_jobs_cv,
    learner_name=None,
):
    """Validate Optuna tuning inputs and return a normalized cross-validation splitter."""

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

    if n_jobs_cv is not None and not isinstance(n_jobs_cv, int):
        raise TypeError(
            "The number of CPUs used to fit the learners must be of int type. "
            f"{n_jobs_cv} of type {type(n_jobs_cv).__name__} was passed for learner '{learner_label}'."
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


def _get_optuna_settings(optuna_settings, learner_name=None, default_learner_name=None):
    """
    Get Optuna settings, considering defaults, user-provided values, and learner-specific overrides.

    Parameters
    ----------
    optuna_settings : dict or None
        User-provided Optuna settings.
    learner_name : str or list or None
        Name(s) of the learner to check for specific settings.
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

    # Determine the search order for learner-specific settings
    learner_candidates = []
    if learner_name:
        if isinstance(learner_name, (list, tuple)):
            learner_candidates.extend(learner_name)
        else:
            learner_candidates.append(learner_name)
    if default_learner_name:
        learner_candidates.append(default_learner_name)

    # Find the first matching learner-specific settings
    learner_specific_settings = {}
    for name in learner_candidates:
        if name in optuna_settings and isinstance(optuna_settings[name], dict):
            learner_specific_settings = optuna_settings[name]
            break

    # Merge settings: defaults < base < learner-specific
    resolved = default_settings.copy()
    resolved.update(base_settings)
    resolved.update(learner_specific_settings)

    # Validate types
    if not isinstance(resolved["study_kwargs"], dict):
        raise TypeError("study_kwargs must be a dict.")
    if not isinstance(resolved["optimize_kwargs"], dict):
        raise TypeError("optimize_kwargs must be a dict.")
    if resolved["callbacks"] is not None and not isinstance(resolved["callbacks"], (list, tuple)):
        raise TypeError("callbacks must be a sequence of callables or None.")
    if resolved["study"] is not None and resolved["study_factory"] is not None:
        raise ValueError("Provide only one of 'study' or 'study_factory' in optuna_settings.")

    return resolved


def _create_study(settings, learner_name):
    """
    Create or retrieve an Optuna study object.

    Parameters
    ----------
    settings : dict
        Resolved Optuna settings containing study configuration.

    Returns
    -------
    optuna.study.Study
        The Optuna study object.
    """
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is not installed. Please install Optuna (e.g., pip install optuna) to use Optuna tuning."
        ) from exc

    # Check if a study instance is provided directly
    study_instance = settings.get("study")
    if study_instance is not None:
        return study_instance

    # Check if a study factory is provided
    study_factory = settings.get("study_factory")
    if callable(study_factory):
        study_kwargs = settings.get("study_kwargs", {})
        # Try to pass kwargs, but fall back to no-arg call if it fails
        try:
            maybe_study = study_factory(**study_kwargs)
        except TypeError:
            maybe_study = study_factory()

        if isinstance(maybe_study, optuna.study.Study):
            return maybe_study
        elif maybe_study is not None:
            raise TypeError("study_factory must return an optuna.study.Study or None.")
        # If factory returns None, proceed to create a default study below

    # Build study kwargs from settings
    study_kwargs = settings.get("study_kwargs", {}).copy()
    if "direction" not in study_kwargs:
        study_kwargs["direction"] = settings.get("direction", "maximize")
        logger.info(f"Optuna study direction set to '{study_kwargs['direction']}' for learner '{learner_name}'.")
    if settings.get("sampler") is not None:
        study_kwargs["sampler"] = settings["sampler"]
        logger.info(f"Using sampler {settings['sampler'].__class__.__name__} for learner '{learner_name}'.")
    if settings.get("pruner") is not None:
        study_kwargs["pruner"] = settings["pruner"]
        logger.info(f"Using pruner {settings['pruner'].__class__.__name__} for learner '{learner_name}'.")

    return optuna.create_study(**study_kwargs, study_name=f"tune_{learner_name}")


def _create_objective(param_grid_func, learner, x, y, cv, scoring_method, n_jobs_cv, learner_name):
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
    scoring_method : str or callable
        Scoring method for cross-validation.
    n_jobs_cv : int or None
        Number of parallel jobs for cross-validation.
    learner_name : str
        Name of the learner.

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
        cv_results = cross_validate(
            estimator,
            x,
            y,
            cv=cv,
            scoring=scoring_method,
            n_jobs=n_jobs_cv,
            return_train_score=False,
            error_score="raise",
        )

        # Return mean test score
        return np.nanmean(cv_results["test_score"])

    return objective


def _dml_tune_optuna(
    y,
    x,
    learner,
    param_grid_func,
    scoring_method,
    cv,
    n_jobs_cv,
    optuna_settings,
    learner_name=None,
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
    scoring_method : str or callable
        Scoring method for cross-validation.
    cv : int, cross-validation splitter, or iterable of (train_indices, test_indices)
        Cross-validation strategy used during tuning. If an integer is provided, a shuffled
        :class:`sklearn.model_selection.KFold` with the specified number of splits and ``random_state=42`` is used.
    n_jobs_cv : int or None
        Number of parallel jobs for cross-validation.
    optuna_settings : dict or None
        Optuna-specific settings.
    learner_name : str or None
        Name of the learner for settings selection.

    Returns
    -------
    _OptunaSearchResult
        A tuning result containing the fitted estimator with the optimal parameters.
    """
    learner_label = learner_name or learner.__class__.__name__
    scoring_method, scoring_message = _resolve_optuna_scoring(scoring_method, learner, learner_label)
    if scoring_message:
        logger.info(scoring_message)

    cv_splitter = _check_tuning_inputs(
        y,
        x,
        learner,
        param_grid_func,
        scoring_method,
        cv,
        n_jobs_cv,
        learner_label,
    )

    skip_tuning = param_grid_func is None

    if skip_tuning:
        estimator = clone(learner)
        estimator.fit(x, y)
        best_params = estimator.get_params(deep=False)
        return _OptunaSearchResult(
            estimator=estimator,
            best_params=best_params,
            best_score=np.nan,
            study=None,
            trials_dataframe=None,
            tuned=False,
        )

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is not installed. Please install Optuna (e.g., pip install optuna) to use Optuna tuning."
        ) from exc

    settings = _get_optuna_settings(optuna_settings, learner_name, learner.__class__.__name__)

    # Set Optuna logging verbosity if specified
    verbosity = settings.get("verbosity")
    if verbosity is not None:
        optuna.logging.set_verbosity(verbosity)
    else:
        # Sync DoubleML logger level with Optuna logger level
        doubleml_level = logger.getEffectiveLevel()
        if doubleml_level == logging.DEBUG:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
        elif doubleml_level == logging.INFO:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        elif doubleml_level == logging.WARNING:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif doubleml_level >= logging.ERROR:
            optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Create the study
    study = _create_study(settings, learner_label)

    # Create the objective function
    objective = _create_objective(param_grid_func, learner, x, y, cv_splitter, scoring_method, n_jobs_cv, learner_label)

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

    return _OptunaSearchResult(
        estimator=best_estimator,
        best_params=best_params,
        best_score=best_score,
        study=study,
        trials_dataframe=trials_df,
        tuned=True,
    )
