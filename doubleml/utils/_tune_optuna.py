"""
Optuna-based hyperparameter tuning utilities for DoubleML.

This module provides Optuna-specific functionality for hyperparameter optimization,
decoupled from sklearn-based grid/randomized search.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate


class _OptunaSearchResult:
    """Lightweight container mimicking selected GridSearchCV attributes."""

    def __init__(self, estimator, best_params, best_score, study, trials_dataframe):
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.study_ = study
        self.trials_dataframe_ = trials_dataframe

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("The wrapped estimator does not support predict_proba().")
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


def _resolve_optuna_settings(optuna_settings):
    """
    Merge user-provided Optuna settings with defaults.

    Parameters
    ----------
    optuna_settings : dict or None
        User-provided Optuna settings.

    Returns
    -------
    dict
        Resolved settings dictionary.
    """
    default_settings = {
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
        "n_jobs_optuna": None,  # Parallel trial execution
        "verbosity": None,  # Optuna logging verbosity level
    }

    if optuna_settings is None:
        return default_settings

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    resolved = default_settings.copy()
    resolved.update(optuna_settings)
    if not isinstance(resolved["study_kwargs"], dict):
        raise TypeError("optuna_settings['study_kwargs'] must be a dict.")
    if not isinstance(resolved["optimize_kwargs"], dict):
        raise TypeError("optuna_settings['optimize_kwargs'] must be a dict.")
    if resolved["callbacks"] is not None and not isinstance(resolved["callbacks"], (list, tuple)):
        raise TypeError("optuna_settings['callbacks'] must be a sequence of callables or None.")
    if resolved["study"] is not None and resolved["study_factory"] is not None:
        raise ValueError("Provide only one of 'study' or 'study_factory' in optuna_settings.")
    return resolved


def _select_optuna_settings(optuna_settings, learner_names):
    """
    Select appropriate Optuna settings, considering learner-specific overrides.

    Parameters
    ----------
    optuna_settings : dict or None
        Optuna settings dictionary that may contain learner-specific overrides.
    learner_names : str or list or None
        Name(s) of the learner to check for specific settings.

    Returns
    -------
    dict
        Resolved settings for the learner.
    """
    if optuna_settings is None:
        return _resolve_optuna_settings(None)

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    base_keys = {
        "n_trials",
        "timeout",
        "direction",
        "study_kwargs",
        "optimize_kwargs",
        "sampler",
        "pruner",
        "callbacks",
        "catch",
        "show_progress_bar",
        "gc_after_trial",
        "study_factory",
        "study",
        "n_jobs_optuna",
        "verbosity",
    }

    base_settings = {key: value for key, value in optuna_settings.items() if key in base_keys}

    if learner_names is None:
        learner_candidates = []
    elif isinstance(learner_names, (list, tuple)):
        learner_candidates = [name for name in learner_names if name is not None]
    else:
        learner_candidates = [learner_names]

    for learner_name in learner_candidates:
        learner_specific = optuna_settings.get(learner_name)
        if learner_specific is None:
            continue
        if not isinstance(learner_specific, dict):
            raise TypeError(f"optuna_settings for learner '{learner_name}' must be a dict or None.")

        merged = base_settings.copy()
        merged.update(learner_specific)
        return _resolve_optuna_settings(merged)

    return _resolve_optuna_settings(base_settings)


def _create_study(settings):
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
        try:
            maybe_study = study_factory(study_kwargs)
        except TypeError:
            # Factory doesn't accept kwargs, call without args
            maybe_study = study_factory()

        if maybe_study is None:
            # Factory returned None, create default study
            return optuna.create_study(**study_kwargs)
        elif isinstance(maybe_study, optuna.study.Study):
            return maybe_study
        else:
            raise TypeError("study_factory must return an optuna.study.Study or None.")

    # Build study kwargs from settings
    study_kwargs = settings.get("study_kwargs", {}).copy()
    if "direction" not in study_kwargs:
        study_kwargs["direction"] = settings.get("direction", "maximize")
    if settings.get("sampler") is not None:
        study_kwargs["sampler"] = settings["sampler"]
    if settings.get("pruner") is not None:
        study_kwargs["pruner"] = settings["pruner"]

    return optuna.create_study(**study_kwargs)


def _create_objective(param_grid_func, learner, x, y, cv, scoring_method, n_jobs_cv):
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
                f"param_grid function must return a dict. Got {type(params).__name__}. "
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
    train_inds,
    learner,
    param_grid_func,
    scoring_method,
    n_folds_tune,
    n_jobs_cv,
    optuna_settings,
    learner_name=None,
):
    """
    Tune hyperparameters using Optuna on the whole dataset with cross-validation.

    Unlike the grid/randomized search which tunes separately for each fold, this function
    tunes once on the full dataset and returns the same optimal parameters for all folds.

    Parameters
    ----------
    y : np.ndarray
        Target variable (full dataset).
    x : np.ndarray
        Features (full dataset).
    train_inds : list
        List of training indices for each fold (used only to determine number of folds to return).
    learner : estimator
        The machine learning model to tune.
    param_grid_func : callable
        Function that takes an Optuna trial and returns a parameter dictionary.
        Example: def params(trial): return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)}
    scoring_method : str or callable
        Scoring method for cross-validation.
    n_folds_tune : int
        Number of folds for cross-validation during tuning.
    n_jobs_cv : int or None
        Number of parallel jobs for cross-validation.
    optuna_settings : dict or None
        Optuna-specific settings.
    learner_name : str or None
        Name of the learner for settings selection.

    Returns
    -------
    list
        List of tuning results (one per fold in train_inds), each containing the same optimal parameters.
    """
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is not installed. Please install Optuna (e.g., pip install optuna) to use Optuna tuning."
        ) from exc

    # Input validation
    if not callable(param_grid_func):
        raise TypeError(
            "param_grid must be a callable function that takes a trial and returns a dict. "
            "Example: def params(trial): return {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)}"
        )
    if not train_inds:
        raise ValueError("train_inds cannot be empty.")

    # Get learner key (prefer logical learner name, fall back to estimator class)
    candidate_names = []
    if learner_name is not None:
        if isinstance(learner_name, (list, tuple)):
            candidate_names.extend(list(learner_name))
        else:
            candidate_names.append(learner_name)
    candidate_names.append(learner.__class__.__name__)
    # remove duplicates while preserving order
    seen = set()
    ordered_candidates = []
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        ordered_candidates.append(name)

    settings = _select_optuna_settings(optuna_settings, ordered_candidates)

    # Set Optuna logging verbosity if specified
    verbosity = settings.get("verbosity")
    if verbosity is not None:
        optuna.logging.set_verbosity(verbosity)

    # Pre-create KFold object for cross-validation during tuning (fixed random state for reproducibility)
    cv = KFold(n_splits=n_folds_tune, shuffle=True, random_state=42)

    # Create the study
    study = _create_study(settings)

    # Create the objective function
    objective = _create_objective(param_grid_func, learner, x, y, cv, scoring_method, n_jobs_cv)

    # Build optimize kwargs (filter out None values except for boolean flags)
    optimize_kwargs = {
        "n_trials": settings.get("n_trials"),
        "timeout": settings.get("timeout"),
        "callbacks": settings.get("callbacks"),
        "catch": settings.get("catch"),
        "show_progress_bar": settings.get("show_progress_bar", False),
        "gc_after_trial": settings.get("gc_after_trial", False),
    }

    # Add n_jobs for parallel trial execution if specified
    n_jobs_optuna = settings.get("n_jobs_optuna")
    if n_jobs_optuna is not None:
        optimize_kwargs["n_jobs"] = n_jobs_optuna

    # Update with any additional optimize_kwargs from settings
    optimize_kwargs.update(settings.get("optimize_kwargs", {}))

    # Filter out None values (but keep boolean flags)
    optimize_kwargs = {
        k: v for k, v in optimize_kwargs.items() if v is not None or k in ["show_progress_bar", "gc_after_trial"]
    }

    # Run optimization once on the full dataset
    study.optimize(objective, **optimize_kwargs)

    # Validate optimization results
    if study.best_trial is None:
        complete_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        raise RuntimeError(
            f"Optuna optimization failed to find any successful trials. "
            f"Total trials: {len(study.trials)}, Complete trials: {complete_trials}"
        )

    # Extract best parameters and score
    best_params = study.best_trial.params
    best_score = study.best_value

    # Cache trials dataframe (computed once and reused for all folds)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    # Create tuning results for each fold
    # All folds use the same optimal parameters, but each gets a fitted estimator on its training data
    tune_res = []
    for train_index in train_inds:
        # Fit the best estimator on this fold's training data
        best_estimator = clone(learner).set_params(**best_params)
        best_estimator.fit(x[train_index, :], y[train_index])

        # Create result object (study and trials_df are shared across all folds)
        tune_res.append(
            _OptunaSearchResult(
                estimator=best_estimator,
                best_params=best_params,
                best_score=best_score,
                study=study,
                trials_dataframe=trials_df,
            )
        )

    return tune_res
