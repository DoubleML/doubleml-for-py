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
        "n_jobs_optuna": None,
        "verbosity": None,
    }

    if optuna_settings is None:
        return default_settings

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    # Base settings are the user-provided settings filtered by default keys
    base_settings = {key: value for key, value in optuna_settings.items() if key in default_settings}

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
    if settings.get("sampler") is not None:
        study_kwargs["sampler"] = settings["sampler"]
    if settings.get("pruner") is not None:
        study_kwargs["pruner"] = settings["pruner"]

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
        all_params = param_grid_func(trial)

        if not isinstance(all_params, dict):
            raise TypeError(
                f"param function must return a dict. Got {type(all_params).__name__}. "
                f"Example: def params(trial): return {{'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)}}"
            )

        # Filter and strip prefix for the current learner
        prefix = f"{learner_name}_"
        learner_params = {
            key.replace(prefix, ""): value for key, value in all_params.items() if key.startswith(prefix)
        }

        # Clone learner and set parameters
        estimator = clone(learner).set_params(**learner_params)

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

    settings = _get_optuna_settings(optuna_settings, learner_name, learner.__class__.__name__)

    # Set Optuna logging verbosity if specified
    verbosity = settings.get("verbosity")
    if verbosity is not None:
        optuna.logging.set_verbosity(verbosity)

    # Pre-create KFold object for cross-validation during tuning (fixed random state for reproducibility)
    cv = KFold(n_splits=n_folds_tune, shuffle=True, random_state=42)

    # Create the study
    study = _create_study(settings, learner_name)

    # Create the objective function
    objective = _create_objective(param_grid_func, learner, x, y, cv, scoring_method, n_jobs_cv, learner_name)

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
    # drop learner_name prefix from keys and only keep parameters for the current learner
    best_params = {
        key.replace(f"{learner_name}_", ""): value
        for key, value in study.best_trial.params.items()
        if key.startswith(f"{learner_name}_")
    }
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
