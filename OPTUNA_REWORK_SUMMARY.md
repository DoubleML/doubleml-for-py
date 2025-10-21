# Optuna Tuning Implementation - Simplified Summary

## Overview

The Optuna tuning integration in DoubleML now follows a simple, consistent design:

1. **Single global tuning**: Tune once on the whole dataset using cross-validation.
2. **Shared hyperparameters**: The same optimal hyperparameters are reused for every fold.
3. **Native Optuna sampling**: Parameters are specified via callables that delegate to Optuna's `trial.suggest_*` APIs.
4. **Streamlined API**: Only callable specifications are supported, reducing branching logic and surprises.

## Key Changes

### 1. `_dml_tune_optuna()` (doubleml/utils/_estimation.py)
- Runs a single Optuna study on the full dataset.
- Evaluates candidates via `sklearn.model_selection.cross_validate` to respect the requested scoring function.
- Re-fits the best estimator on each fold's training data to mimic the GridSearchCV API.
- Shares the study object and trial history across folds for downstream inspection.

### 2. Search-space callables
- Users provide one function per learner that maps a trial to a parameter dictionary.
- Simplifies the API compared with nested dictionaries of lambdas.
- Keeps Optuna's sampling logic in user land while DoubleML handles evaluation.

### 3. Learner-specific Optuna settings
- `_dml_tune` forwards an explicit `learner_name` so overrides can be keyed by the entries in `param_grids` (for example `"ml_l"`, `"ml_m"`).
- Falls back to the estimator class name when no learner-specific block is provided, preserving flexibility.

## Documentation Updates

- `DoubleML.tune()` docstring now documents callable-only Optuna grids and clarifies the override semantics for `optuna_settings`.
- Example and helper scripts (`examples/optuna_tuning_example.py`, `test_new_optuna.py`, `check_params_structure.py`) were updated to use callable grids exclusively.

## Example

```python
def ml_l_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_l_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("ml_l_max_depth", 3, 15),
        "max_features": trial.suggest_categorical("ml_l_max_features", ["sqrt", 0.5, 0.7]),
    }


def ml_m_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_m_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("ml_m_max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("ml_m_min_samples_leaf", 1, 20),
    }


param_grids = {"ml_l": ml_l_params, "ml_m": ml_m_params}

optuna_settings = {
    "n_trials": 50,
    "sampler": optuna.samplers.TPESampler(seed=42),
    "show_progress_bar": True,
    "ml_l": {"n_trials": 40},  # learner-specific override via param_grids key
}

dml_plr.tune_optuna(
    param_grids=param_grids,
    optuna_settings=optuna_settings,
    n_folds_tune=3,
)
```

## Testing

- `pytest doubleml/tests/test_optuna_tune.py` verifies core behaviour.
- Supplementary scripts demonstrate callable grids and ensure tuned parameters are identical across folds.

## Benefits

- Less code and fewer branching paths to maintain.
- Immediate, informative feedback when parameter grids are misconfigured.
- Consistent, performant Optuna integration aligned with the main DoubleML package.
