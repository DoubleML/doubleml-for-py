# Optuna Tuning Refactoring - Migration Guide

## Overview

The Optuna hyperparameter tuning implementation in DoubleML has been refactored to:

1. **Decouple Optuna from sklearn-based tuning** - Separate method `tune_optuna()` instead of `tune(search_mode="optuna")`
2. **Simplify parameter specification** - Use callable functions instead of dict with lambdas
3. **Improve code organization** - All Optuna-specific code moved to `doubleml/utils/_tune_optuna.py`
4. **Better structure** - Helper functions `_create_study()`, `_create_objective()` for clarity

## What Changed

### 1. New Method: `tune_optuna()`

**Before:**
```python
dml_plr.tune(
    param_grids=param_grids,
    search_mode="optuna",
    optuna_settings=optuna_settings
)
```

**After:**
```python
dml_plr.tune_optuna(
    param_grids=param_grids,
    optuna_settings=optuna_settings
)
```

### 2. Parameter Specification Format

**Before (dict with lambdas):**
```python
param_grid_lgbm = {
    "ml_l": {
        "n_estimators": lambda trial, name: trial.suggest_int(name, 100, 500, step=50),
        "num_leaves": lambda trial, name: trial.suggest_int(name, 20, 256),
        "learning_rate": lambda trial, name: trial.suggest_float(name, 0.01, 0.3, log=True),
        "min_child_samples": lambda trial, name: trial.suggest_int(name, 5, 100),
    },
    "ml_m": {
        "n_estimators": lambda trial, name: trial.suggest_int(name, 100, 500, step=50),
        "num_leaves": lambda trial, name: trial.suggest_int(name, 20, 256),
        "learning_rate": lambda trial, name: trial.suggest_float(name, 0.01, 0.3, log=True),
        "min_child_samples": lambda trial, name: trial.suggest_int(name, 5, 100),
    },
}
```

**After (callable functions):**
```python
def ml_l_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_l_n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("ml_l_num_leaves", 20, 256),
        "learning_rate": trial.suggest_float("ml_l_learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("ml_l_min_child_samples", 5, 100),
    }

def ml_m_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_m_n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("ml_m_num_leaves", 20, 256),
        "learning_rate": trial.suggest_float("ml_m_learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("ml_m_min_child_samples", 5, 100),
    }

param_grids = {
    "ml_l": ml_l_params,
    "ml_m": ml_m_params,
}
```

### 3. Benefits of New API

**Cleaner Syntax:**
- No need to pass `name` parameter to lambda functions
- Parameter names are explicit in the suggest calls
- More readable and maintainable

**Better IDE Support:**
- Functions can have docstrings
- Better auto-completion
- Easier to debug

**More Flexible:**
- Can add conditional logic within the function
- Can share common parameter definitions
- Can add validation or constraints

## Code Organization

### File Structure

**New files:**
- `doubleml/utils/_tune_optuna.py` - All Optuna-specific code

**Modified files:**
- `doubleml/double_ml.py` - Added `tune_optuna()` method, removed Optuna from `tune()`
- `doubleml/utils/_estimation.py` - Removed Optuna code, kept sklearn-based tuning
- `doubleml/plm/plr.py` - Added `_nuisance_tuning_optuna()` method

### Helper Functions

The new `_tune_optuna.py` module includes:

1. **`_OptunaSearchResult`** - Result container mimicking GridSearchCV
2. **`_create_study(settings)`** - Creates or retrieves Optuna study
3. **`_create_objective(param_grid_func, learner, x, y, cv, scoring_method, n_jobs_cv)`** - Creates objective function
4. **`_dml_tune_optuna(...)`** - Main tuning logic
5. **`_resolve_optuna_settings(optuna_settings)`** - Merges settings with defaults
6. **`_select_optuna_settings(optuna_settings, learner_names)`** - Selects learner-specific settings

## Migration Steps

### For Users

1. **Replace `tune()` calls with `tune_optuna()`**:
   ```python
   # Old
   dml_plr.tune(param_grids, search_mode="optuna", optuna_settings=settings)
   
   # New
   dml_plr.tune_optuna(param_grids, optuna_settings=settings)
   ```

2. **Update parameter specifications**:
   - Change from lambda dict to callable functions
   - Remove `name` parameter from lambda
   - Use explicit parameter names in `trial.suggest_*()` calls

3. **Update imports** (if directly importing tuning functions):
   ```python
   # Old
   from doubleml.utils._estimation import _dml_tune_optuna
   
   # New
   from doubleml.utils._tune_optuna import _dml_tune_optuna
   ```

### For Developers

If you've implemented custom DoubleML models:

1. **Update `_nuisance_tuning()` signature** - Remove `optuna_settings` parameter

2. **Implement `_nuisance_tuning_optuna()` method**:
   ```python
   def _nuisance_tuning_optuna(
       self,
       param_grids,
       scoring_methods,
       n_folds_tune,
       n_jobs_cv,
       optuna_settings,
   ):
       from ..utils._tune_optuna import _dml_tune_optuna
       
       # Your tuning logic here
       # Use param_grids as callables instead of dicts with lambdas
       ...
   ```

## Complete Example

```python
import numpy as np
import doubleml as dml
from doubleml import DoubleMLData
from doubleml.datasets import make_plr_CCDDHNR2018
from lightgbm import LGBMRegressor
import optuna

# Generate data
np.random.seed(42)
data = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, return_type="DataFrame")
x_cols = [col for col in data.columns if col.startswith("X")]
dml_data = DoubleMLData(data, "y", "d", x_cols)

# Initialize model
ml_l = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
ml_m = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2)

# Define parameter grid functions (NEW API)
def ml_l_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
    }

def ml_m_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
    }

param_grids = {"ml_l": ml_l_params, "ml_m": ml_m_params}

# Configure Optuna
optuna_settings = {
    "n_trials": 20,
    "sampler": optuna.samplers.TPESampler(seed=42),
    "show_progress_bar": False,
}

# Tune with Optuna (NEW METHOD)
dml_plr.tune_optuna(
    param_grids=param_grids,
    optuna_settings=optuna_settings,
    n_folds_tune=3,
    set_as_params=True,
)

# Fit and get results
dml_plr.fit()
print(f"Treatment effect: {dml_plr.coef[0]:.4f} (SE: {dml_plr.se[0]:.4f})")
```

## Backwards Compatibility

**Breaking Changes:**
- `tune(search_mode="optuna")` is no longer supported
- Old lambda-based parameter specification format not supported by `tune_optuna()`

**Migration Timeline:**
- The old API should be deprecated with clear warnings
- Users should migrate to `tune_optuna()` with new parameter format

## Testing

Make sure to test:
1. All Optuna samplers (TPE, GP, Random, NSGA-II, BruteForce)
2. Parameter specification with different types (int, float, categorical)
3. Learner-specific settings overrides
4. Study creation and reuse
5. Integration with different DoubleML models (PLR, IRM, etc.)

## Documentation

Update:
1. User guide with new API examples
2. API reference for `tune_optuna()` method
3. Migration guide for users
4. Examples in notebooks and scripts

## Summary

The refactoring provides:
- ✅ Cleaner separation between sklearn and Optuna tuning
- ✅ More intuitive parameter specification API
- ✅ Better code organization and maintainability
- ✅ Improved helper function structure
- ✅ Better testability and extensibility
