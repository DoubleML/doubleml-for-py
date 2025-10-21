"""
Example demonstrating the new Optuna tuning interface for DoubleML.

This example shows how to use Optuna's native sampling methods for hyperparameter tuning.
The key improvement is that tuning happens once on the whole dataset, and the same
optimal hyperparameters are used for all folds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml import DoubleMLData

# Generate synthetic data
np.random.seed(42)
n_obs = 500
n_vars = 10

# Generate features
x = np.random.normal(size=(n_obs, n_vars))
# Treatment assignment
d = np.random.binomial(1, 0.5, size=n_obs)
# Outcome
y = 0.5 * d + x[:, 0] + 0.5 * x[:, 1] + np.random.normal(scale=0.5, size=n_obs)

# Create DataFrame
x_cols = [f"X{i+1}" for i in range(n_vars)]
df = pd.DataFrame(np.column_stack((y, d, x)), columns=["y", "d"] + x_cols)

# Create DoubleML data object
dml_data = DoubleMLData(df, "y", ["d"], x_cols)

# Initialize learners
ml_l = RandomForestRegressor(random_state=123)
ml_m = RandomForestClassifier(random_state=456)

# Create DoubleML model
dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, score="partialling out")

# ============================================================================
# Example: Using callable specification (recommended)
# ============================================================================
print("=" * 80)
print("Callable specification for Optuna parameters")
print("=" * 80)

def ml_l_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_l_n_estimators", 10, 200, log=True),
        "max_depth": trial.suggest_int("ml_l_max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("ml_l_min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("ml_l_max_features", ["sqrt", "log2", None]),
    }


def ml_m_params(trial):
    return {
        "n_estimators": trial.suggest_int("ml_m_n_estimators", 10, 200, log=True),
        "max_depth": trial.suggest_int("ml_m_max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("ml_m_min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("ml_m_max_features", ["sqrt", "log2", None]),
    }


param_grids_callable = {"ml_l": ml_l_params, "ml_m": ml_m_params}

try:
    import optuna

    # Tune with Optuna using callable specs
    tune_res = dml_plr.tune_optuna(
        param_grids=param_grids_callable,
        optuna_settings={
            "n_trials": 30,
            "sampler": optuna.samplers.RandomSampler(seed=42),
            "show_progress_bar": False,
        },
        n_folds_tune=3,
        return_tune_res=True,
    )

    print("\nOptimal parameters found:")
    print("ml_l:", dml_plr.params["ml_l"]["d"][0][0])
    print("ml_m:", dml_plr.params["ml_m"]["d"][0][0])

    # Fit the model with tuned parameters
    dml_plr.fit()
    print(f"\nCoefficient: {dml_plr.coef[0]:.4f}")
    print(f"Standard error: {dml_plr.se[0]:.4f}")

except ImportError:
    print("Optuna is not installed. Please install it to run this example:")
    print("pip install optuna")

print("\n" + "=" * 80)
print("Benefits of the new implementation:")
print("- Tuning happens ONCE on the whole dataset using cross-validation")
print("- Same optimal hyperparameters are used for ALL folds")
print("- Uses Optuna's native sampling methods (no grid conversion)")
print("- More efficient and follows best practices for hyperparameter optimization")
print("=" * 80)
