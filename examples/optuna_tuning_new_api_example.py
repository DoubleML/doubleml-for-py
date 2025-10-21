"""
Example script demonstrating the new Optuna tuning API for DoubleML.

This script shows how to use the new tune_optuna() method with the updated
parameter specification format.
"""

import numpy as np
import doubleml as dml
from doubleml import DoubleMLData
from doubleml.datasets import make_plr_CCDDHNR2018
from lightgbm import LGBMRegressor
import optuna

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("DoubleML Optuna Tuning Example - New API")
print("=" * 80)

# Generate data
np.random.seed(42)
n_obs = 500
n_vars = 20
data = make_plr_CCDDHNR2018(n_obs=n_obs, dim_x=n_vars, return_type="DataFrame")

# Prepare DoubleML data
x_cols = [col for col in data.columns if col.startswith("X")]
dml_data = DoubleMLData(data, "y", "d", x_cols)

# Initialize learners
ml_l = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
ml_m = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)

# Initialize model
dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

print(f"\nData: n={n_obs}, p={n_vars}")
print(f"Model: DoubleMLPLR with LightGBM learners")

# =============================================================================
# NEW API: Define parameter grids as functions
# =============================================================================

print("\n" + "-" * 80)
print("NEW API: Parameter specification as callable functions")
print("-" * 80)


def ml_l_params(trial):
    """
    Parameter grid function for the outcome model (ml_l).
    
    The function takes an Optuna trial object and returns a dictionary
    of hyperparameters to try for this trial.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def ml_m_params(trial):
    """
    Parameter grid function for the treatment model (ml_m).
    
    Same structure as ml_l_params but allows for different search spaces
    if needed for different learners.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


# Create param_grids dictionary
param_grids = {
    "ml_l": ml_l_params,
    "ml_m": ml_m_params,
}

print("\nParameter grid functions defined:")
print("  • ml_l_params(trial) -> dict of hyperparameters")
print("  • ml_m_params(trial) -> dict of hyperparameters")

# Configure Optuna settings
optuna_settings = {
    "n_trials": 15,  # Number of optimization trials
    "sampler": optuna.samplers.TPESampler(seed=42),  # Bayesian optimization sampler
    "show_progress_bar": False,
    "verbosity": optuna.logging.WARNING,
}

print("\nOptuna settings:")
print(f"  • n_trials: {optuna_settings['n_trials']}")
print(f"  • sampler: TPESampler (Tree-structured Parzen Estimator)")

# =============================================================================
# Run Optuna tuning with the new tune_optuna() method
# =============================================================================

print("\n" + "-" * 80)
print("Running Optuna hyperparameter tuning...")
print("-" * 80)

tune_res = dml_plr.tune_optuna(
    param_grids=param_grids,
    optuna_settings=optuna_settings,
    n_folds_tune=3,
    set_as_params=True,
    return_tune_res=True,
)

print("\n✓ Tuning complete!")

# Display tuning results
print("\n" + "-" * 80)
print("Tuning Results")
print("-" * 80)

# Extract study objects
study_ml_l = tune_res[0]["tune_res"]["l_tune"][0].study_
study_ml_m = tune_res[0]["tune_res"]["m_tune"][0].study_

print("\nOutcome model (ml_l) - Best parameters:")
for param_name, param_value in study_ml_l.best_params.items():
    print(f"  • {param_name}: {param_value}")
print(f"  Best score: {study_ml_l.best_value:.4f}")

print("\nTreatment model (ml_m) - Best parameters:")
for param_name, param_value in study_ml_m.best_params.items():
    print(f"  • {param_name}: {param_value}")
print(f"  Best score: {study_ml_m.best_value:.4f}")

# =============================================================================
# Fit the model with tuned parameters
# =============================================================================

print("\n" + "-" * 80)
print("Fitting DoubleML model with tuned parameters...")
print("-" * 80)

dml_plr.fit()

print("\n✓ Model fitted!")
print("\nCausal estimate (treatment effect):")
print(f"  • Coefficient: {dml_plr.coef[0]:.4f}")
print(f"  • Standard error: {dml_plr.se[0]:.4f}")
print(f"  • 95% CI: [{dml_plr.confint().values[0][0]:.4f}, {dml_plr.confint().values[0][1]:.4f}]")

# =============================================================================
# Compare with different samplers
# =============================================================================

print("\n" + "=" * 80)
print("Comparing Different Optuna Samplers")
print("=" * 80)

samplers_to_test = [
    ("TPE", optuna.samplers.TPESampler(seed=42)),
    ("Random", optuna.samplers.RandomSampler(seed=42)),
    ("GP", optuna.samplers.GPSampler(seed=42)),
]

results = []

for sampler_name, sampler in samplers_to_test:
    print(f"\nTesting {sampler_name} sampler...")
    
    # Re-initialize model
    ml_l = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
    ml_m = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
    dml_plr_test = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")
    
    # Configure Optuna with this sampler
    optuna_settings_test = {
        "n_trials": 10,
        "sampler": sampler,
        "show_progress_bar": False,
        "verbosity": optuna.logging.WARNING,
    }
    
    # Tune and fit
    dml_plr_test.tune_optuna(
        param_grids=param_grids,
        optuna_settings=optuna_settings_test,
        n_folds_tune=3,
        set_as_params=True,
    )
    dml_plr_test.fit()
    
    results.append({
        "sampler": sampler_name,
        "coef": dml_plr_test.coef[0],
        "se": dml_plr_test.se[0],
    })
    
    print(f"  ✓ {sampler_name}: θ̂ = {dml_plr_test.coef[0]:.4f} (SE = {dml_plr_test.se[0]:.4f})")

print("\n" + "-" * 80)
print("Summary of results across samplers:")
print("-" * 80)
for res in results:
    print(f"  {res['sampler']:10s}: θ̂ = {res['coef']:.4f} ± {res['se']:.4f}")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
