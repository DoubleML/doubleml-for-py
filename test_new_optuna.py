"""
Quick test of the new Optuna tuning implementation.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml import DoubleMLData

# Generate simple data
np.random.seed(123)
n = 100
x = np.random.normal(size=(n, 3))
d = np.random.binomial(1, 0.5, n)
y = 0.5 * d + x[:, 0] + np.random.normal(0, 0.5, n)

df = pd.DataFrame(np.column_stack((y, d, x)), columns=["y", "d", "X1", "X2", "X3"])
dml_data = DoubleMLData(df, "y", ["d"], ["X1", "X2", "X3"])

ml_l = DecisionTreeRegressor(random_state=123)
ml_m = DecisionTreeClassifier(random_state=456)

dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

try:
    import optuna

    print("Testing Optuna tuning with callable specification...")
    def ml_l_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_l_max_depth", 1, 5),
            "min_samples_leaf": trial.suggest_int("ml_l_min_samples_leaf", 1, 10),
        }

    def ml_m_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_m_max_depth", 1, 5),
            "min_samples_leaf": trial.suggest_int("ml_m_min_samples_leaf", 1, 10),
        }

    param_grids_callable = {"ml_l": ml_l_params, "ml_m": ml_m_params}

    dml_plr.tune_optuna(
        param_grids=param_grids_callable,
        optuna_settings={
            "n_trials": 5,
            "show_progress_bar": False,
            "sampler": optuna.samplers.RandomSampler(seed=42),
        },
        n_folds_tune=2,
    )

    print("[OK] Tuning with callables completed successfully!")
    print(f"  ml_l params: {dml_plr.params['ml_l']['d'][0][0]}")
    print(f"  ml_m params: {dml_plr.params['ml_m']['d'][0][0]}")

    # Verify all folds have the same parameters
    ml_l_params = dml_plr.params['ml_l']['d'][0]
    ml_m_params = dml_plr.params['ml_m']['d'][0]

    assert all(p == ml_l_params[0] for p in ml_l_params), "ml_l parameters differ across folds!"
    assert all(p == ml_m_params[0] for p in ml_m_params), "ml_m parameters differ across folds!"
    print("[OK] All folds use the same parameters (as expected)")

    dml_plr.fit()
    print(f"[OK] Model fitted successfully. Coefficient: {dml_plr.coef[0]:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! [SUCCESS]")
    print("=" * 60)

except ImportError:
    print("Optuna is not installed. Skipping test.")
except Exception as e:
    print(f"[FAILED] Test failed with error: {e}")
    import traceback
    traceback.print_exc()
