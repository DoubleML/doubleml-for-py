"""
Quick check of parameter structure after tuning.
"""
import numpy as np
import optuna
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


param_grids = {"ml_l": ml_l_params, "ml_m": ml_m_params}

dml_plr.tune_optuna(
    param_grids=param_grids,
    optuna_settings={
        "n_trials": 5,
        "show_progress_bar": False,
        "sampler": optuna.samplers.RandomSampler(seed=123),
    },
    n_folds_tune=2,
)

print("Parameter structure:")
print("dml_plr.params:", dml_plr.params)
print("\nml_l params:", dml_plr.params['ml_l'])
print("\nml_m params:", dml_plr.params['ml_m'])
