import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_irm_data

from .test_dml_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_cvar_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3145)
    dml_data = make_irm_data(n_obs=100, dim_x=6)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_cvar = dml.DoubleMLCVAR(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=2)

    optuna_params = {"ml_g": _small_tree_params, "ml_m": _small_tree_params}

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_cvar.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    tuned_params_g = tune_res[0]["ml_g"].best_params_
    tuned_params_m = tune_res[0]["ml_m"].best_params_

    _assert_tree_params(tuned_params_g)
    _assert_tree_params(tuned_params_m)
