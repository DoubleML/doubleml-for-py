import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_iivm_data

from .test_dml_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_iivm_optuna_tune(sampler_name, optuna_sampler):
    """Test IIVM with ml_g0, ml_g1, ml_m, ml_r0, ml_r1 nuisance models."""

    np.random.seed(3143)
    dml_data = make_iivm_data(n_obs=100, dim_x=5)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
    ml_r = DecisionTreeClassifier(random_state=789, max_depth=5, min_samples_leaf=4)

    dml_iivm = dml.DoubleMLIIVM(dml_data, ml_g, ml_m, ml_r, n_folds=2, subgroups={"always_takers": True, "never_takers": True})

    optuna_params = {
        "ml_g0": _small_tree_params,
        "ml_g1": _small_tree_params,
        "ml_m": _small_tree_params,
        "ml_r0": _small_tree_params,
        "ml_r1": _small_tree_params,
    }

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_iivm.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    tuned_params_g0 = tune_res[0]["ml_g0"].best_params_
    tuned_params_g1 = tune_res[0]["ml_g1"].best_params_
    tuned_params_m = tune_res[0]["ml_m"].best_params_
    tuned_params_r0 = tune_res[0]["ml_r0"].best_params_
    tuned_params_r1 = tune_res[0]["ml_r1"].best_params_

    _assert_tree_params(tuned_params_g0)
    _assert_tree_params(tuned_params_g1)
    _assert_tree_params(tuned_params_m)
    _assert_tree_params(tuned_params_r0)
    _assert_tree_params(tuned_params_r1)
