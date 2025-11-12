import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import doubleml as dml
from doubleml.plm.datasets import make_pliv_CHS2015

from .test_dml_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_pliv_optuna_tune(sampler_name, optuna_sampler):
    """Test PLIV with ml_l, ml_m, ml_r nuisance models."""

    np.random.seed(3144)
    dml_data = make_pliv_CHS2015(n_obs=100, dim_x=15, dim_z=3)

    ml_l = DecisionTreeRegressor(random_state=123, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=456, max_depth=5, min_samples_leaf=4)
    ml_r = DecisionTreeRegressor(random_state=789, max_depth=5, min_samples_leaf=4)

    dml_pliv = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=2)

    optuna_params = _build_param_space(dml_pliv, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_pliv.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    for learner_name in dml_pliv.params_names:
        tuned_params = tune_res[0][learner_name].best_params_
        _assert_tree_params(tuned_params)
