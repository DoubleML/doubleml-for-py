import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_plr_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3141)
    alpha = 0.5
    dml_data = make_plr_CCDDHNR2018(n_obs=500, dim_x=5, alpha=alpha)

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")
    dml_plr.fit()
    untuned_score = dml_plr.evaluate_learners()

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    tune_res = dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    dml_plr.fit()
    tuned_score = dml_plr.evaluate_learners()

    tuned_params_l = tune_res[0]["ml_l"].best_params
    tuned_params_m = tune_res[0]["ml_m"].best_params

    _assert_tree_params(tuned_params_l)
    _assert_tree_params(tuned_params_m)

    # ensure results contain optuna objects and best params
    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == {"ml_l", "ml_m"}
    assert hasattr(tune_res[0]["ml_l"], "best_params")
    assert tune_res[0]["ml_l"].best_params["max_depth"] == tuned_params_l["max_depth"]
    assert hasattr(tune_res[0]["ml_m"], "best_params")
    assert tune_res[0]["ml_m"].best_params["max_depth"] == tuned_params_m["max_depth"]

    # ensure tuning improved RMSE
    assert tuned_score["ml_l"] < untuned_score["ml_l"]
    assert tuned_score["ml_m"] < untuned_score["ml_m"]


@pytest.mark.ci
def test_doubleml_plr_optuna_tune_with_ml_g():
    np.random.seed(3150)
    dml_data = make_plr_CCDDHNR2018(n_obs=200, dim_x=5, alpha=0.5)

    ml_l = DecisionTreeRegressor(random_state=11)
    ml_m = DecisionTreeRegressor(random_state=12)
    ml_g = DecisionTreeRegressor(random_state=13)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, ml_g, n_folds=2, score="IV-type")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params, "ml_g": _small_tree_params}

    tune_res = dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"n_trials": 1}),
        return_tune_res=True,
    )

    assert "ml_g" in tune_res[0]
    ml_g_res = tune_res[0]["ml_g"]
    assert ml_g_res.best_params is not None
