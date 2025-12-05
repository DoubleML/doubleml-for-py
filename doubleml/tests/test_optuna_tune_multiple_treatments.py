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
def test_doubleml_plr_optuna_multiple_treatments(sampler_name, optuna_sampler):
    np.random.seed(3141)
    alpha = 0.5
    data = make_plr_CCDDHNR2018(n_obs=500, dim_x=5, alpha=alpha, return_type="DataFrame")
    treats = ["d", "X1"]
    dml_data = dml.DoubleMLData(
        data, y_col="y", d_cols=treats, x_cols=[col for col in data.columns if col not in ["y", "d", "X1"]]
    )

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

    for i, _ in enumerate(treats):
        tuned_params_l = tune_res[i]["ml_l"].best_params
        tuned_params_m = tune_res[i]["ml_m"].best_params

        _assert_tree_params(tuned_params_l)
        _assert_tree_params(tuned_params_m)

        # ensure results contain optuna objects and best params
        assert isinstance(tune_res[i], dict)
        assert set(tune_res[i].keys()) == {"ml_l", "ml_m"}
        assert hasattr(tune_res[i]["ml_l"], "best_params")
        assert tune_res[i]["ml_l"].best_params["max_depth"] == tuned_params_l["max_depth"]
        assert hasattr(tune_res[i]["ml_m"], "best_params")
        assert tune_res[i]["ml_m"].best_params["max_depth"] == tuned_params_m["max_depth"]

        # ensure tuning improved RMSE
        assert tuned_score["ml_l"].squeeze()[i] < untuned_score["ml_l"].squeeze()[i]
        assert tuned_score["ml_m"].squeeze()[i] < untuned_score["ml_m"].squeeze()[i]
