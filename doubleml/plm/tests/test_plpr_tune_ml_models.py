import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import doubleml as dml
from doubleml.plm.datasets import make_plpr_CP2025
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="module", params=["cre_general", "cre_normal", "fd_exact", "wg_approx"])
def approach(request):
    return request.param


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_plr_optuna_tune(sampler_name, optuna_sampler, approach, score):
    np.random.seed(3141)
    df = make_plpr_CP2025(theta=0.5, dim_x=5)
    dml_data = dml.DoubleMLPanelData(
        df,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)
    ml_g = DecisionTreeRegressor(random_state=789) if score == "IV-type" else None

    dml_obj = dml.DoubleMLPLPR(
        dml_data,
        ml_l,
        ml_m,
        ml_g=ml_g,
        n_folds=2,
        score=score,
        approach=approach,
    )
    dml_obj.fit()
    untuned_score = dml_obj.evaluate_learners()

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    if score == "IV-type":
        optuna_params["ml_g"] = _small_tree_params

    tune_res = dml_obj.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    dml_obj.fit()
    tuned_score = dml_obj.evaluate_learners()

    best_param_dict = {
        "ml_l": tune_res[0]["ml_l"].best_params,
        "ml_m": tune_res[0]["ml_m"].best_params,
    }
    if score == "IV-type":
        best_param_dict["ml_g"] = tune_res[0]["ml_g"].best_params

    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == best_param_dict.keys()

    for params_name, params in best_param_dict.items():
        _assert_tree_params(params)

        assert hasattr(tune_res[0][params_name], "best_params")
        assert tune_res[0][params_name].best_params["max_depth"] == params["max_depth"]

    # ensure tuning improved RMSE
    assert tuned_score["ml_l"] < untuned_score["ml_l"]
    assert tuned_score["ml_m"] < untuned_score["ml_m"]
    if score == "IV-type":
        assert tuned_score["ml_g"] < untuned_score["ml_g"]
