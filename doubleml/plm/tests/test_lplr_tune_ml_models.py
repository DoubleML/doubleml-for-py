import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.plm.datasets import make_lplr_LZZ2020
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def score(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[DecisionTreeRegressor(random_state=567), None],
)
def ml_a(request):
    return request.param


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_lplr_optuna_tune(sampler_name, optuna_sampler, score, ml_a):
    np.random.seed(3141)
    alpha = 0.5
    dml_data = make_lplr_LZZ2020(n_obs=200, dim_x=15, alpha=alpha)

    ml_M = DecisionTreeClassifier(random_state=123)
    ml_t = DecisionTreeRegressor(random_state=234)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_lplr = dml.DoubleMLLPLR(
        dml_data,
        ml_M=ml_M,
        ml_t=ml_t,
        ml_m=ml_m,
        ml_a=ml_a,
        n_folds=2,
        n_folds_inner=2,
        score=score,
    )
    dml_lplr.fit()
    untuned_score = dml_lplr.evaluate_learners()

    optuna_params = {
        "ml_M": _small_tree_params,
        "ml_m": _small_tree_params,
        "ml_t": _small_tree_params,
        "ml_a": _small_tree_params,
    }

    tune_res = dml_lplr.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 5}),
        return_tune_res=True,
    )

    dml_lplr.fit()
    tuned_score = dml_lplr.evaluate_learners()

    tuned_params_M = tune_res[0]["ml_M"].best_params
    tuned_params_t = tune_res[0]["ml_t"].best_params
    tuned_params_m = tune_res[0]["ml_m"].best_params
    tuned_params_a = tune_res[0]["ml_a"].best_params

    _assert_tree_params(tuned_params_M)
    _assert_tree_params(tuned_params_t)
    _assert_tree_params(tuned_params_m)
    _assert_tree_params(tuned_params_a)

    # ensure results contain optuna objects and best params
    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == {"ml_M", "ml_m", "ml_t", "ml_a"}
    assert hasattr(tune_res[0]["ml_M"], "best_params")
    assert tune_res[0]["ml_M"].best_params["max_depth"] == tuned_params_M["max_depth"]
    assert hasattr(tune_res[0]["ml_t"], "best_params")
    assert tune_res[0]["ml_t"].best_params["max_depth"] == tuned_params_t["max_depth"]
    assert hasattr(tune_res[0]["ml_m"], "best_params")
    assert tune_res[0]["ml_m"].best_params["max_depth"] == tuned_params_m["max_depth"]
    assert hasattr(tune_res[0]["ml_a"], "best_params")
    assert tune_res[0]["ml_a"].best_params["max_depth"] == tuned_params_a["max_depth"]

    # ensure tuning improved RMSE #  not actually possible for ml_t as the targets are not available
    assert tuned_score["ml_M"] < untuned_score["ml_M"]
    assert tuned_score["ml_m"] < untuned_score["ml_m"]
    assert tuned_score["ml_a"] < untuned_score["ml_a"]
