import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.did.datasets import make_did_SZ2020
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_doubleml_did_cs_optuna_tune(sampler_name, optuna_sampler, score):
    np.random.seed(3151)
    dml_data = make_did_SZ2020(
        n_obs=500,
        dgp_type=4,
        cross_sectional_data=True,
        return_type="DoubleMLDIDData",
    )

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=1)  # underfit
    if score == "observational":
        ml_m = DecisionTreeClassifier(random_state=654)
        dml_did_cs = dml.DoubleMLDIDCS(dml_data, ml_g, ml_m, score=score, n_folds=5)
    else:
        dml_did_cs = dml.DoubleMLDIDCS(dml_data, ml_g, score=score, n_folds=5)
    dml_did_cs.fit()
    untuned_score = dml_did_cs.evaluate_learners()

    optuna_params = _build_param_space(dml_did_cs, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_did_cs.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    dml_did_cs.fit()
    tuned_score = dml_did_cs.evaluate_learners()

    for learner_name in dml_did_cs.params_names:
        tuned_params = tune_res[0][learner_name].best_params
        _assert_tree_params(tuned_params)

        # ensure tuning improved RMSE
        assert tuned_score[learner_name] < untuned_score[learner_name]
