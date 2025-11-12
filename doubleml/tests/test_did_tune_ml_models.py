import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.did.datasets import make_did_SZ2020

from .test_dml_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_doubleml_did_optuna_tune(sampler_name, optuna_sampler, score):
    """Test DID with ml_g0, ml_g1 (and ml_m for observational score) nuisance models."""

    np.random.seed(3150)
    dml_data = make_did_SZ2020(n_obs=100, dgp_type=1, return_type="DoubleMLDIDData")

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    if score == "observational":
        ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
        dml_did = dml.DoubleMLDID(dml_data, ml_g, ml_m, score=score, n_folds=2)
    else:
        dml_did = dml.DoubleMLDID(dml_data, ml_g, score=score, n_folds=2)

    optuna_params = _build_param_space(dml_did, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_did.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    for learner_name in dml_did.params_names:
        tuned_params = tune_res[0][learner_name].best_params_
        _assert_tree_params(tuned_params)
