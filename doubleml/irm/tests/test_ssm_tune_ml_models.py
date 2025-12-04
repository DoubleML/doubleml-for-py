import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_ssm_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


# test NotImplementedError for nonignorable score
@pytest.mark.ci
def test_doubleml_ssm_optuna_tune_not_implemented():
    np.random.seed(3149)
    dml_data = make_ssm_data(n_obs=500, dim_x=10, mar=False)

    ml_g = DecisionTreeRegressor(random_state=321)
    ml_pi = DecisionTreeClassifier(random_state=654)
    ml_m = DecisionTreeClassifier(random_state=987)

    dml_ssm = dml.DoubleMLSSM(dml_data, ml_g=ml_g, ml_pi=ml_pi, ml_m=ml_m, n_folds=2, score="nonignorable")

    optuna_params = _build_param_space(dml_ssm, _small_tree_params)
    optuna_settings = _basic_optuna_settings({"sampler": "TPESampler"})

    with pytest.raises(NotImplementedError, match="Optuna tuning for nonignorable score is not implemented yet."):
        dml_ssm.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings)


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_ssm_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3149)
    dml_data = make_ssm_data(n_obs=500, dim_x=10, mar=True)

    ml_g = DecisionTreeRegressor(random_state=321)
    ml_pi = DecisionTreeClassifier(random_state=654)
    ml_m = DecisionTreeClassifier(random_state=987)

    dml_ssm = dml.DoubleMLSSM(dml_data, ml_g=ml_g, ml_pi=ml_pi, ml_m=ml_m, n_folds=2, score="missing-at-random")
    dml_ssm.fit()
    untuned_score = dml_ssm.evaluate_learners()

    optuna_params = _build_param_space(dml_ssm, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_ssm.tune_ml_models(
        ml_param_space=optuna_params, optuna_settings=optuna_settings, set_as_params=True, return_tune_res=True
    )

    dml_ssm.fit()
    tuned_score = dml_ssm.evaluate_learners()

    for learner_name in dml_ssm.params_names:
        tuned_params = tune_res[0][learner_name].best_params
        _assert_tree_params(tuned_params)

        # ensure tuning improved RMSE
        assert tuned_score[learner_name] < untuned_score[learner_name]
