import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_apo_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3146)
    theta = 0.5
    dml_data = make_irm_data(n_obs=500, dim_x=6, theta=theta)

    ml_g = DecisionTreeRegressor(random_state=321)
    ml_m = DecisionTreeClassifier(random_state=654)

    dml_apo = dml.DoubleMLAPO(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=2, treatment_level=1)
    dml_apo.fit()
    untuned_score = dml_apo.evaluate_learners()

    optuna_params = _build_param_space(dml_apo, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_apo.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    dml_apo.fit()
    tuned_score = dml_apo.evaluate_learners()

    for learner_name in dml_apo.params_names:
        tuned_params = tune_res[0][learner_name].best_params
        _assert_tree_params(tuned_params)

        # ensure tuning improved RMSE
        assert tuned_score[learner_name] < untuned_score[learner_name]
