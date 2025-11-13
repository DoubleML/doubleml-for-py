import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import doubleml as dml
from doubleml.irm.datasets import make_iivm_data

from .test_dml_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _small_tree_params,
)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_lpq_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3148)
    dml_data = make_iivm_data(n_obs=500, dim_x=5)

    ml_g = DecisionTreeClassifier(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_lpq = dml.DoubleMLLPQ(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=2)
    dml_lpq.fit()
    untuned_score = dml_lpq.evaluate_learners()

    optuna_params = _build_param_space(dml_lpq, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_lpq.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)

    dml_lpq.fit()
    tuned_score = dml_lpq.evaluate_learners()

    for learner_name in dml_lpq.params_names:
        tuned_params = tune_res[0][learner_name].best_params_
        _assert_tree_params(tuned_params)

        # ensure tuning improved RMSE
        assert tuned_score[learner_name] < untuned_score[learner_name]
