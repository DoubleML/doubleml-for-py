import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDBinary
from doubleml.did.datasets import make_did_CS2021
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _build_param_space,
    _select_binary_periods,
    _small_tree_params,
)


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_doubleml_did_binary_optuna_tune(sampler_name, optuna_sampler, score):
    np.random.seed(3152)
    df_panel = make_did_CS2021(
        n_obs=500,
        dgp_type=4,
        include_never_treated=True,
        time_type="float",
        n_periods=4,
        n_pre_treat_periods=2,
    )
    panel_data = DoubleMLPanelData(
        df_panel,
        y_col="y",
        d_cols="d",
        id_col="id",
        t_col="t",
        x_cols=["Z1", "Z2", "Z3", "Z4"],
    )

    g_value, t_value_pre, t_value_eval = _select_binary_periods(panel_data)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=1)  # underfit
    ml_m = DecisionTreeClassifier(random_state=654)
    dml_did_binary = DoubleMLDIDBinary(
        obj_dml_data=panel_data,
        g_value=g_value,
        t_value_pre=t_value_pre,
        t_value_eval=t_value_eval,
        ml_g=ml_g,
        ml_m=ml_m,
        score=score,
        n_folds=5,
    )
    dml_did_binary.fit()
    untuned_score = dml_did_binary.evaluate_learners()

    optuna_params = _build_param_space(dml_did_binary, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    tune_res = dml_did_binary.tune_ml_models(
        ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True
    )

    dml_did_binary.fit()
    tuned_score = dml_did_binary.evaluate_learners()

    for learner_name in dml_did_binary.params_names:
        tuned_params = tune_res[0][learner_name].best_params
        _assert_tree_params(tuned_params)

        # ensure tuning improved RMSE
        assert tuned_score[learner_name] < untuned_score[learner_name]
