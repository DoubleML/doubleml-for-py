import re

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso

from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.utils._tune_optuna import (
    _check_tuning_inputs,
    _create_objective,
    _default_optuna_settings,
    _dml_tune_optuna,
    _get_optuna_settings,
    _resolve_optuna_scoring,
    resolve_optuna_cv,
)

np.random.seed(42)
data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5, return_type="DataFrame")
dml_data = DoubleMLData(data, "y", "d")
ml_l = Lasso()
ml_m = Lasso()
dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)


def ml_l_params(trial):
    return {"alpha": trial.suggest_float("alpha", 0.01, 0.1)}


def ml_m_params(trial):
    return {"alpha": trial.suggest_float("alpha", 0.01, 0.1)}


valid_param_space = {"ml_l": ml_l_params, "ml_m": ml_m_params}


@pytest.mark.ci
@pytest.mark.parametrize(
    "ml_param_space, msg",
    [
        (None, "ml_param_space must be a non-empty dictionary."),
        (
            {"ml_l": ml_l_params, "invalid_key": ml_m_params},
            r"Invalid ml_param_space keys for DoubleMLPLR: invalid_key. Valid keys are: ml_l, ml_m.",
        ),
        (
            {"ml_l": ml_l_params, "ml_m": "invalid"},
            "Parameter space for 'ml_m' must be a callable function that takes a trial and returns a dict. Got str.",
        ),
    ],
)
def test_tune_ml_models_invalid_param_space(ml_param_space, msg):
    with pytest.raises(ValueError if "keys" in msg or "non-empty" in msg else TypeError, match=msg):
        dml_plr.tune_ml_models(ml_param_space)


@pytest.mark.ci
@pytest.mark.parametrize(
    "scoring_methods, exc, msg",
    [
        ("invalid", TypeError, "scoring_methods must be provided as a dictionary keyed by learner name."),
        (
            {"ml_l": "accuracy", "invalid_learner": "accuracy"},
            ValueError,
            r"Invalid scoring_methods keys for DoubleMLPLR: invalid_learner. Valid keys are: ml_l, ml_m.",
        ),
        (
            {"ml_l": 123},
            TypeError,
            r"scoring_method must be None, a string, a callable, accepted by scikit-learn. Got int for learner 'ml_l'.",
        ),
    ],
)
def test_tune_ml_models_invalid_scoring_methods(scoring_methods, exc, msg):
    with pytest.raises(exc, match=re.escape(msg)):
        dml_plr.tune_ml_models(valid_param_space, scoring_methods=scoring_methods)


@pytest.mark.ci
@pytest.mark.parametrize(
    "cv, msg",
    [
        ("invalid", "cv must not be provided as a string. Pass an integer or a cross-validation splitter."),
        (1, "The number of folds used for tuning must be at least two. 1 was passed."),
    ],
)
def test_tune_ml_models_invalid_cv(cv, msg):
    with pytest.raises(ValueError if "folds" in msg else TypeError, match=msg):
        dml_plr.tune_ml_models(valid_param_space, cv=cv)


@pytest.mark.ci
@pytest.mark.parametrize(
    "set_as_params, msg",
    [
        ("invalid", "set_as_params must be True or False. Got invalid."),
        (None, "set_as_params must be True or False. Got None."),
    ],
)
def test_tune_ml_models_invalid_set_as_params(set_as_params, msg):
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune_ml_models(valid_param_space, set_as_params=set_as_params)


@pytest.mark.ci
@pytest.mark.parametrize(
    "return_tune_res, msg",
    [
        ("invalid", "return_tune_res must be True or False. Got invalid."),
        (None, "return_tune_res must be True or False. Got None."),
    ],
)
def test_tune_ml_models_invalid_return_tune_res(return_tune_res, msg):
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune_ml_models(valid_param_space, return_tune_res=return_tune_res)


@pytest.mark.ci
@pytest.mark.parametrize(
    "optuna_settings, msg",
    [
        ("invalid", "optuna_settings must be a dict or None. Got <class 'str'>."),
        (
            {"invalid_key": "value"},
            r"Invalid optuna_settings keys for DoubleMLPLR: invalid_key. Valid learner-specific keys are: ml_l, ml_m.",
        ),
        ({"ml_l": "invalid"}, "Optuna settings for 'ml_l' must be a dict."),
    ],
)
def test_tune_ml_models_invalid_optuna_settings(optuna_settings, msg):
    with pytest.raises(TypeError if "dict" in msg else ValueError, match=msg):
        dml_plr.tune_ml_models(valid_param_space, optuna_settings=optuna_settings)


# add test for giving non iterable cv object
@pytest.mark.ci
def test_tune_ml_models_non_iterable_cv():
    class NonIterableCV:
        pass

    non_iterable_cv = NonIterableCV()
    msg = re.escape(
        "cv must be an integer >= 2, a scikit-learn cross-validation splitter, "
        "or an iterable of (train_indices, test_indices) pairs."
    )
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune_ml_models(valid_param_space, cv=non_iterable_cv)


@pytest.mark.ci
def test_resolve_optuna_cv_invalid_iterable_pairs():
    invalid_cv = [(np.array([0, 1]),)]
    msg = re.escape("cv iterable must yield (train_indices, test_indices) pairs.")
    with pytest.raises(TypeError, match=msg):
        resolve_optuna_cv(invalid_cv)


@pytest.mark.ci
def test_resolve_optuna_scoring_unknown_estimator_type():
    class GenericEstimator(BaseEstimator):
        def fit(self, x, y):
            return self

        def set_params(self, **params):
            return self

    msg = (
        "No scoring method provided and estimator type could not be inferred. "
        "Please provide a scoring_method for learner 'ml_l'."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        _resolve_optuna_scoring(None, GenericEstimator(), "ml_l")


@pytest.mark.ci
def test_check_tuning_inputs_mismatched_dimensions():
    x = np.zeros((3, 2))
    y = np.zeros(5)
    with pytest.raises(
        ValueError,
        match=re.escape("Features and target must contain the same number of observations for learner 'ml_l'."),
    ):
        _check_tuning_inputs(y, x, Lasso(), lambda trial: {}, "neg_mean_squared_error", 2, "ml_l")


@pytest.mark.ci
def test_check_tuning_inputs_empty_target():
    x = np.zeros((0, 2))
    y = np.zeros(0)
    with pytest.raises(
        ValueError,
        match=re.escape("Empty target passed to Optuna tuner for learner 'ml_l'."),
    ):
        _check_tuning_inputs(y, x, Lasso(), lambda trial: {}, "neg_mean_squared_error", 2, "ml_l")


@pytest.mark.ci
def test_check_tuning_inputs_invalid_learner_interface():
    class BadLearner:
        def set_params(self, **kwargs):
            return self

    x = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(
        TypeError,
        match=re.escape("Learner 'ml_l' must implement fit and set_params to be tuned with Optuna."),
    ):
        _check_tuning_inputs(y, x, BadLearner(), lambda trial: {}, "neg_mean_squared_error", 2, "ml_l")


@pytest.mark.ci
def test_check_tuning_inputs_non_callable_param_grid():
    x = np.zeros((5, 2))
    y = np.zeros(5)
    msg = "param_grid must be a callable function that takes a trial and returns a dict. " "Got str for learner 'ml_l'."
    with pytest.raises(TypeError, match=re.escape(msg)):
        _check_tuning_inputs(y, x, Lasso(), "not-callable", "neg_mean_squared_error", 2, "ml_l")


@pytest.mark.ci
def test_get_optuna_settings_requires_dict():
    with pytest.raises(TypeError, match="optuna_settings must be a dict or None."):
        _get_optuna_settings("invalid", "ml_l")


@pytest.mark.ci
def test_get_optuna_settings_returns_default_copy_for_none():
    resolved_a = _get_optuna_settings(None, "ml_l")
    resolved_b = _get_optuna_settings(None, "ml_l")
    # Ensure defaults are preserved
    for key, value in _default_optuna_settings().items():
        assert resolved_a[key] == value
    # Ensure copies are independent
    resolved_a["n_trials"] = 5
    assert resolved_b["n_trials"] == _default_optuna_settings()["n_trials"]


@pytest.mark.ci
def test_get_optuna_settings_validates_study_kwargs_type():
    with pytest.raises(TypeError, match="study_kwargs must be a dict."):
        _get_optuna_settings({"study_kwargs": "invalid"}, "ml_l")


@pytest.mark.ci
def test_get_optuna_settings_validates_optimize_kwargs_type():
    with pytest.raises(TypeError, match="optimize_kwargs must be a dict."):
        _get_optuna_settings({"optimize_kwargs": "invalid"}, "ml_l")


@pytest.mark.ci
def test_get_optuna_settings_validates_callbacks_type():
    with pytest.raises(TypeError, match="callbacks must be a sequence of callables or None."):
        _get_optuna_settings({"callbacks": "invalid"}, "ml_l")


@pytest.mark.ci
def test_create_objective_requires_dict_params():
    x = np.asarray(dml_data.x)
    y = np.asarray(dml_data.y)

    def bad_param_func(trial):
        return ["not-a-dict"]

    objective = _create_objective(
        bad_param_func,
        Lasso(),
        x,
        y,
        resolve_optuna_cv(2),
        "neg_mean_squared_error",
    )
    msg = (
        "param function must return a dict. Got list. Example: def params(trial): "
        "return {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        objective(None)


@pytest.mark.ci
def test_dml_tune_optuna_raises_when_no_trials_complete():
    class FailingRegressor(BaseEstimator, RegressorMixin):
        def fit(self, x, y):
            raise ValueError("fail")

        def predict(self, x):
            return np.zeros(x.shape[0])

    x = np.asarray(dml_data.x)
    y = np.asarray(dml_data.y)
    optuna_settings = {
        "n_trials": 1,
        "catch": (ValueError,),
        "study_kwargs": {},
        "optimize_kwargs": {},
    }
    with pytest.raises(
        RuntimeError,
        match="Optuna optimization failed to produce any complete trials.",
    ):
        _dml_tune_optuna(
            y,
            x,
            FailingRegressor(),
            lambda trial: {},
            "neg_mean_squared_error",
            2,
            optuna_settings,
            "ml_l",
            "ml_l",
        )
