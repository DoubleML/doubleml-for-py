"""Tests for DoubleMLScalar.tune_ml_models() input validation and error handling."""

import re

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR
from doubleml.tests._utils_tune_optuna import _basic_optuna_settings, _small_tree_params

# ── Shared fixtures ────────────────────────────────────────────────────────────

np.random.seed(42)
_plr_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)
_irm_data = make_irm_data(n_obs=100, dim_x=5)


@pytest.fixture(scope="module")
def plr_model():
    """Fitted PLR scalar model for reuse across exception tests."""
    model = PLR(_plr_data)
    model.set_learners(
        ml_l=DecisionTreeRegressor(random_state=1),
        ml_m=DecisionTreeRegressor(random_state=2),
    )
    return model


@pytest.fixture(scope="module")
def irm_model():
    """Fitted IRM scalar model for reuse across exception tests."""
    model = IRM(_irm_data)
    model.set_learners(
        ml_g=DecisionTreeRegressor(random_state=1),
        ml_m=DecisionTreeClassifier(random_state=2),
    )
    return model


# ── ml_param_space validation ──────────────────────────────────────────────────


@pytest.mark.ci
@pytest.mark.parametrize(
    "ml_param_space, exc, msg",
    [
        (None, TypeError, "ml_param_space must be a dict. Got NoneType."),
        ({}, ValueError, "ml_param_space must be a non-empty dictionary."),
        (
            {"ml_l": "not-callable"},
            TypeError,
            "Parameter space for 'ml_l' must be a callable function that takes a trial and returns a dict. Got str.",
        ),
    ],
)
def test_scalar_tune_invalid_param_space(plr_model, ml_param_space, exc, msg):
    """tune_ml_models() raises on None, empty, or non-callable ml_param_space."""
    with pytest.raises(exc, match=re.escape(msg)):
        plr_model.tune_ml_models(ml_param_space, optuna_settings=_basic_optuna_settings())


@pytest.mark.ci
@pytest.mark.parametrize(
    "bad_key, model_name",
    [
        ("ml_z", "PLR"),
        ("ml_g0", "PLR"),
    ],
)
def test_scalar_tune_invalid_param_space_key_plr(plr_model, bad_key, model_name):
    """_expand_tuning_param_space() raises ValueError for keys not valid for PLR."""
    with pytest.raises(ValueError, match=re.escape(f"Invalid key '{bad_key}' in ml_param_space")):
        plr_model.tune_ml_models(
            {bad_key: _small_tree_params},
            optuna_settings=_basic_optuna_settings(),
        )


@pytest.mark.ci
@pytest.mark.parametrize("bad_key", ["ml_l", "ml_z"])
def test_scalar_tune_invalid_param_space_key_irm(irm_model, bad_key):
    """_expand_tuning_param_space() raises ValueError for keys not valid for IRM."""
    with pytest.raises(ValueError, match=re.escape(f"Invalid key '{bad_key}' in ml_param_space")):
        irm_model.tune_ml_models(
            {bad_key: _small_tree_params},
            optuna_settings=_basic_optuna_settings(),
        )


# ── Boolean flag validation ────────────────────────────────────────────────────


@pytest.mark.ci
@pytest.mark.parametrize("set_as_params", ["invalid", None, 1])
def test_scalar_tune_invalid_set_as_params(plr_model, set_as_params):
    """tune_ml_models() raises TypeError for non-bool set_as_params."""
    msg = re.escape(f"set_as_params must be True or False. Got {str(set_as_params)}.")
    with pytest.raises(TypeError, match=msg):
        plr_model.tune_ml_models(
            {"ml_l": _small_tree_params},
            set_as_params=set_as_params,
            optuna_settings=_basic_optuna_settings(),
        )


@pytest.mark.ci
@pytest.mark.parametrize("return_tune_res", ["invalid", None, 1])
def test_scalar_tune_invalid_return_tune_res(plr_model, return_tune_res):
    """tune_ml_models() raises TypeError for non-bool return_tune_res."""
    msg = re.escape(f"return_tune_res must be True or False. Got {str(return_tune_res)}.")
    with pytest.raises(TypeError, match=msg):
        plr_model.tune_ml_models(
            {"ml_l": _small_tree_params},
            return_tune_res=return_tune_res,
            optuna_settings=_basic_optuna_settings(),
        )


# ── optuna_settings validation ─────────────────────────────────────────────────


@pytest.mark.ci
@pytest.mark.parametrize(
    "optuna_settings, exc, msg",
    [
        ("invalid", TypeError, "optuna_settings must be a dict or None. Got <class 'str'>."),
        (
            {"ml_g0": {"n_trials": 2}},
            ValueError,
            "Invalid optuna_settings keys for PLR: ml_g0. Valid learner-specific keys are:",
        ),
        ({"ml_l": "not-a-dict"}, TypeError, "Optuna settings for 'ml_l' must be a dict."),
    ],
)
def test_scalar_tune_invalid_optuna_settings_plr(plr_model, optuna_settings, exc, msg):
    """tune_ml_models() raises on non-dict, invalid learner key, or non-dict learner settings for PLR."""
    with pytest.raises(exc, match=re.escape(msg)):
        plr_model.tune_ml_models({"ml_l": _small_tree_params}, optuna_settings=optuna_settings)


@pytest.mark.ci
@pytest.mark.parametrize(
    "invalid_key",
    ["ml_l", "ml_z"],
)
def test_scalar_tune_invalid_optuna_settings_key_irm(irm_model, invalid_key):
    """tune_ml_models() raises ValueError for optuna_settings keys not valid for IRM."""
    with pytest.raises(ValueError, match=f"Invalid optuna_settings keys for IRM: {invalid_key}"):
        irm_model.tune_ml_models(
            {"ml_g": _small_tree_params, "ml_m": _small_tree_params},
            optuna_settings={invalid_key: {"n_trials": 2}},
        )


# ── cv validation (delegated to resolve_optuna_cv) ────────────────────────────


@pytest.mark.ci
@pytest.mark.parametrize(
    "cv, exc, msg",
    [
        ("invalid", TypeError, "cv must not be provided as a string."),
        (1, ValueError, "The number of folds used for tuning must be at least two. 1 was passed."),
    ],
)
def test_scalar_tune_invalid_cv(plr_model, cv, exc, msg):
    """tune_ml_models() raises for string cv or cv < 2."""
    with pytest.raises(exc, match=re.escape(msg)):
        plr_model.tune_ml_models(
            {"ml_l": _small_tree_params},
            cv=cv,
            optuna_settings=_basic_optuna_settings(),
        )


@pytest.mark.ci
def test_scalar_tune_non_iterable_cv(plr_model):
    """tune_ml_models() raises TypeError for a non-iterable cv object."""

    class NonIterableCV:
        pass

    msg = (
        "cv must be an integer >= 2, a scikit-learn cross-validation splitter, "
        "or an iterable of (train_indices, test_indices) pairs."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        plr_model.tune_ml_models(
            {"ml_l": _small_tree_params},
            cv=NonIterableCV(),
            optuna_settings=_basic_optuna_settings(),
        )


# ── cv variants (positive behavior) ───────────────────────────────────────────


@pytest.mark.ci
def test_scalar_tune_cv_variants(plr_model):
    """tune_ml_models() accepts integer and KFold splitter as cv."""
    param_space = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    settings = _basic_optuna_settings()

    # integer cv
    result = plr_model.tune_ml_models(param_space, cv=3, optuna_settings=settings, return_tune_res=True)
    assert "ml_l" in result

    # KFold splitter
    result = plr_model.tune_ml_models(
        param_space, cv=KFold(n_splits=3, shuffle=True, random_state=0), optuna_settings=settings, return_tune_res=True
    )
    assert "ml_l" in result
