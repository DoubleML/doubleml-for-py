"""Tests for IRM scalar hyperparameter tuning via tune_ml_models()."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)
from doubleml.utils._tune_optuna import DMLOptunaResult

# CV splitter matching tune_ml_models() default (cv=5)
_TUNE_CV = KFold(n_splits=5, shuffle=True, random_state=42)


@pytest.fixture(scope="module")
def irm_data():
    """IRM data fixture shared across all tests in this module."""
    np.random.seed(3142)
    return make_irm_data(n_obs=500, dim_x=5)


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def score(request):
    """Score function variants for IRM."""
    return request.param


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[c[0] for c in _SAMPLER_CASES])
def test_irm_scalar_tune_basic(irm_data, score, sampler_name, optuna_sampler):
    """tune_ml_models() returns DMLOptunaResult with valid tree params and applies them to learners."""
    ml_g = DecisionTreeRegressor(random_state=321)
    ml_m = DecisionTreeClassifier(random_state=654)

    model = IRM(irm_data, score=score)
    model.set_learners(ml_g=ml_g, ml_m=ml_m)

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g0": _small_tree_params, "ml_g1": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    # Return type and keys
    assert isinstance(tune_res, dict)
    assert set(tune_res.keys()) == {"ml_g0", "ml_g1", "ml_m"}

    # Each result is a DMLOptunaResult with valid tree params
    for key in ("ml_g0", "ml_g1"):
        assert isinstance(tune_res[key], DMLOptunaResult)
        assert tune_res[key].tuned is True
        _assert_tree_params(tune_res[key].best_params)

    assert isinstance(tune_res["ml_m"], DMLOptunaResult)
    assert tune_res["ml_m"].tuned is True
    _assert_tree_params(tune_res["ml_m"].best_params)

    # Best params are applied to the registered learner objects
    assert model.get_params("ml_g0")["max_depth"] == tune_res["ml_g0"].best_params["max_depth"]
    assert model.get_params("ml_g1")["max_depth"] == tune_res["ml_g1"].best_params["max_depth"]
    assert model.get_params("ml_m")["max_depth"] == tune_res["ml_m"].best_params["max_depth"]

    # Model fits successfully after tuning
    model.fit(n_folds=3)
    assert np.isfinite(model.coef).all()


@pytest.mark.ci
def test_irm_scalar_tune_improves_score(irm_data, score):
    """Tuning default (overfitting) trees improves cross-validated neg_rmse for ml_g0 and ml_g1."""
    x, y, d = irm_data.x, irm_data.y, irm_data.d

    ml_g = DecisionTreeRegressor(random_state=321)
    ml_m = DecisionTreeClassifier(random_state=654)

    # Baseline: default trees overfit on training folds → very negative neg_rmse
    mask0, mask1 = d == 0, d == 1
    baseline_g0 = cross_val_score(clone(ml_g), x[mask0], y[mask0], cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()
    baseline_g1 = cross_val_score(clone(ml_g), x[mask1], y[mask1], cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()

    model = IRM(irm_data, score=score)
    model.set_learners(ml_g=ml_g, ml_m=ml_m)

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g0": _small_tree_params, "ml_g1": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    # Optuna best_score should exceed baseline (less overfitting)
    assert tune_res["ml_g0"].best_score > baseline_g0
    assert tune_res["ml_g1"].best_score > baseline_g1


@pytest.mark.ci
def test_irm_scalar_tune_ml_g_alias(irm_data):
    """ml_g alias expands to both ml_g0 and ml_g1; result keys are the concrete learner names."""
    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    # Alias expands: result has ml_g0, ml_g1 (not ml_g)
    assert set(tune_res.keys()) == {"ml_g0", "ml_g1", "ml_m"}
    _assert_tree_params(tune_res["ml_g0"].best_params)
    _assert_tree_params(tune_res["ml_g1"].best_params)
    _assert_tree_params(tune_res["ml_m"].best_params)

    # Model fits after tuning
    model.fit(n_folds=3)
    assert np.isfinite(model.coef).all()


@pytest.mark.ci
def test_irm_scalar_tune_ml_g_alias_explicit_override(irm_data):
    """Explicit ml_g0 key overrides the ml_g alias; ml_g1 still gets the alias function."""

    def specific_g0_params(trial):
        """Restricts max_depth to 1-3 to distinguish from _small_tree_params (1-20)."""
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g": _small_tree_params, "ml_g0": specific_g0_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    assert set(tune_res.keys()) == {"ml_g0", "ml_g1", "ml_m"}
    # ml_g0 used specific_g0_params: max_depth constrained to [1, 3]
    assert tune_res["ml_g0"].best_params["max_depth"] <= 3
    # ml_g1 used _small_tree_params: all three keys present, max_depth up to 20
    _assert_tree_params(tune_res["ml_g1"].best_params)


@pytest.mark.ci
def test_irm_scalar_tune_returns_self(irm_data):
    """tune_ml_models() with return_tune_res=False returns self."""
    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))

    result = model.tune_ml_models(
        ml_param_space={"ml_g": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
    )

    assert result is model


@pytest.mark.ci
def test_irm_scalar_tune_set_as_params_false(irm_data):
    """tune_ml_models(set_as_params=False) finds best params but does not apply them to learners."""
    model = IRM(irm_data)
    model.set_learners(
        ml_g=DecisionTreeRegressor(max_depth=1, random_state=1),
        ml_m=DecisionTreeClassifier(max_depth=1, random_state=2),
    )

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
        set_as_params=False,
        return_tune_res=True,
    )

    # Learner params are unchanged
    assert model.get_params("ml_g0")["max_depth"] == 1
    assert model.get_params("ml_g1")["max_depth"] == 1
    assert model.get_params("ml_m")["max_depth"] == 1
    # But tune_res still has valid best params
    _assert_tree_params(tune_res["ml_g0"].best_params)
    _assert_tree_params(tune_res["ml_g1"].best_params)
    _assert_tree_params(tune_res["ml_m"].best_params)


@pytest.mark.ci
def test_irm_scalar_tune_invalid_key(irm_data):
    """_expand_tuning_param_space() raises ValueError for unknown keys."""
    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))

    with pytest.raises(ValueError, match="Invalid key 'ml_z' in ml_param_space"):
        model.tune_ml_models(
            ml_param_space={"ml_z": _small_tree_params},
            optuna_settings=_basic_optuna_settings(),
        )


@pytest.mark.ci
def test_irm_scalar_tune_partial_space(irm_data):
    """Tuning only a subset of learners leaves unspecified learners unchanged."""
    model = IRM(irm_data)
    model.set_learners(
        ml_g=DecisionTreeRegressor(max_depth=5, random_state=1),
        ml_m=DecisionTreeClassifier(max_depth=5, random_state=2),
    )

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g0": _small_tree_params},  # only ml_g0
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    # Only ml_g0 was tuned
    assert set(tune_res.keys()) == {"ml_g0"}
    _assert_tree_params(tune_res["ml_g0"].best_params)
    # ml_g1 and ml_m max_depth are unchanged
    assert model.get_params("ml_g1")["max_depth"] == 5
    assert model.get_params("ml_m")["max_depth"] == 5


@pytest.fixture(
    scope="module",
    params=["int", "kfold_splitter"],
    ids=["cv=int", "cv=KFold"],
)
def cv_variant(request):
    """Different cv argument types accepted by tune_ml_models(): int and splitter."""
    if request.param == "int":
        return 3
    return KFold(n_splits=3, shuffle=True, random_state=7)


@pytest.mark.ci
def test_irm_scalar_tune_cv_types(irm_data, cv_variant):
    """tune_ml_models() succeeds for supported cv argument types: int and splitter."""
    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_g": _small_tree_params, "ml_m": _small_tree_params},
        cv=cv_variant,
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    for name in ("ml_g0", "ml_g1", "ml_m"):
        assert name in tune_res
        assert tune_res[name].tuned is True
        assert isinstance(tune_res[name].best_params, dict)
        assert np.isfinite(tune_res[name].best_score)


@pytest.mark.ci
def test_irm_scalar_tune_cv_list_raises(irm_data):
    """tune_ml_models() raises TypeError when cv is a list of pre-made split pairs."""
    model = IRM(irm_data)
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeClassifier(random_state=2))
    cv_list = list(KFold(n_splits=3).split(np.arange(irm_data.n_obs)))

    msg = r"cv as a list of pre-made \(train_idx, test_idx\) pairs is not supported"
    with pytest.raises(TypeError, match=msg):
        model.tune_ml_models(
            ml_param_space={"ml_g": _small_tree_params, "ml_m": _small_tree_params},
            cv=cv_list,
            optuna_settings=_basic_optuna_settings(),
        )
