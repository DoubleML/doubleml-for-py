"""Tests for PLR scalar hyperparameter tuning via tune_ml_models()."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR
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
def plr_data():
    """PLR data fixture shared across all tests in this module."""
    np.random.seed(3141)
    return make_plr_CCDDHNR2018(n_obs=500, dim_x=5, alpha=0.5)


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    """Score function variants for PLR."""
    return request.param


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[c[0] for c in _SAMPLER_CASES])
def test_plr_scalar_tune_basic(plr_data, score, sampler_name, optuna_sampler):
    """tune_ml_models() returns DMLOptunaResult with valid tree params and applies them to learners."""
    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    model = PLR(plr_data, score=score)
    model.set_learners(ml_l=ml_l, ml_m=ml_m)
    if score == "IV-type":
        model.set_learners(ml_g=DecisionTreeRegressor(random_state=789))

    param_space = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    if score == "IV-type":
        param_space["ml_g"] = _small_tree_params

    tune_res = model.tune_ml_models(
        ml_param_space=param_space,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    # Return type and keys
    assert isinstance(tune_res, dict)
    expected_keys = {"ml_l", "ml_m"}
    if score == "IV-type":
        expected_keys.add("ml_g")
    assert set(tune_res.keys()) == expected_keys

    # Each result is a DMLOptunaResult with valid tree params
    for key in tune_res:
        assert isinstance(tune_res[key], DMLOptunaResult)
        assert tune_res[key].tuned is True
        _assert_tree_params(tune_res[key].best_params)

    # Best params are applied to the registered learner objects
    assert model.get_params("ml_l")["max_depth"] == tune_res["ml_l"].best_params["max_depth"]
    assert model.get_params("ml_m")["max_depth"] == tune_res["ml_m"].best_params["max_depth"]
    if score == "IV-type":
        assert model.get_params("ml_g")["max_depth"] == tune_res["ml_g"].best_params["max_depth"]

    # Model fits successfully after tuning
    model.fit(n_folds=3)
    assert np.isfinite(model.coef).all()


@pytest.mark.ci
def test_plr_scalar_tune_improves_score(plr_data, score):
    """Tuning a default (overfitting) tree improves cross-validated neg_rmse."""
    x, y, d = plr_data.x, plr_data.y, plr_data.d

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    # Baseline: default trees overfit on training folds → high test RMSE → very negative neg_rmse
    baseline_l = cross_val_score(clone(ml_l), x, y, cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()
    baseline_m = cross_val_score(clone(ml_m), x, d, cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()

    model = PLR(plr_data, score=score)
    model.set_learners(ml_l=ml_l, ml_m=ml_m)
    if score == "IV-type":
        model.set_learners(ml_g=DecisionTreeRegressor(random_state=789))

    param_space = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    if score == "IV-type":
        param_space["ml_g"] = _small_tree_params

    tune_res = model.tune_ml_models(
        ml_param_space=param_space,
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    # Optuna best_score (neg_root_mean_squared_error) should exceed baseline (less overfitting)
    assert tune_res["ml_l"].best_score > baseline_l
    assert tune_res["ml_m"].best_score > baseline_m

    if score == "IV-type":
        # Replicate _get_tuning_data's 2-stage target for ml_g: y - theta_initial * d.
        # Uses _TUNE_CV which matches resolve_optuna_cv(cv=5) used internally.
        ml_g = DecisionTreeRegressor(random_state=789)
        l_hat = cross_val_predict(clone(ml_l), x, y, cv=_TUNE_CV)
        m_hat = cross_val_predict(clone(ml_m), x, d, cv=_TUNE_CV)
        psi_a = -((d - m_hat) ** 2)
        psi_b = (d - m_hat) * (y - l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
        y_g = y - theta_initial * d
        baseline_g = cross_val_score(clone(ml_g), x, y_g, cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()
        assert tune_res["ml_g"].best_score > baseline_g


@pytest.mark.ci
def test_plr_scalar_tune_returns_self(plr_data):
    """tune_ml_models() with return_tune_res=False returns self."""
    model = PLR(plr_data)
    model.set_learners(ml_l=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeRegressor(random_state=2))

    result = model.tune_ml_models(
        ml_param_space={"ml_l": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
    )

    assert result is model


@pytest.mark.ci
def test_plr_scalar_tune_set_as_params_false(plr_data):
    """tune_ml_models(set_as_params=False) finds best params but does not apply them to learners."""
    model = PLR(plr_data)
    model.set_learners(
        ml_l=DecisionTreeRegressor(max_depth=1, random_state=1),
        ml_m=DecisionTreeRegressor(max_depth=1, random_state=2),
    )

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_l": _small_tree_params, "ml_m": _small_tree_params},
        optuna_settings=_basic_optuna_settings(),
        set_as_params=False,
        return_tune_res=True,
    )

    # Learner params are unchanged
    assert model.get_params("ml_l")["max_depth"] == 1
    assert model.get_params("ml_m")["max_depth"] == 1
    # But tune_res still has valid best params
    _assert_tree_params(tune_res["ml_l"].best_params)
    _assert_tree_params(tune_res["ml_m"].best_params)


@pytest.mark.ci
def test_plr_scalar_tune_invalid_key(plr_data):
    """_expand_tuning_param_space() raises ValueError for unknown keys."""
    model = PLR(plr_data)
    model.set_learners(ml_l=DecisionTreeRegressor(), ml_m=DecisionTreeRegressor())

    with pytest.raises(ValueError, match="Invalid key 'ml_z' in ml_param_space"):
        model.tune_ml_models(
            ml_param_space={"ml_z": _small_tree_params},
            optuna_settings=_basic_optuna_settings(),
        )


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
def test_plr_scalar_tune_cv_types(plr_data, cv_variant):
    """tune_ml_models() succeeds for supported cv argument types: int and splitter."""
    model = PLR(plr_data)
    model.set_learners(ml_l=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeRegressor(random_state=2))

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_l": _small_tree_params, "ml_m": _small_tree_params},
        cv=cv_variant,
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    for name in ("ml_l", "ml_m"):
        assert name in tune_res
        assert tune_res[name].tuned is True
        assert isinstance(tune_res[name].best_params, dict)
        assert np.isfinite(tune_res[name].best_score)


@pytest.mark.ci
def test_plr_scalar_tune_cv_list_raises(plr_data):
    """tune_ml_models() raises TypeError when cv is a list of pre-made split pairs."""
    model = PLR(plr_data)
    model.set_learners(ml_l=DecisionTreeRegressor(random_state=1), ml_m=DecisionTreeRegressor(random_state=2))
    cv_list = list(KFold(n_splits=3).split(np.arange(plr_data.n_obs)))

    msg = r"cv as a list of pre-made \(train_idx, test_idx\) pairs is not supported"
    with pytest.raises(TypeError, match=msg):
        model.tune_ml_models(
            ml_param_space={"ml_l": _small_tree_params, "ml_m": _small_tree_params},
            cv=cv_list,
            optuna_settings=_basic_optuna_settings(),
        )


@pytest.mark.ci
def test_plr_scalar_tune_partial_space(plr_data):
    """Tuning only a subset of learners leaves unspecified learners unchanged."""
    model = PLR(plr_data)
    model.set_learners(
        ml_l=DecisionTreeRegressor(max_depth=5, random_state=1),
        ml_m=DecisionTreeRegressor(max_depth=5, random_state=2),
    )

    tune_res = model.tune_ml_models(
        ml_param_space={"ml_l": _small_tree_params},  # only ml_l
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    # Only ml_l was tuned
    assert set(tune_res.keys()) == {"ml_l"}
    _assert_tree_params(tune_res["ml_l"].best_params)
    # ml_m max_depth is unchanged
    assert model.get_params("ml_m")["max_depth"] == 5


@pytest.mark.ci
def test_plr_scalar_tune_ml_g_missing_ml_l_ml_m(plr_data):
    """Tuning ml_g without ml_l and ml_m registered raises ValueError."""
    model = PLR(plr_data, score="IV-type")
    model.set_learners(ml_g=DecisionTreeRegressor(random_state=1))

    msg = r"Tuning 'ml_g' requires 'ml_l' and 'ml_m' to be registered\."
    with pytest.raises(ValueError, match=msg):
        model.tune_ml_models(
            ml_param_space={"ml_g": _small_tree_params},
            optuna_settings=_basic_optuna_settings(),
        )
