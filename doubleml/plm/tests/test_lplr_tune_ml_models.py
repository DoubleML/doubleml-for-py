"""Validate LPLR tuning against its cross-validation objective and applied parameters."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.plm.datasets import make_lplr_LZZ2020
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[DecisionTreeRegressor(random_state=567), None])
def ml_a(request):
    return request.param


@pytest.mark.ci
@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_lplr_optuna_tune(sampler_name, optuna_sampler, score, ml_a):
    """Tuning improves its own CV objective and applies the selected parameters."""
    np.random.seed(3141)
    dml_data = make_lplr_LZZ2020(n_obs=500, dim_x=15, alpha=0.5)
    dml_lplr = dml.DoubleMLLPLR(
        dml_data,
        ml_M=DecisionTreeClassifier(random_state=123),
        ml_t=DecisionTreeRegressor(random_state=234),
        ml_m=DecisionTreeRegressor(random_state=456),
        ml_a=ml_a,
        n_folds=3,
        n_folds_inner=2,
        score=score,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    x, y, d = dml_data.x, dml_data.y, dml_data.d
    mask_m = y == 0 if score == "nuisance_space" else np.ones(len(y), dtype=bool)
    tuning_data = {
        "ml_M": (np.column_stack((d, x)), y, "neg_log_loss"),
        "ml_m": (x[mask_m], d[mask_m], "neg_root_mean_squared_error"),
        "ml_a": (x, d, "neg_root_mean_squared_error"),
    }
    baseline_scores = {
        name: cross_val_score(clone(dml_lplr.learner[name]), features, target, cv=cv, scoring=scorer).mean()
        for name, (features, target, scorer) in tuning_data.items()
    }
    optuna_params = {name: _small_tree_params for name in ["ml_M", "ml_m", "ml_a", "ml_t"]}
    tune_res = dml_lplr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=cv,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 5}),
        return_tune_res=True,
    )

    assert len(tune_res) == 1
    assert set(tune_res[0]) == set(optuna_params)
    for name, result in tune_res[0].items():
        assert result.tuned
        _assert_tree_params(result.best_params)
        assert dml_lplr.get_params(name)["d"] == [[result.best_params] * dml_lplr.n_folds]
        for key, value in result.best_params.items():
            assert result.best_estimator.get_params()[key] == value

    for name, (features, target, scorer) in tuning_data.items():
        result = tune_res[0][name]
        verified_score = cross_val_score(clone(result.best_estimator), features, target, cv=cv, scoring=scorer).mean()
        assert result.scoring_method == scorer
        assert result.best_score == pytest.approx(verified_score)
        assert verified_score > baseline_scores[name]

    dml_lplr.fit()
    assert np.isfinite(dml_lplr.coef).all()
    assert np.isfinite(dml_lplr.se).all()

    expected_a = np.empty_like(d)
    for (_, test), inner_splits in zip(dml_lplr.smpls[0], dml_lplr.smpls_inner[0]):
        predictions = [
            clone(tune_res[0]["ml_a"].best_estimator).fit(x[train], d[train]).predict(x[test]) for train, _ in inner_splits
        ]
        expected_a[test] = np.mean(predictions, axis=0)
    np.testing.assert_allclose(dml_lplr.predictions["ml_a"][:, 0, 0], expected_a)
