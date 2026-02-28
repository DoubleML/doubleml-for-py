"""Tests for per-fold pruning support in DoubleMLScalar.tune_ml_models()."""

import numpy as np
import optuna
import pytest
from sklearn.tree import DecisionTreeRegressor

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR
from doubleml.tests._utils_tune_optuna import _small_tree_params

# ── Shared fixtures ────────────────────────────────────────────────────────────

np.random.seed(42)
_plr_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)


@pytest.fixture(scope="module")
def plr_model():
    """PLR scalar model for reuse across pruning tests."""
    model = PLR(_plr_data)
    model.set_learners(
        ml_l=DecisionTreeRegressor(random_state=1),
        ml_m=DecisionTreeRegressor(random_state=2),
    )
    return model


# ── Pruning tests ──────────────────────────────────────────────────────────────


@pytest.mark.ci
def test_scalar_tune_with_median_pruner(plr_model):
    """tune_ml_models() completes successfully when MedianPruner is passed via study_kwargs."""
    param_space = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    settings = {
        "n_trials": 8,
        "sampler": optuna.samplers.RandomSampler(seed=3141),
        "study_kwargs": {"pruner": optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)},
        "verbosity": optuna.logging.WARNING,
    }

    result = plr_model.tune_ml_models(param_space, cv=3, optuna_settings=settings, return_tune_res=True)

    for name in ("ml_l", "ml_m"):
        assert name in result
        assert result[name].tuned is True
        assert isinstance(result[name].best_params, dict)
        assert np.isfinite(result[name].best_score)
        # At least one complete trial must exist (RuntimeError raised otherwise)
        complete = [t for t in result[name].study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(complete) >= 1


@pytest.mark.ci
def test_scalar_tune_pruner_produces_pruned_trials(plr_model):
    """MedianPruner with n_startup_trials=1 produces at least one pruned trial over enough trials."""
    param_space = {"ml_l": _small_tree_params}
    settings = {
        "n_trials": 20,
        "sampler": optuna.samplers.RandomSampler(seed=99),
        "study_kwargs": {"pruner": optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)},
        "verbosity": optuna.logging.WARNING,
    }

    result = plr_model.tune_ml_models(param_space, cv=3, optuna_settings=settings, return_tune_res=True)

    study = result["ml_l"].study
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) >= 1, "Expected at least one pruned trial with MedianPruner(n_startup_trials=1) over 20 trials"


@pytest.mark.ci
def test_scalar_tune_all_trials_pruned_raises(plr_model):
    """tune_ml_models() raises RuntimeError when a pruner eliminates all trials."""

    class _AlwaysPruner(optuna.pruners.BasePruner):
        """Prune every trial unconditionally (even step 0)."""

        def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
            return True

    param_space = {"ml_l": _small_tree_params}
    settings = {
        "n_trials": 3,
        "study_kwargs": {"pruner": _AlwaysPruner()},
        "verbosity": optuna.logging.WARNING,
    }

    with pytest.raises(RuntimeError, match="Optuna optimization failed to produce any complete trials."):
        plr_model.tune_ml_models(param_space, cv=3, optuna_settings=settings)


@pytest.mark.ci
def test_scalar_tune_pruner_per_learner(plr_model):
    """Per-learner study_kwargs pruner applies only to that learner; the other learner is unaffected."""
    param_space = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    settings = {
        "n_trials": 20,
        "sampler": optuna.samplers.RandomSampler(seed=3141),
        "verbosity": optuna.logging.WARNING,
        # ml_l: aggressive pruner → expect pruned trials
        "ml_l": {
            "study_kwargs": {"pruner": optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)},
        },
        # ml_m: explicitly disable pruning → zero pruned trials
        "ml_m": {
            "study_kwargs": {"pruner": optuna.pruners.NopPruner()},
        },
    }

    result = plr_model.tune_ml_models(param_space, cv=3, optuna_settings=settings, return_tune_res=True)

    # ml_l: expect at least one pruned trial due to the per-learner MedianPruner
    ml_l_pruned = [t for t in result["ml_l"].study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(ml_l_pruned) >= 1, "Expected ml_l to have pruned trials with a per-learner MedianPruner"

    # ml_m: NoPruner → all 20 trials should be complete
    ml_m_pruned = [t for t in result["ml_m"].study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(ml_m_pruned) == 0, "Expected ml_m to have no pruned trials since NoPruner was configured"
