import pytest
from sklearn.tree import DecisionTreeRegressor

import doubleml as dml
from doubleml import DoubleMLData

try:  # pragma: no cover - optional dependency
    import optuna
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    optuna = None

pytestmark = pytest.mark.skipif(optuna is None, reason="Optuna is not installed.")


def _qmc_sampler():
    return getattr(optuna.samplers, "QMCSampler", None)


def _partial_fixed_sampler():
    return getattr(optuna.samplers, "PartialFixedSampler", None)


def _gp_sampler():
    return getattr(optuna.samplers, "GPSampler", None)


def _basic_optuna_settings(base_sampler, overrides=None):
    settings = {"n_trials": 2, "sampler": base_sampler}
    if overrides:
        settings.update(overrides)
    return settings


@pytest.mark.skipif(_qmc_sampler() is None, reason="QMCSampler not available in this Optuna version")
def test_doubleml_plr_qmc_sampler(generate_data1):
    data = generate_data1
    x_cols = [col for col in data.columns if col.startswith("X")]

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_data = DoubleMLData(data, "y", ["d"], x_cols)
    plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    sampler = _qmc_sampler()(seed=3141)
    def ml_l_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_l_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_l_min_samples_leaf", 1, 2),
        }

    def ml_m_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_m_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_m_min_samples_leaf", 1, 2),
        }

    tune_res = plr.tune_optuna(
        param_grids={"ml_l": ml_l_params, "ml_m": ml_m_params},
        optuna_settings=_basic_optuna_settings(sampler),
        return_tune_res=True,
    )

    tuned_params_l = plr.params["ml_l"]["d"][0][0]
    tuned_params_m = plr.params["ml_m"]["d"][0][0]

    assert set(tuned_params_l.keys()) == {"max_depth", "min_samples_leaf"}
    assert set(tuned_params_m.keys()) == {"max_depth", "min_samples_leaf"}
    assert "params" in tune_res[0]
    assert "tune_res" in tune_res[0]


@pytest.mark.skipif(_partial_fixed_sampler() is None, reason="PartialFixedSampler not available in this Optuna version")
def test_doubleml_plr_partial_fixed_sampler(generate_data1):
    data = generate_data1
    x_cols = [col for col in data.columns if col.startswith("X")]

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_data = DoubleMLData(data, "y", ["d"], x_cols)
    plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    base_sampler = optuna.samplers.RandomSampler(seed=3141)
    sampler_cls = _partial_fixed_sampler()
    sampler = sampler_cls(base_sampler=base_sampler, fixed_params={"max_depth": 2})

    def ml_l_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_l_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_l_min_samples_leaf", 1, 2),
        }

    def ml_m_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_m_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_m_min_samples_leaf", 1, 2),
        }

    tune_res = plr.tune_optuna(
        param_grids={"ml_l": ml_l_params, "ml_m": ml_m_params},
        optuna_settings=_basic_optuna_settings(sampler),
        return_tune_res=True,
    )

    tuned_params_l = plr.params["ml_l"]["d"][0][0]
    tuned_params_m = plr.params["ml_m"]["d"][0][0]

    assert tuned_params_l["max_depth"] == 2
    assert tuned_params_m["max_depth"] == 2
    assert "params" in tune_res[0]
    assert "tune_res" in tune_res[0]


@pytest.mark.skipif(_gp_sampler() is None, reason="GPSampler not available in this Optuna version")
def test_doubleml_plr_gp_sampler(generate_data1):
    data = generate_data1
    x_cols = [col for col in data.columns if col.startswith("X")]

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_data = DoubleMLData(data, "y", ["d"], x_cols)
    plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    sampler_cls = _gp_sampler()
    sampler = sampler_cls(seed=3141)

    def ml_l_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_l_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_l_min_samples_leaf", 1, 2),
        }

    def ml_m_params(trial):
        return {
            "max_depth": trial.suggest_int("ml_m_max_depth", 1, 2),
            "min_samples_leaf": trial.suggest_int("ml_m_min_samples_leaf", 1, 2),
        }

    plr.tune_optuna(
        param_grids={"ml_l": ml_l_params, "ml_m": ml_m_params},
        optuna_settings=_basic_optuna_settings(sampler),
    )

    tuned_params_l = plr.params["ml_l"]["d"][0][0]
    tuned_params_m = plr.params["ml_m"]["d"][0][0]

    assert tuned_params_l["max_depth"] in {1, 2}
    assert tuned_params_m["max_depth"] in {1, 2}
