import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml import DoubleMLData

try:  # pragma: no cover - optional dependency
    import optuna
    from optuna.samplers import TPESampler
    try:
        from optuna.integration import SkoptSampler
    except Exception:  # pragma: no cover - optional dependency
        SkoptSampler = None
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    optuna = None
    TPESampler = None
    SkoptSampler = None

pytestmark = pytest.mark.skipif(optuna is None, reason="Optuna is not installed.")


def _basic_optuna_settings(additional=None):
    base_settings = {"n_trials": 1, "sampler": optuna.samplers.RandomSampler(seed=3141)}
    if additional is not None:
        base_settings.update(additional)
    return base_settings


_SAMPLER_CASES = [
    ("random", optuna.samplers.RandomSampler(seed=3141)),
]

if TPESampler is not None:  # pragma: no cover - optional dependency
    _SAMPLER_CASES.append(("tpe", TPESampler(seed=3141)))

if SkoptSampler is not None:  # pragma: no cover - optional dependency
    _SAMPLER_CASES.append(("skopt", SkoptSampler(seed=3141)))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_plr_optuna_tune(generate_data1, sampler_name, optuna_sampler):
    data = generate_data1
    x_cols = [col for col in data.columns if col.startswith("X")]

    ml_l = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeRegressor(random_state=456)

    dml_data = DoubleMLData(data, "y", ["d"], x_cols)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    param_grids = {
        "ml_l": {"max_depth": [1, 2], "min_samples_leaf": [1, 2]},
        "ml_m": {"max_depth": [1, 2], "min_samples_leaf": [1, 2]},
    }

    tune_res = dml_plr.tune(
        param_grids=param_grids,
        search_mode="optuna",
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    tuned_params_l = dml_plr.params["ml_l"]["d"][0][0]
    tuned_params_m = dml_plr.params["ml_m"]["d"][0][0]

    assert set(tuned_params_l.keys()) == {"max_depth", "min_samples_leaf"}
    assert set(tuned_params_m.keys()) == {"max_depth", "min_samples_leaf"}
    assert tuned_params_l["max_depth"] in {1, 2}
    assert tuned_params_m["max_depth"] in {1, 2}

    # ensure results contain optuna objects and best params
    assert "params" in tune_res[0]
    assert "tune_res" in tune_res[0]
    assert tune_res[0]["params"]["ml_l"][0]["max_depth"] == tuned_params_l["max_depth"]


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_irm_optuna_tune(sampler_name, optuna_sampler):
    rng = np.random.default_rng(42)
    n_obs = 60
    x = rng.normal(size=(n_obs, 3))
    p_d = 1 / (1 + np.exp(-(x[:, 0] - 0.5 * x[:, 1])))
    d = rng.binomial(1, p_d)
    y = 0.8 * d + x[:, 1] - 0.25 * x[:, 2] + rng.normal(scale=0.1, size=n_obs)

    df = pd.DataFrame(np.column_stack((y, d, x)), columns=["y", "d", "X1", "X2", "X3"])
    dml_data = DoubleMLData(df, "y", ["d"], ["X1", "X2", "X3"])

    ml_g = DecisionTreeRegressor(random_state=321)
    ml_m = DecisionTreeClassifier(random_state=654)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2)

    param_grids = {
        "ml_g": {"max_depth": [1, 2], "min_samples_leaf": [1, 3]},
        "ml_m": {"max_depth": [1, 2], "min_samples_leaf": [1, 3]},
    }

    per_ml_settings = {
        "ml_m": {"sampler": optuna_sampler, "n_trials": 1},
    }
    # vary g nuisance to ensure per-learner overrides still inherit base sampler
    if sampler_name != "random":
        per_ml_settings["ml_g0"] = {"sampler": optuna.samplers.RandomSampler(seed=7), "n_trials": 1}

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler, **per_ml_settings})

    dml_irm.tune(param_grids=param_grids, search_mode="optuna", optuna_settings=optuna_settings)

    tuned_params_g0 = dml_irm.params["ml_g0"]["d"][0][0]
    tuned_params_g1 = dml_irm.params["ml_g1"]["d"][0][0]
    tuned_params_m = dml_irm.params["ml_m"]["d"][0][0]

    assert tuned_params_g0["max_depth"] in {1, 2}
    assert tuned_params_g1["max_depth"] in {1, 2}
    assert tuned_params_m["max_depth"] in {1, 2}
    assert set(tuned_params_g0.keys()) == {"max_depth", "min_samples_leaf"}
    assert set(tuned_params_g1.keys()) == {"max_depth", "min_samples_leaf"}
    assert set(tuned_params_m.keys()) == {"max_depth", "min_samples_leaf"}
