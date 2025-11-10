import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDBinary, DoubleMLDIDCSBinary
from doubleml.did.datasets import (
    make_did_CS2021,
    make_did_cs_CS2021,
    make_did_SZ2020,
)
from doubleml.irm.datasets import (
    make_iivm_data,
    make_irm_data,
    make_ssm_data,
)
from doubleml.plm.datasets import (
    make_pliv_CHS2015,
    make_plr_CCDDHNR2018,
)

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


def _small_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
    }


def _medium_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
    }


def _assert_tree_params(param_dict, depth_range=(1, 2), leaf_range=(1, 3)):
    assert set(param_dict.keys()) == {"max_depth", "min_samples_leaf"}
    assert depth_range[0] <= param_dict["max_depth"] <= depth_range[1]
    assert leaf_range[0] <= param_dict["min_samples_leaf"] <= leaf_range[1]


def _first_params(dml_obj, learner):
    learner_params = dml_obj.params[learner]
    first_target = learner_params[next(iter(learner_params))]
    return first_target[0][0]


def _build_param_grid(dml_obj, param_fn):
    """Build parameter grid using the actual params_names from the DML object."""
    param_grid = {learner_name: param_fn for learner_name in dml_obj.params_names}
    return param_grid


def _select_binary_periods(panel_data):
    t_values = np.sort(panel_data.t_values)
    finite_g = sorted(val for val in panel_data.g_values if np.isfinite(val))
    for candidate in finite_g:
        pre_candidates = [t for t in t_values if t < candidate]
        if pre_candidates:
            return candidate, pre_candidates[-1], candidate
    raise RuntimeError("No valid treatment group found for binary DID data.")


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_plr_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=80, dim_x=6)

    ml_l = DecisionTreeRegressor(random_state=123, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=456, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    tune_res = dml_plr.tune_optuna(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler}),
        return_tune_res=True,
    )

    tuned_params_l = _first_params(dml_plr, "ml_l")
    tuned_params_m = _first_params(dml_plr, "ml_m")

    _assert_tree_params(tuned_params_l, depth_range=(1, 2))
    _assert_tree_params(tuned_params_m, depth_range=(1, 2))

    # ensure results contain optuna objects and best params
    assert "params" in tune_res[0]
    assert "tune_res" in tune_res[0]
    assert tune_res[0]["params"]["ml_l"]["max_depth"] == tuned_params_l["max_depth"]


def test_doubleml_optuna_cv_variants():
    np.random.seed(3142)
    dml_data = make_plr_CCDDHNR2018(n_obs=64, dim_x=5)

    ml_l_int = DecisionTreeRegressor(random_state=10, max_depth=5, min_samples_leaf=4)
    ml_m_int = DecisionTreeRegressor(random_state=11, max_depth=5, min_samples_leaf=4)
    dml_plr_int = dml.DoubleMLPLR(dml_data, ml_l_int, ml_m_int, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr_int.tune_optuna(
        ml_param_space=optuna_params,
        cv=3,
        optuna_settings=_basic_optuna_settings(),
    )

    int_l_params = dml_plr_int.get_params("ml_l")["d"][0][0]
    int_m_params = dml_plr_int.get_params("ml_m")["d"][0][0]

    assert int_l_params is not None
    assert int_m_params is not None

    ml_l_split = DecisionTreeRegressor(random_state=12, max_depth=5, min_samples_leaf=4)
    ml_m_split = DecisionTreeRegressor(random_state=13, max_depth=5, min_samples_leaf=4)
    dml_plr_split = dml.DoubleMLPLR(dml_data, ml_l_split, ml_m_split, n_folds=2, score="partialling out")

    cv_splitter = KFold(n_splits=3, shuffle=True, random_state=3142)

    dml_plr_split.tune_optuna(
        ml_param_space=optuna_params,
        cv=cv_splitter,
        optuna_settings=_basic_optuna_settings(),
    )

    split_l_params = dml_plr_split.get_params("ml_l")["d"][0][0]
    split_m_params = dml_plr_split.get_params("ml_m")["d"][0][0]

    assert split_l_params is not None
    assert split_m_params is not None


def test_doubleml_optuna_partial_tuning_single_learner():
    np.random.seed(3143)
    dml_data = make_plr_CCDDHNR2018(n_obs=64, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=20, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=21, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params}

    tune_res = dml_plr.tune_optuna(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    tuned_l = dml_plr.get_params("ml_l")["d"][0][0]
    untouched_m = dml_plr.get_params("ml_m")["d"][0]

    assert tuned_l is not None
    assert untouched_m is None

    assert set(tune_res[0]["params"].keys()) == {"ml_l"}
    assert "l_tune" in tune_res[0]["tune_res"]
    assert tune_res[0]["tune_res"]["l_tune"].tuned_ is True

    m_tune = tune_res[0]["tune_res"].get("m_tune")
    if m_tune is not None:
        assert not m_tune.tuned_


def test_doubleml_optuna_sets_params_for_all_folds():
    np.random.seed(3153)
    dml_data = make_plr_CCDDHNR2018(n_obs=60, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=101, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=202, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=3, n_rep=2)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr.tune_optuna(ml_param_space=optuna_params, optuna_settings=_basic_optuna_settings())

    l_params = dml_plr.get_params("ml_l")
    m_params = dml_plr.get_params("ml_m")

    assert set(l_params.keys()) == {"d"}
    assert set(m_params.keys()) == {"d"}

    expected_l = dict(l_params["d"][0][0])
    expected_m = dict(m_params["d"][0][0])

    assert len(l_params["d"]) == dml_plr.n_rep
    assert len(m_params["d"]) == dml_plr.n_rep

    for rep_idx in range(dml_plr.n_rep):
        assert len(l_params["d"][rep_idx]) == dml_plr.n_folds
        assert len(m_params["d"][rep_idx]) == dml_plr.n_folds
        for fold_idx in range(dml_plr.n_folds):
            l_fold_params = l_params["d"][rep_idx][fold_idx]
            m_fold_params = m_params["d"][rep_idx][fold_idx]
            assert l_fold_params is not None
            assert m_fold_params is not None
            assert l_fold_params == expected_l
            assert m_fold_params == expected_m


def test_doubleml_optuna_fit_uses_tuned_params():
    np.random.seed(3154)
    dml_data = make_plr_CCDDHNR2018(n_obs=60, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=303, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=404, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr.tune_optuna(ml_param_space=optuna_params, optuna_settings=_basic_optuna_settings())

    expected_l = dict(dml_plr.get_params("ml_l")["d"][0][0])
    expected_m = dict(dml_plr.get_params("ml_m")["d"][0][0])

    dml_plr.fit(store_predictions=False, store_models=True)

    for rep_idx in range(dml_plr.n_rep):
        for fold_idx in range(dml_plr.n_folds):
            ml_l_model = dml_plr.models["ml_l"]["d"][rep_idx][fold_idx]
            ml_m_model = dml_plr.models["ml_m"]["d"][rep_idx][fold_idx]
            assert ml_l_model is not None
            assert ml_m_model is not None
            for key, value in expected_l.items():
                assert ml_l_model.get_params()[key] == value
            for key, value in expected_m.items():
                assert ml_m_model.get_params()[key] == value


def test_doubleml_optuna_invalid_settings_key_raises():
    np.random.seed(3155)
    dml_data = make_irm_data(n_obs=80, dim_x=4)

    ml_g = DecisionTreeRegressor(random_state=111, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=222, max_depth=5, min_samples_leaf=4)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2)

    optuna_params = {"ml_g0": _medium_tree_params, "ml_g1": _medium_tree_params, "ml_m": _medium_tree_params}
    invalid_settings = _basic_optuna_settings({"ml_l": {"n_trials": 2}})

    with pytest.raises(ValueError, match="ml_l"):
        dml_irm.tune_optuna(ml_param_space=optuna_params, optuna_settings=invalid_settings)


def test_doubleml_optuna_class_name_setting_allowed():
    np.random.seed(3156)
    dml_data = make_plr_CCDDHNR2018(n_obs=60, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=515, max_depth=5, min_samples_leaf=3)
    ml_m = DecisionTreeRegressor(random_state=616, max_depth=5, min_samples_leaf=3)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    class_key = ml_l.__class__.__name__
    optuna_settings = _basic_optuna_settings({class_key: {"n_trials": 1}})

    dml_plr.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_irm_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3142)
    dml_data = make_irm_data(n_obs=120, dim_x=6)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2)

    optuna_params = {"ml_g0": _medium_tree_params, "ml_g1": _medium_tree_params, "ml_m": _medium_tree_params}

    per_ml_settings = {
        "ml_m": {"sampler": optuna_sampler, "n_trials": 1},
    }
    # vary g nuisance to ensure per-learner overrides still inherit base sampler
    if sampler_name != "random":
        per_ml_settings["ml_g0"] = {"sampler": optuna.samplers.RandomSampler(seed=7), "n_trials": 1}

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler, **per_ml_settings})

    dml_irm.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    tuned_params_g0 = _first_params(dml_irm, "ml_g0")
    tuned_params_g1 = _first_params(dml_irm, "ml_g1")
    tuned_params_m = _first_params(dml_irm, "ml_m")

    _assert_tree_params(tuned_params_g0, depth_range=(1, 3))
    _assert_tree_params(tuned_params_g1, depth_range=(1, 3))
    _assert_tree_params(tuned_params_m, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_iivm_optuna_tune(sampler_name, optuna_sampler):
    """Test IIVM with ml_g0, ml_g1, ml_m, ml_r0, ml_r1 nuisance models."""

    np.random.seed(3143)
    dml_data = make_iivm_data(n_obs=150, dim_x=6)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
    ml_r = DecisionTreeClassifier(random_state=789, max_depth=5, min_samples_leaf=4)

    dml_iivm = dml.DoubleMLIIVM(dml_data, ml_g, ml_m, ml_r, n_folds=2, subgroups={"always_takers": True, "never_takers": True})

    optuna_params = {
        "ml_g0": _medium_tree_params,
        "ml_g1": _medium_tree_params,
        "ml_m": _medium_tree_params,
        "ml_r0": _medium_tree_params,
        "ml_r1": _medium_tree_params,
    }

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_iivm.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    tuned_params_g0 = _first_params(dml_iivm, "ml_g0")
    tuned_params_g1 = _first_params(dml_iivm, "ml_g1")
    tuned_params_m = _first_params(dml_iivm, "ml_m")
    tuned_params_r0 = _first_params(dml_iivm, "ml_r0")
    tuned_params_r1 = _first_params(dml_iivm, "ml_r1")

    _assert_tree_params(tuned_params_g0, depth_range=(1, 3))
    _assert_tree_params(tuned_params_g1, depth_range=(1, 3))
    _assert_tree_params(tuned_params_m, depth_range=(1, 3))
    _assert_tree_params(tuned_params_r0, depth_range=(1, 3))
    _assert_tree_params(tuned_params_r1, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_pliv_optuna_tune(sampler_name, optuna_sampler):
    """Test PLIV with ml_l, ml_m, ml_r nuisance models."""

    np.random.seed(3144)
    dml_data = make_pliv_CHS2015(n_obs=120, dim_x=15, dim_z=3)

    ml_l = DecisionTreeRegressor(random_state=123, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=456, max_depth=5, min_samples_leaf=4)
    ml_r = DecisionTreeRegressor(random_state=789, max_depth=5, min_samples_leaf=4)

    dml_pliv = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=2)

    optuna_params = _build_param_grid(dml_pliv, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_pliv.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_pliv.params_names:
        tuned_params = _first_params(dml_pliv, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 2))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_cvar_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3145)
    dml_data = make_irm_data(n_obs=120, dim_x=6)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_cvar = dml.DoubleMLCVAR(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=2)

    optuna_params = {"ml_g": _medium_tree_params, "ml_m": _medium_tree_params}

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_cvar.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    tuned_params_g = _first_params(dml_cvar, "ml_g")
    tuned_params_m = _first_params(dml_cvar, "ml_m")

    _assert_tree_params(tuned_params_g, depth_range=(1, 3))
    _assert_tree_params(tuned_params_m, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_apo_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3146)
    dml_data = make_irm_data(n_obs=200, dim_x=6)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_apo = dml.DoubleMLAPO(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=2, treatment_level=1)

    optuna_params = _build_param_grid(dml_apo, _medium_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_apo.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_apo.params_names:
        tuned_params = _first_params(dml_apo, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_pq_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3147)
    dml_data = make_irm_data(n_obs=160, dim_x=6)

    ml_g = DecisionTreeClassifier(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_pq = dml.DoubleMLPQ(dml_data, ml_g, ml_m, n_folds=2)

    optuna_params = _build_param_grid(dml_pq, _medium_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_pq.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_pq.params_names:
        tuned_params = _first_params(dml_pq, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_lpq_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3148)
    dml_data = make_iivm_data(n_obs=180, dim_x=6)

    ml_g = DecisionTreeClassifier(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_lpq = dml.DoubleMLLPQ(dml_data, ml_g, ml_m, n_folds=2)

    optuna_params = _build_param_grid(dml_lpq, _medium_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_lpq.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_lpq.params_names:
        tuned_params = _first_params(dml_lpq, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_ssm_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3149)
    dml_data = make_ssm_data(n_obs=800, dim_x=12, mar=True)

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_pi = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=987, max_depth=5, min_samples_leaf=4)

    dml_ssm = dml.DoubleMLSSM(dml_data, ml_g, ml_pi, ml_m, n_folds=2, score="missing-at-random")

    optuna_params = _build_param_grid(dml_ssm, _medium_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_ssm.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_ssm.params_names:
        tuned_params = _first_params(dml_ssm, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 3))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_doubleml_did_optuna_tune(sampler_name, optuna_sampler, score):
    """Test DID with ml_g0, ml_g1 (and ml_m for observational score) nuisance models."""

    np.random.seed(3150)
    dml_data = make_did_SZ2020(n_obs=250, dgp_type=1, return_type="DoubleMLDIDData")

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    if score == "observational":
        ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
        dml_did = dml.DoubleMLDID(dml_data, ml_g, ml_m, score=score, n_folds=2)
    else:
        dml_did = dml.DoubleMLDID(dml_data, ml_g, score=score, n_folds=2)

    optuna_params = _build_param_grid(dml_did, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_did.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_did.params_names:
        tuned_params = _first_params(dml_did, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 2))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_doubleml_did_cs_optuna_tune(sampler_name, optuna_sampler, score):
    np.random.seed(3151)
    dml_data = make_did_SZ2020(
        n_obs=260,
        dgp_type=2,
        cross_sectional_data=True,
        return_type="DoubleMLDIDData",
    )

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    if score == "observational":
        ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)
        dml_did_cs = dml.DoubleMLDIDCS(dml_data, ml_g, ml_m, score=score, n_folds=2)
    else:
        dml_did_cs = dml.DoubleMLDIDCS(dml_data, ml_g, score=score, n_folds=2)

    optuna_params = _build_param_grid(dml_did_cs, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_did_cs.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_did_cs.params_names:
        tuned_params = _first_params(dml_did_cs, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 2))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_did_binary_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3152)
    df_panel = make_did_CS2021(
        n_obs=400,
        dgp_type=1,
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

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_did_binary = DoubleMLDIDBinary(
        obj_dml_data=panel_data,
        g_value=g_value,
        t_value_pre=t_value_pre,
        t_value_eval=t_value_eval,
        ml_g=ml_g,
        ml_m=ml_m,
        score="observational",
        n_folds=2,
    )

    optuna_params = _build_param_grid(dml_did_binary, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_did_binary.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_did_binary.params_names:
        tuned_params = _first_params(dml_did_binary, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 2))


@pytest.mark.parametrize("sampler_name,optuna_sampler", _SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def test_doubleml_did_cs_binary_optuna_tune(sampler_name, optuna_sampler):
    np.random.seed(3153)
    df_panel = make_did_cs_CS2021(
        n_obs=500,
        dgp_type=2,
        include_never_treated=True,
        lambda_t=0.6,
        time_type="float",
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

    ml_g = DecisionTreeRegressor(random_state=321, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=654, max_depth=5, min_samples_leaf=4)

    dml_did_cs_binary = DoubleMLDIDCSBinary(
        obj_dml_data=panel_data,
        g_value=g_value,
        t_value_pre=t_value_pre,
        t_value_eval=t_value_eval,
        ml_g=ml_g,
        ml_m=ml_m,
        score="observational",
        n_folds=2,
    )

    optuna_params = _build_param_grid(dml_did_cs_binary, _small_tree_params)

    optuna_settings = _basic_optuna_settings({"sampler": optuna_sampler})
    dml_did_cs_binary.tune_optuna(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    for learner_name in dml_did_cs_binary.params_names:
        tuned_params = _first_params(dml_did_cs_binary, learner_name)
        _assert_tree_params(tuned_params, depth_range=(1, 2))
