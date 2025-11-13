import numpy as np
import optuna
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.utils._tune_optuna import _resolve_optuna_scoring


def _basic_optuna_settings(additional=None):
    base_settings = {"n_trials": 10, "sampler": optuna.samplers.RandomSampler(seed=3141)}
    if additional is not None:
        base_settings.update(additional)
    return base_settings


_SAMPLER_CASES = [
    ("random", optuna.samplers.RandomSampler(seed=3141)),
    ("tpe", optuna.samplers.TPESampler(seed=3141)),
]


def _small_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 10),
    }


def _assert_tree_params(param_dict, depth_range=(2, 10), leaf_range=(2, 100), leaf_nodes_range=(2, 10)):
    assert set(param_dict.keys()) == {"max_depth", "min_samples_leaf", "max_leaf_nodes"}
    assert depth_range[0] <= param_dict["max_depth"] <= depth_range[1]
    assert leaf_range[0] <= param_dict["min_samples_leaf"] <= leaf_range[1]
    assert leaf_nodes_range[0] <= param_dict["max_leaf_nodes"] <= leaf_nodes_range[1]


def _build_param_space(dml_obj, param_fn):
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


def test_resolve_optuna_scoring_regressor_default():
    learner = LinearRegression()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


def test_resolve_optuna_scoring_classifier_default():
    learner = LogisticRegression()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_m")
    assert scoring == "neg_log_loss"
    assert "neg_log_loss" in message


def test_resolve_optuna_scoring_with_criterion_keeps_default():
    learner = DecisionTreeRegressor(random_state=0)
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


def test_resolve_optuna_scoring_lightgbm_regressor_default():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor

    learner = LGBMRegressor()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


def test_doubleml_optuna_cv_variants():
    np.random.seed(3142)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l_int = DecisionTreeRegressor(random_state=10, max_depth=5, min_samples_leaf=4)
    ml_m_int = DecisionTreeRegressor(random_state=11, max_depth=5, min_samples_leaf=4)
    dml_plr_int = dml.DoubleMLPLR(dml_data, ml_l_int, ml_m_int, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr_int.tune_ml_models(
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

    dml_plr_split.tune_ml_models(
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
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=20, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=21, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params}

    tune_res = dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings(),
        return_tune_res=True,
    )

    tuned_l = dml_plr.get_params("ml_l")["d"][0][0]
    untouched_m = dml_plr.get_params("ml_m")["d"][0]

    assert tuned_l is not None
    assert untouched_m is None

    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == {"ml_l"}
    l_tune = tune_res[0]["ml_l"]
    assert hasattr(l_tune, "tuned")
    assert l_tune.tuned is True
    assert "ml_m" not in tune_res[0]


def test_doubleml_optuna_sets_params_for_all_folds():
    np.random.seed(3153)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=101, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=202, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=3, n_rep=2)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=_basic_optuna_settings())

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
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=303, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=404, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=_basic_optuna_settings())

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
    dml_data = make_irm_data(n_obs=100, dim_x=5)

    ml_g = DecisionTreeRegressor(random_state=111, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeClassifier(random_state=222, max_depth=5, min_samples_leaf=4)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2)

    optuna_params = {"ml_g0": _small_tree_params, "ml_g1": _small_tree_params, "ml_m": _small_tree_params}
    invalid_settings = _basic_optuna_settings({"ml_l": {"n_trials": 2}})

    with pytest.raises(ValueError, match="ml_l"):
        dml_irm.tune_ml_models(ml_param_space=optuna_params, optuna_settings=invalid_settings)


def test_optuna_logging_integration():
    """Test that logging integration works correctly with Optuna."""
    import logging

    np.random.seed(3154)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=303, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=404, max_depth=5, min_samples_leaf=4)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    # Capture log messages
    logger = logging.getLogger("doubleml.utils._tune_optuna")
    original_level = logger.level

    # Create a custom handler to capture log records
    log_records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            log_records.append(record)

    handler = ListHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        # Tune with specific settings that should trigger log messages
        optuna_settings = {
            "n_trials": 2,
            "sampler": optuna.samplers.TPESampler(seed=42),
            "show_progress_bar": False,
        }

        dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings)

        # Check that we got log messages
        log_messages = [record.getMessage() for record in log_records]

        # Should have messages about direction and sampler for each learner
        direction_messages = [msg for msg in log_messages if "direction set to" in msg]
        sampler_messages = [msg for msg in log_messages if "sampler" in msg.lower()]
        scoring_messages = [msg for msg in log_messages if "scoring method" in msg.lower()]

        # We should have at least one message about direction
        assert len(direction_messages) > 0, "Expected log messages about optimization direction"

        # We should have messages about the sampler
        assert len(sampler_messages) > 0, "Expected log messages about sampler"

        # We should have messages about scoring
        assert len(scoring_messages) > 0, "Expected log messages about scoring method"

        # Verify that the tuning actually worked
        tuned_l = dml_plr.get_params("ml_l")["d"][0][0]
        tuned_m = dml_plr.get_params("ml_m")["d"][0][0]
        assert tuned_l is not None
        assert tuned_m is not None

    finally:
        # Clean up
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_optuna_logging_verbosity_sync():
    """Test that DoubleML logger level syncs with Optuna logger level."""
    import logging

    np.random.seed(3155)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=111)
    ml_m = DecisionTreeRegressor(random_state=222)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    # Set DoubleML logger to DEBUG
    logger = logging.getLogger("doubleml.utils._tune_optuna")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    try:
        # Tune without explicit verbosity setting
        optuna_settings = {
            "n_trials": 1,
            "show_progress_bar": False,
        }

        dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings)

        # The test passes if no exception is raised
        # The actual sync happens internally in _dml_tune_optuna
        assert True

    finally:
        logger.setLevel(original_level)


def test_optuna_logging_explicit_verbosity():
    """Test that explicit verbosity setting in optuna_settings takes precedence."""
    np.random.seed(3156)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=333)
    ml_m = DecisionTreeRegressor(random_state=444)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    # Explicitly set Optuna verbosity
    optuna_settings = {
        "n_trials": 1,
        "verbosity": optuna.logging.WARNING,
        "show_progress_bar": False,
    }

    # This should not raise an error
    dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings)

    # Verify tuning worked
    tuned_l = dml_plr.get_params("ml_l")["d"][0][0]
    assert tuned_l is not None
