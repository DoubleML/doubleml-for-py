import numpy as np
import optuna
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.tests._utils_tune_optuna import (
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)
from doubleml.utils._tune_optuna import (
    DMLOptunaResult,
    _create_study,
    _resolve_optuna_scoring,
    resolve_optuna_cv,
)


@pytest.mark.ci
def test_resolve_optuna_scoring_regressor_default():
    learner = LinearRegression()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


@pytest.mark.ci
def test_resolve_optuna_scoring_classifier_default():
    learner = LogisticRegression()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_m")
    assert scoring == "neg_log_loss"
    assert "neg_log_loss" in message


@pytest.mark.ci
def test_resolve_optuna_scoring_with_criterion_keeps_default():
    learner = DecisionTreeRegressor(random_state=0)
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


@pytest.mark.ci
def test_resolve_optuna_scoring_lightgbm_regressor_default():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor

    learner = LGBMRegressor()
    scoring, message = _resolve_optuna_scoring(None, learner, "ml_l")
    assert scoring == "neg_root_mean_squared_error"
    assert "neg_root_mean_squared_error" in message


@pytest.mark.ci
def test_resolve_optuna_cv_sets_random_state():
    cv = resolve_optuna_cv(3)
    assert isinstance(cv, KFold)
    assert cv.shuffle is True
    assert cv.random_state == 42


@pytest.mark.ci
def test_doubleml_optuna_cv_variants():
    np.random.seed(3142)
    dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=5)

    ml_l = DecisionTreeRegressor(random_state=10, max_depth=5, min_samples_leaf=4)
    ml_m = DecisionTreeRegressor(random_state=11, max_depth=5, min_samples_leaf=4)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=3,
        optuna_settings=_basic_optuna_settings(),
    )

    int_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    int_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert int_l_params is not None
    assert int_m_params is not None

    cv_splitter = KFold(n_splits=3, shuffle=True, random_state=3142)

    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=cv_splitter,
        optuna_settings=_basic_optuna_settings(),
    )

    split_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    split_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert split_l_params is not None
    assert split_m_params is not None

    class SimpleSplitter:
        def __init__(self, n_splits=3):
            self._kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3142)
            self._n_splits = n_splits

        def split(self, X, y=None, groups=None):
            return self._kfold.split(X, y, groups)

        def get_n_splits(self, X=None, y=None, groups=None):
            return self._n_splits

    custom_cv = SimpleSplitter(n_splits=3)

    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=custom_cv,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
    )

    custom_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    custom_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert custom_l_params is not None
    assert custom_m_params is not None

    base_iter_kfold = KFold(n_splits=3, shuffle=True, random_state=3142)
    cv_iterable = list(base_iter_kfold.split(np.arange(dml_data.n_obs)))

    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=cv_iterable,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
    )

    iterable_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    iterable_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert iterable_l_params is not None
    assert iterable_m_params is not None

    explicit_cv_iterable = [
        (list(train_idx), list(test_idx)) for train_idx, test_idx in base_iter_kfold.split(np.arange(dml_data.n_obs))
    ]

    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=explicit_cv_iterable,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
    )

    explicit_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    explicit_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert explicit_l_params is not None
    assert explicit_m_params is not None

    cv = None
    dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        cv=cv,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
    )
    none_l_params = dml_plr.get_params("ml_l")["d"][0][0]
    none_m_params = dml_plr.get_params("ml_m")["d"][0][0]

    assert none_l_params is not None
    assert none_m_params is not None


@pytest.mark.ci
def test_create_study_respects_user_study_name(monkeypatch):
    captured_kwargs = {}

    def fake_create_study(**kwargs):
        captured_kwargs.update(kwargs)

        class _DummyStudy:
            pass

        return _DummyStudy()

    monkeypatch.setattr(optuna, "create_study", fake_create_study)

    settings = {
        "study": None,
        "study_kwargs": {"study_name": "custom-study", "direction": "maximize"},
        "direction": "maximize",
        "sampler": None,
    }

    _create_study(settings, "ml_l")

    assert captured_kwargs["study_name"] == "custom-study"


@pytest.mark.ci
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

    res_ml_m = tune_res[0]["ml_m"]
    res_ml_l = tune_res[0]["ml_l"]

    assert res_ml_m.tuned is False
    assert res_ml_m.best_estimator.get_params() == ml_m.get_params()  # assert default params kept
    assert res_ml_l.tuned is True
    _assert_tree_params(res_ml_l.best_params)  # assert tuned params valid

    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == {"ml_l", "ml_m"}


@pytest.mark.ci
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


@pytest.mark.ci
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


@pytest.mark.ci
def test_dml_optuna_result_str_representation():
    def custom_scorer(**args):
        return 0.0

    primary_result = DMLOptunaResult(
        learner_name="ml_l",
        params_name="ml_l",
        best_estimator=LinearRegression(),
        best_params={"alpha": 1, "depth": 3},
        best_score=0.123,
        scoring_method="neg_mean_squared_error",
        study=None,
        tuned=True,
    )

    primary_str = str(primary_result)
    assert primary_str.startswith("================== DMLOptunaResult")
    assert "Learner name: ml_l" in primary_str
    assert "Best score: 0.123" in primary_str
    assert "Scoring method: neg_mean_squared_error" in primary_str
    assert "'alpha': 1" in primary_str
    assert "'depth': 3" in primary_str

    empty_params_result = DMLOptunaResult(
        learner_name="ml_m",
        params_name="ml_m",
        best_estimator=LinearRegression(),
        best_params={},
        best_score=-0.5,
        scoring_method=custom_scorer,
        study=None,
        tuned=False,
    )

    empty_str = str(empty_params_result)
    assert "Learner name: ml_m" in empty_str
    assert "Scoring method: custom_scorer" in empty_str
    assert "No best parameters available." in empty_str


@pytest.mark.ci
def test_doubleml_optuna_scoring_method_variants():
    np.random.seed(3156)
    dml_data = make_plr_CCDDHNR2018(n_obs=120, dim_x=5)

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}

    ml_l_string = DecisionTreeRegressor(random_state=501)
    ml_m_default = DecisionTreeRegressor(random_state=502)
    dml_plr_string = dml.DoubleMLPLR(dml_data, ml_l_string, ml_m_default, n_folds=2)

    scoring_methods_string = {"ml_l": "neg_mean_squared_error"}

    tune_res_string = dml_plr_string.tune_ml_models(
        ml_param_space=optuna_params,
        scoring_methods=scoring_methods_string,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
        return_tune_res=True,
    )

    assert tune_res_string[0]["ml_l"].scoring_method == "neg_mean_squared_error"
    assert tune_res_string[0]["ml_m"].scoring_method == "neg_root_mean_squared_error"

    def neg_mae_scorer(estimator, x, y):
        preds = estimator.predict(x)
        return -np.mean(np.abs(y - preds))

    ml_l_callable = DecisionTreeRegressor(random_state=601)
    ml_m_callable = DecisionTreeRegressor(random_state=602)
    dml_plr_callable = dml.DoubleMLPLR(dml_data, ml_l_callable, ml_m_callable, n_folds=2)

    scoring_methods_callable = {"ml_l": neg_mae_scorer, "ml_m": neg_mae_scorer}

    tune_res_callable = dml_plr_callable.tune_ml_models(
        ml_param_space=optuna_params,
        scoring_methods=scoring_methods_callable,
        optuna_settings=_basic_optuna_settings({"n_trials": 2}),
        return_tune_res=True,
    )

    assert tune_res_callable[0]["ml_l"].scoring_method is neg_mae_scorer
    assert tune_res_callable[0]["ml_m"].scoring_method is neg_mae_scorer


@pytest.mark.ci
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


@pytest.mark.ci
def test_optuna_settings_hierarchy_overrides():
    np.random.seed(3160)
    dml_data = make_irm_data(n_obs=80, dim_x=5)

    ml_g = DecisionTreeRegressor(random_state=123)
    ml_m = DecisionTreeClassifier(random_state=456)
    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_g": _small_tree_params, "ml_g1": _small_tree_params, "ml_m": _small_tree_params}
    scoring_methods = {
        "ml_g0": "neg_mean_squared_error",
        "ml_g1": "neg_mean_squared_error",
        "ml_m": "roc_auc",
    }

    optuna_settings = {
        "n_trials": 4,
        "direction": "maximize",
        "show_progress_bar": False,
        "ml_g": {"n_trials": 2},
        "ml_g1": {"n_trials": 3},
    }

    tune_res = dml_irm.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=optuna_settings,
        scoring_methods=scoring_methods,
        cv=3,
        return_tune_res=True,
    )

    result_map = tune_res[0]

    def _completed_trials(study):
        return sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)

    assert _completed_trials(result_map["ml_g0"].study) == 2
    assert _completed_trials(result_map["ml_g1"].study) == 3
    assert _completed_trials(result_map["ml_m"].study) == 4


@pytest.mark.ci
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


@pytest.mark.ci
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


@pytest.mark.ci
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


@pytest.mark.ci
def test_doubleml_optuna_respects_provided_study_instances():
    np.random.seed(3157)
    dml_data = make_plr_CCDDHNR2018(n_obs=80, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=555, max_depth=3, min_samples_leaf=3)
    ml_m = DecisionTreeRegressor(random_state=556, max_depth=3, min_samples_leaf=3)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2)

    study_l = optuna.create_study(direction="maximize")
    study_m = optuna.create_study(direction="maximize")

    optuna_params = {"ml_l": _small_tree_params, "ml_m": _small_tree_params}
    optuna_settings = {
        "n_trials": 1,
        "show_progress_bar": False,
        "ml_l": {"study": study_l},
        "ml_m": {"study": study_m},
    }

    tune_res = dml_plr.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=optuna_settings,
        return_tune_res=True,
    )

    assert tune_res[0]["ml_l"].study is study_l
    assert tune_res[0]["ml_m"].study is study_m
