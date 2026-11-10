"""Tests for the check whether a learner supports sample weights."""

import re

import pytest
from sklearn import config_context
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from doubleml.utils import GlobalClassifier, GlobalRegressor
from doubleml.utils._checks import _check_supports_sample_weights

reg_estimators = [
    ("lr local", LinearRegression()),
    ("rf local", RandomForestRegressor(n_estimators=10, random_state=42)),
    ("lr global", GlobalRegressor(base_estimator=LinearRegression())),
    ("rf global", GlobalRegressor(base_estimator=RandomForestRegressor(n_estimators=10, random_state=42))),
]

class_estimators = [
    ("lr local", LogisticRegression(random_state=42)),
    ("rf local", RandomForestClassifier(n_estimators=10, random_state=42)),
    ("lr global", GlobalClassifier(base_estimator=LogisticRegression(random_state=42))),
    ("rf global", GlobalClassifier(base_estimator=RandomForestClassifier(n_estimators=10, random_state=42))),
]

ml_g = StackingRegressor(
    estimators=reg_estimators,
    final_estimator=LinearRegression(),
)

ml_m = StackingClassifier(
    estimators=class_estimators,
    final_estimator=LogisticRegression(random_state=42),
)


@pytest.mark.ci
@pytest.mark.parametrize(
    "learner, learner_name",
    [(learner, "ml_g") for _, learner in reg_estimators]
    + [(learner, "ml_m") for _, learner in class_estimators]
    + [(ml_g, "ml_g"), (ml_m, "ml_m")],
)
def test_check_supports_sample_weights_valid(learner, learner_name):
    # explicit sample_weight parameter (base learners + Global wrappers)
    # or **fit_params catch-all (StackingRegressor/StackingClassifier)
    _check_supports_sample_weights(learner, learner_name)


@pytest.mark.ci
@pytest.mark.parametrize(
    "learner, learner_name",
    [
        (KNeighborsRegressor(), "ml_g"),
        (KNeighborsClassifier(), "ml_m"),
    ],
)
def test_check_supports_sample_weights_invalid(learner, learner_name):
    msg = re.escape(
        f"The {learner_name} learner {str(learner)} does not support sample weights. "
        f"{type(learner).__name__}.fit() has no sample_weight argument. "
        "Please choose a learner that supports sample weights."
    )
    with pytest.raises(ValueError, match=msg):
        _check_supports_sample_weights(learner, learner_name)


def _pipe(request_weights=False):
    """Scaler + Lasso pipeline; optionally request sample weights on both steps (needs metadata routing)."""
    pipe = make_pipeline(StandardScaler(), Lasso())
    if request_weights:
        for _, step in pipe.steps:
            step.set_fit_request(sample_weight=True)
    return pipe


@pytest.mark.ci
@pytest.mark.parametrize(
    "learner",
    [
        GridSearchCV(Lasso(), {"alpha": [0.1, 1.0]}),
        BaggingRegressor(_pipe()),
        TransformedTargetRegressor(regressor=Lasso(), transformer=FunctionTransformer()),
        GlobalRegressor(base_estimator=KNeighborsRegressor()),
    ],
)
def test_check_supports_sample_weights_valid_meta_estimators(learner):
    """Meta-estimators that handle the weights themselves or forward them to learners that accept them pass."""
    _check_supports_sample_weights(learner, "ml_g")


@pytest.mark.ci
@pytest.mark.parametrize(
    "learner, reason",
    [
        (_pipe(), r"A Pipeline only passes sample weights to its steps with metadata routing enabled"),
        (make_pipeline(StandardScaler(), LogisticRegression()), r"A Pipeline only passes sample weights"),
        (GridSearchCV(_pipe(), {"lasso__alpha": [0.1, 1.0]}), r"The wrapped learner Pipeline\(.*\) does not support"),
        (StackingRegressor([("pipe", _pipe())]), r"The wrapped learner Pipeline\(.*\) does not support"),
        (StackingRegressor([("knn", KNeighborsRegressor())]), r"The wrapped learner KNeighborsRegressor\(\) does not"),
        (TransformedTargetRegressor(regressor=_pipe()), r"The wrapped learner Pipeline\(.*\) does not support"),
        (TransformedTargetRegressor(regressor=KNeighborsRegressor()), r"The wrapped learner KNeighborsRegressor\(\)"),
        (
            StackingRegressor([("lasso", Lasso())], final_estimator=KNeighborsRegressor()),
            r"The wrapped learner KNeighborsRegressor\(\) does not",
        ),
    ],
)
def test_check_supports_sample_weights_invalid_meta_estimators(learner, reason):
    """Pipelines (without metadata routing) and meta-estimators forwarding weights to unsupported learners fail."""
    msg = r"(?s)The ml_g learner .* does not support sample weights\. " + reason
    with pytest.raises(ValueError, match=msg):
        _check_supports_sample_weights(learner, "ml_g")


def _valid_routing_learners():
    pipe_scaler_unweighted = make_pipeline(
        StandardScaler().set_fit_request(sample_weight=False), Lasso().set_fit_request(sample_weight=True)
    )
    return [
        _pipe(request_weights=True),
        pipe_scaler_unweighted,
        GridSearchCV(_pipe(request_weights=True), {"lasso__alpha": [0.1, 1.0]}),
        StackingRegressor([("pipe", _pipe(request_weights=True))]),
        TransformedTargetRegressor(regressor=_pipe(request_weights=True)),
    ]


def _invalid_routing_learners():
    pipe_half_requested = make_pipeline(StandardScaler(), Lasso().set_fit_request(sample_weight=True))
    return [
        _pipe(),
        GridSearchCV(Lasso(), {"alpha": [0.1, 1.0]}),
        GridSearchCV(pipe_half_requested, {"lasso__alpha": [0.1, 1.0]}),
        StackingRegressor([("pipe", _pipe())]),
    ]


@pytest.mark.ci
@pytest.mark.parametrize("i_learner", range(5))
def test_check_supports_sample_weights_metadata_routing_valid(i_learner):
    """With metadata routing, learners whose (nested) steps request the weights pass."""
    with config_context(enable_metadata_routing=True):
        _check_supports_sample_weights(_valid_routing_learners()[i_learner], "ml_g")


@pytest.mark.ci
@pytest.mark.parametrize("i_learner", range(4))
def test_check_supports_sample_weights_metadata_routing_invalid(i_learner):
    """With metadata routing, learners with (nested) steps that do not request the weights fail."""
    msg = r"(?s)The ml_g learner .* does not support sample weights\. .*Metadata routing: \[sample_weight\] are passed but"
    with config_context(enable_metadata_routing=True):
        with pytest.raises(ValueError, match=msg):
            _check_supports_sample_weights(_invalid_routing_learners()[i_learner], "ml_g")
