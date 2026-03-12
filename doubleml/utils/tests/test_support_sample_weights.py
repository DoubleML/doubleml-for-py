import re

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
        "Please choose a learner that supports sample weights."
    )
    with pytest.raises(ValueError, match=msg):
        _check_supports_sample_weights(learner, learner_name)
