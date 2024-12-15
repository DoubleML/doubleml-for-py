import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from doubleml.utils import GlobalRegressor, GlobalClassifier


@pytest.mark.ci
def test_global_regressor_input():
    msg = "base_estimator must be a regressor. Got LogisticRegression instead."
    with pytest.raises(ValueError, match=msg):
        _ = GlobalRegressor(base_estimator=LogisticRegression(random_state=42))


@pytest.mark.ci
def test_global_classifier_input():
    msg = "base_estimator must be a classifier. Got LinearRegression instead."
    with pytest.raises(ValueError, match=msg):
        _ = GlobalClassifier(base_estimator=LinearRegression())
