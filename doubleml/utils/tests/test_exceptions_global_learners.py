import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.utils import GlobalClassifier, GlobalRegressor


@pytest.mark.ci
def test_global_regressor_input():
    msg = "base_estimator must be a regressor. Got LogisticRegression instead."
    with pytest.raises(ValueError, match=msg):
        reg = GlobalRegressor(base_estimator=LogisticRegression(random_state=42))
        reg.fit(X=[[1, 2], [3, 4]], y=[1, 2])


@pytest.mark.ci
def test_global_classifier_input():
    msg = "base_estimator must be a classifier. Got LinearRegression instead."
    with pytest.raises(ValueError, match=msg):
        clas = GlobalClassifier(base_estimator=LinearRegression())
        clas.fit(X=[[1, 2], [3, 4]], y=[1, 2])
