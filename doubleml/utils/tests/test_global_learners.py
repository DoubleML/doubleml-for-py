import pytest
import numpy as np
from doubleml.utils import GlobalRegressor, GlobalClassifier
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression


@pytest.fixture(scope="module")
def gl_fixture():
    fixture = {
        "GlobalRegressor": GlobalRegressor(base_estimator=LinearRegression()),
        "GlobalClassifier": GlobalClassifier(base_estimator=LogisticRegression(random_state=42)),
        "LinearRegression": LinearRegression(),
        "LogisticRegression": LogisticRegression(random_state=42),
        "X": np.random.normal(0, 1, size=(100, 10)),
        "y_con": np.random.normal(0, 1, size=(100, 1)),
        "y_cat": np.random.binomial(1, 0.5, size=(100, 1)),
        "sample_weight": np.random.random(size=(100, 1))
    }

    return fixture


@pytest.mark.ci
def test_fit(gl_fixture):
    gl_fixture["GlobalRegressor"].fit(y=gl_fixture["y_con"], X=gl_fixture["X"], sample_weight=gl_fixture["sample_weight"])
    gl_fixture["GlobalClassifier"].fit(y=gl_fixture["y_cat"], X=gl_fixture["X"], sample_weight=gl_fixture["sample_weight"])
    gl_fixture["LinearRegression"].fit(y=gl_fixture["y_con"], X=gl_fixture["X"])
    gl_fixture["LogisticRegression"].fit(y=gl_fixture["y_cat"], X=gl_fixture["X"])


@pytest.mark.ci
def test_predict(gl_fixture):
    pred_global_reg = gl_fixture["GlobalRegressor"].predict(X=gl_fixture["X"])
    pred_global_clas = gl_fixture["GlobalClassifier"].predict(X=gl_fixture["X"])
    pred_reg = gl_fixture["LinearRegression"].predict(X=gl_fixture["X"])
    pred_clas = gl_fixture["LogisticRegression"].predict(X=gl_fixture["X"])
    np.allclose(pred_global_reg, pred_reg)
    np.allclose(pred_global_clas, pred_clas)


@pytest.mark.ci
def test_predict_proba(gl_fixture):
    pred_global_clas = gl_fixture["GlobalClassifier"].predict_proba(X=gl_fixture["X"])
    pred_clas = gl_fixture["LogisticRegression"].predict_proba(X=gl_fixture["X"])
    np.allclose(pred_global_clas, pred_clas)


@pytest.mark.ci
def test_clone(gl_fixture):
    try:
        clone_reg = clone(gl_fixture["GlobalRegressor"])
        clone_clas = clone(gl_fixture["GlobalClassifier"])
    except Exception as e:
        pytest.fail(f"clone() raised an exception:\n{str(e)}\n")

    # Test if they still work cloned
    clone_reg.fit(y=gl_fixture["y_con"], X=gl_fixture["X"], sample_weight=gl_fixture["sample_weight"])
    clone_clas.fit(y=gl_fixture["y_cat"], X=gl_fixture["X"], sample_weight=gl_fixture["sample_weight"])

    pred_global_reg = gl_fixture["GlobalRegressor"].predict(X=gl_fixture["X"])
    pred_global_clas = gl_fixture["GlobalClassifier"].predict_proba(X=gl_fixture["X"])
    pred_clone_reg = clone_reg.predict(X=gl_fixture["X"])
    pred_clone_clas = clone_clas.predict_proba(X=gl_fixture["X"])

    np.allclose(pred_global_reg, pred_clone_reg)
    np.allclose(pred_global_clas, pred_clone_clas)
