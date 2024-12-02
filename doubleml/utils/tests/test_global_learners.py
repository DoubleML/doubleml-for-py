import pytest
import numpy as np
from doubleml.utils import GlobalRegressor, GlobalClassifier
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


@pytest.fixture(scope='module',
                params=[LinearRegression(),
                        RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)])
def regressor(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression(random_state=42),
                        RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)])
def classifier(request):
    return request.param


@pytest.fixture(scope="module")
def gl_fixture(regressor, classifier):

    global_reg = GlobalRegressor(base_estimator=regressor)
    weighted_reg = clone(regressor)
    unweighted_reg = clone(regressor)

    global_clas = GlobalClassifier(base_estimator=classifier)
    weighted_clas = clone(classifier)
    unweighted_clas = clone(classifier)

    X = np.random.normal(0, 1, size=(100, 10))
    y_con = np.random.normal(0, 1, size=(100))
    y_cat = np.random.binomial(1, 0.5, size=(100))
    sample_weight = np.random.random(size=(100))

    # fit models
    global_reg.fit(y=y_con, X=X, sample_weight=sample_weight)
    weighted_reg.fit(y=y_con, X=X, sample_weight=sample_weight)
    unweighted_reg.fit(y=y_con, X=X)

    global_clas.fit(y=y_cat, X=X, sample_weight=sample_weight)
    weighted_clas.fit(y=y_cat, X=X, sample_weight=sample_weight)
    unweighted_clas.fit(y=y_cat, X=X)

    global_reg_pred = global_reg.predict(X=X)
    weighted_reg_pred = weighted_reg.predict(X=X)
    unweighted_reg_pred = unweighted_reg.predict(X=X)

    global_clas_pred = global_clas.predict(X=X)
    weighted_clas_pred = weighted_clas.predict(X=X)
    unweighted_clas_pred = unweighted_clas.predict(X=X)

    global_clas_pred_proba = global_clas.predict(X=X)
    weighted_clas_pred_proba = weighted_clas.predict(X=X)
    unweighted_clas_pred_proba = unweighted_clas.predict(X=X)

    result_dict = {
        "GlobalRegressor": global_reg,
        "WeightedRegressor": weighted_reg,
        "UnweightedRegressor": unweighted_reg,
        "GlobalClassifier": global_clas,
        "WeightedClassifier": weighted_clas,
        "UnweightedClassifier": unweighted_clas,
        "X": X,
        "y_con": y_con,
        "y_cat": y_cat,
        "sample_weight": sample_weight,
        "global_reg_pred": global_reg_pred,
        "weighted_reg_pred": weighted_reg_pred,
        "unweighted_reg_pred": unweighted_reg_pred,
        "global_clas_pred": global_clas_pred,
        "weighted_clas_pred": weighted_clas_pred,
        "unweighted_clas_pred": unweighted_clas_pred,
        "global_clas_pred_proba": global_clas_pred_proba,
        "weighted_clas_pred_proba": weighted_clas_pred_proba,
        "unweighted_clas_pred_proba": unweighted_clas_pred_proba
    }

    return result_dict


@pytest.mark.ci
def test_predict(gl_fixture):
    assert np.allclose(gl_fixture["global_reg_pred"], gl_fixture["unweighted_reg_pred"])
    assert np.allclose(gl_fixture["global_clas_pred"], gl_fixture["unweighted_clas_pred"])

    assert not np.allclose(gl_fixture["global_reg_pred"], gl_fixture["weighted_reg_pred"])
    assert not np.allclose(gl_fixture["global_clas_pred"], gl_fixture["weighted_clas_pred"])


@pytest.mark.ci
def test_predict_proba(gl_fixture):
    assert np.allclose(gl_fixture["global_clas_pred_proba"], gl_fixture["unweighted_clas_pred_proba"])
    assert not np.allclose(gl_fixture["global_clas_pred_proba"], gl_fixture["weighted_clas_pred_proba"])


@pytest.mark.ci
def test_clone(gl_fixture):
    clone_reg = None
    clone_clas = None

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
