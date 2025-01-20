import numpy as np
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import check_estimator

from doubleml.utils import GlobalClassifier, GlobalRegressor


def parse_version(version):
    return tuple(map(int, version.split('.')[:2]))


# TODO(0.10) can be removed if the sklearn dependency is bumped to 1.6.0
sklearn_post_1_6 = parse_version(sklearn_version) >= (1, 6)


@pytest.fixture(
    scope="module", params=[LinearRegression(), RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)]
)
def regressor(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[LogisticRegression(random_state=42), RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)],
)
def classifier(request):
    return request.param


@pytest.mark.ci
def test_global_regressor(regressor):
    if sklearn_post_1_6:
        check_estimator(
            estimator=GlobalRegressor(base_estimator=regressor),
            expected_failed_checks={
                "check_sample_weight_equivalence_on_dense_data": "weights are ignored",
                "check_estimators_nan_inf": "allowed for some estimators",
            },
        )
    else:
        # TODO(0.10) can be removed if the sklearn dependency is bumped to 1.6.0
        pytest.skip("sklearn version is too old for this test")


@pytest.mark.ci
def test_global_classifier(classifier):
    if sklearn_post_1_6:
        check_estimator(
            estimator=GlobalClassifier(base_estimator=classifier),
            expected_failed_checks={
                "check_sample_weight_equivalence_on_dense_data": "weights are ignored",
                "check_estimators_nan_inf": "allowed for some estimators",
            },
        )
    else:
        # TODO(0.10) can be removed if the sklearn dependency is bumped to 1.6.0
        pytest.skip("sklearn version is too old for this test")


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

    global_clas_pred_proba = global_clas.predict_proba(X=X)
    weighted_clas_pred_proba = weighted_clas.predict_proba(X=X)
    unweighted_clas_pred_proba = unweighted_clas.predict_proba(X=X)

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
        "unweighted_clas_pred_proba": unweighted_clas_pred_proba,
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


@pytest.fixture(scope="module")
def gl_stacking_fixture():

    regressor = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)

    X = np.random.normal(0, 1, size=(100, 2))
    y_con = np.random.normal(0, 1, size=(100))
    y_cat = np.random.binomial(1, 0.5, size=(100))
    sample_weight = np.random.random(size=(100))

    kf = KFold(n_splits=2, shuffle=False)

    global_reg = StackingRegressor(
        [
            ("global", GlobalRegressor(base_estimator=clone(regressor))),
            ("lr", GlobalRegressor(LinearRegression())),
        ],
        final_estimator=GlobalRegressor(LinearRegression()),
        cv=kf,
    )
    unweighted_reg = StackingRegressor(
        [("global", clone(regressor)), ("lr", LinearRegression())],
        final_estimator=LinearRegression(),
        cv=kf,
    )

    global_clas = StackingClassifier(
        [
            ("global", GlobalClassifier(base_estimator=clone(classifier))),
            ("lr", GlobalClassifier(LogisticRegression(random_state=42))),
        ],
        final_estimator=GlobalClassifier(LogisticRegression(random_state=42)),
        cv=kf,
    )
    unweighted_clas = StackingClassifier(
        [
            ("global", clone(classifier)),
            ("lr", LogisticRegression(random_state=42)),
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=kf,
    )

    # fit models
    global_reg.fit(y=y_con, X=X, sample_weight=sample_weight)
    unweighted_reg.fit(y=y_con, X=X)

    global_clas.fit(y=y_cat, X=X, sample_weight=sample_weight)
    unweighted_clas.fit(y=y_cat, X=X)

    global_reg_pred = global_reg.predict(X=X)
    unweighted_reg_pred = unweighted_reg.predict(X=X)

    global_clas_pred = global_clas.predict(X=X)
    unweighted_clas_pred = unweighted_clas.predict(X=X)

    global_clas_pred_proba = global_clas.predict_proba(X=X)

    unweighted_clas_pred_proba = unweighted_clas.predict_proba(X=X)

    result_dict = {
        "global_reg_pred": global_reg_pred,
        "unweighted_reg_pred": unweighted_reg_pred,
        "global_clas_pred": global_clas_pred,
        "unweighted_clas_pred": unweighted_clas_pred,
        "global_clas_pred_proba": global_clas_pred_proba,
        "unweighted_clas_pred_proba": unweighted_clas_pred_proba,
    }

    return result_dict


@pytest.mark.ci
def test_stacking_predict(gl_stacking_fixture):
    assert np.allclose(gl_stacking_fixture["global_reg_pred"], gl_stacking_fixture["unweighted_reg_pred"])
    assert np.allclose(gl_stacking_fixture["global_clas_pred"], gl_stacking_fixture["unweighted_clas_pred"])


@pytest.mark.ci
def test_stacking_predict_proba(gl_stacking_fixture):
    assert np.allclose(gl_stacking_fixture["global_clas_pred_proba"], gl_stacking_fixture["unweighted_clas_pred_proba"])
