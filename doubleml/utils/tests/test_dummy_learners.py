import pytest
import numpy as np
from doubleml.utils import DMLDummyRegressor, DMLDummyClassifier
from sklearn.base import clone


@pytest.fixture(scope="module")
def dl_fixture():
    fixture = {
        "DMLDummyRegressor": DMLDummyRegressor(),
        "DMLDummyClassifier": DMLDummyClassifier(),
        "X": np.random.normal(0, 1, size=(100, 10)),
        "y_con": np.random.normal(0, 1, size=(100, 1)),
        "y_cat": np.random.binomial(1, 0.5, size=(100, 1)),
    }

    return fixture


@pytest.mark.ci
def test_fit(dl_fixture):
    msg = "Accessed fit method of DMLDummyRegressor!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["DMLDummyRegressor"].fit(dl_fixture["X"], dl_fixture["y_con"])
    msg = "Accessed fit method of DMLDummyClassifier!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["DMLDummyClassifier"].fit(dl_fixture["X"], dl_fixture["y_cat"])


@pytest.mark.ci
def test_predict(dl_fixture):
    msg = "Accessed predict method of DMLDummyRegressor!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["DMLDummyRegressor"].predict(dl_fixture["X"])
    msg = "Accessed predict method of DMLDummyClassifier!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["DMLDummyClassifier"].predict(dl_fixture["X"])


@pytest.mark.ci
def test_clone(dl_fixture):
    try:
        _ = clone(dl_fixture["DMLDummyRegressor"])
        _ = clone(dl_fixture["DMLDummyClassifier"])
    except Exception as e:
        pytest.fail(f"clone() raised an exception:\n{str(e)}\n")
