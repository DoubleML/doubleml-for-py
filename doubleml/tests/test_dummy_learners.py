import pytest
import numpy as np
from doubleml.utils import dummy_regressor, dummy_classifier
from sklearn.base import clone


@pytest.fixture(scope="module")
def dl_fixture():
    fixture = {
        "dummy_regressor": dummy_regressor(),
        "dummy_classifier": dummy_classifier(),
        "X": np.random.normal(0, 1, size=(100, 10)),
        "y_con": np.random.normal(0, 1, size=(100, 1)),
        "y_cat": np.random.binomial(1, 0.5, size=(100, 1)),
    }

    return fixture


@pytest.mark.ci
def test_fit(dl_fixture):
    msg = "Accessed fit method of dummy_regressor!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["dummy_regressor"].fit(dl_fixture["X"], dl_fixture["y_con"])
    msg = "Accessed fit method of dummy_classifier!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["dummy_classifier"].fit(dl_fixture["X"], dl_fixture["y_cat"])


@pytest.mark.ci
def test_predict(dl_fixture):
    msg = "Accessed predict method of dummy_regressor!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["dummy_regressor"].predict(dl_fixture["X"])
    msg = "Accessed predict method of dummy_classifier!"
    with pytest.raises(AttributeError, match=msg):
        dl_fixture["dummy_classifier"].predict(dl_fixture["X"])


@pytest.mark.ci
def test_clone(dl_fixture):
    try:
        _ = clone(dl_fixture["dummy_regressor"])
        _ = clone(dl_fixture["dummy_classifier"])
    except Error as e:
        pytest.fail(f"clone() raised an exception:\n{str(e)}\n")
