import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLData
from doubleml.sscf import DoubleMLSSCF, make_green_silence_data


@pytest.fixture(scope="module")
def dml_data():
    np.random.seed(3141)
    return make_green_silence_data(
        n_obs=400, dim_x=4, dim_z=3, n_deciles=3, n_active_outcome=2, n_active_selection_x=1, n_active_selection_z=2
    )


def _learners():
    return LogisticRegression(max_iter=1000), LinearRegression(), LinearRegression()


def test_invalid_data_type():
    ml_pi, ml_g, ml_m = _learners()
    with pytest.raises(TypeError, match="DoubleMLData"):
        _ = DoubleMLSSCF(pd.DataFrame({"y": [1.0], "d": [1]}), ml_pi, ml_g, ml_m)


def test_non_binary_selection_indicator():
    np.random.seed(3141)
    n = 200
    df = pd.DataFrame(np.random.normal(size=(n, 4)), columns=["X1", "X2", "U1", "y"])
    df["d"] = np.random.choice([0, 1, 2], size=n)
    data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=["X1", "X2"], z_cols=["U1"])
    ml_pi, ml_g, ml_m = _learners()
    with pytest.raises(ValueError, match="binary"):
        _ = DoubleMLSSCF(data, ml_pi, ml_g, ml_m)


def test_multiple_treatment_variables():
    np.random.seed(3141)
    n = 200
    df = pd.DataFrame(np.random.normal(size=(n, 3)), columns=["X1", "U1", "y"])
    df["d1"] = np.random.binomial(1, 0.5, n)
    df["d2"] = np.random.binomial(1, 0.5, n)
    data = DoubleMLData(df, y_col="y", d_cols=["d1", "d2"], x_cols=["X1"], z_cols=["U1"])
    ml_pi, ml_g, ml_m = _learners()
    with pytest.raises(ValueError, match="exactly one variable"):
        _ = DoubleMLSSCF(data, ml_pi, ml_g, ml_m)


def test_missing_exclusion_restrictions_warns():
    np.random.seed(3141)
    n = 200
    df = pd.DataFrame(np.random.normal(size=(n, 3)), columns=["X1", "X2", "y"])
    df["d"] = np.random.binomial(1, 0.5, n)
    data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=["X1", "X2"])
    ml_pi, ml_g, ml_m = _learners()
    with pytest.warns(UserWarning, match="exclusion restrictions"):
        _ = DoubleMLSSCF(data, ml_pi, ml_g, ml_m)


def test_invalid_score(dml_data):
    ml_pi, ml_g, ml_m = _learners()
    with pytest.raises(ValueError):
        _ = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, score="nonignorable")


def test_invalid_nested_cross_fitting(dml_data):
    ml_pi, ml_g, ml_m = _learners()
    with pytest.raises(TypeError, match="nested_cross_fitting"):
        _ = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, nested_cross_fitting="yes")


def test_regressor_passed_as_classifier(dml_data):
    _, ml_g, ml_m = _learners()
    with pytest.raises(TypeError):
        _ = DoubleMLSSCF(dml_data, LinearRegression(), ml_g, ml_m)


def test_classifier_passed_as_regressor(dml_data):
    """Following the DoubleML convention a classifier in a regressor slot only warns."""
    ml_pi, _, ml_m = _learners()
    with pytest.warns(UserWarning, match="no regressor"):
        _ = DoubleMLSSCF(dml_data, ml_pi, LogisticRegression(), ml_m)


def test_methods_before_fit(dml_data):
    ml_pi, ml_g, ml_m = _learners()
    dml_obj = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, n_folds=3)
    with pytest.raises(ValueError, match="fit"):
        dml_obj.selection_bias_test()
    with pytest.raises(ValueError, match="fit"):
        dml_obj.bias_quantification()
    with pytest.raises(ValueError, match="fit"):
        _ = dml_obj.imr


def test_invalid_test_level(dml_data):
    ml_pi, ml_g, ml_m = _learners()
    dml_obj = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, n_folds=3)
    dml_obj.fit()
    with pytest.raises(ValueError, match="significance level"):
        dml_obj.selection_bias_test(level=1.5)
