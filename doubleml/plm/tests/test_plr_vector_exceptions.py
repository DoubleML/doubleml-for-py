"""Validate PLRVector input validation and error handling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

import doubleml as dml
from doubleml.plm.plr_vector import PLRVector


def _make_bivariate_data(n_obs: int = 200, dim_x: int = 5) -> dml.DoubleMLData:
    np.random.seed(42)
    x = np.random.normal(size=(n_obs, dim_x))
    d0 = np.random.normal(size=n_obs)
    d1 = np.random.normal(size=n_obs)
    y = 0.5 * d0 + 0.9 * d1 + x[:, 0] + np.random.normal(size=n_obs)
    df = pd.DataFrame(
        np.column_stack([x, y, d0, d1]),
        columns=[f"X{i + 1}" for i in range(dim_x)] + ["y", "d1", "d2"],
    )
    return dml.DoubleMLData(df, y_col="y", d_cols=["d1", "d2"], x_cols=[f"X{i + 1}" for i in range(dim_x)])


def _make_binary_outcome_bivariate_data(n_obs: int = 100) -> dml.DoubleMLData:
    np.random.seed(11)
    x = np.random.normal(size=(n_obs, 3))
    d0 = (np.random.normal(size=n_obs) > 0).astype(float)
    d1 = (np.random.normal(size=n_obs) > 0).astype(float)
    y = (np.random.normal(size=n_obs) > 0).astype(float)
    df = pd.DataFrame({"y": y, "d1": d0, "d2": d1, "X1": x[:, 0], "X2": x[:, 1], "X3": x[:, 2]})
    return dml.DoubleMLData(df, y_col="y", d_cols=["d1", "d2"], x_cols=["X1", "X2", "X3"])


def _make_iv_data(n_obs: int = 200, dim_x: int = 5) -> dml.DoubleMLData:
    np.random.seed(42)
    x = np.random.normal(size=(n_obs, dim_x))
    d0 = np.random.normal(size=n_obs)
    d1 = np.random.normal(size=n_obs)
    z = np.random.normal(size=n_obs)
    y = 0.5 * d0 + 0.9 * d1 + x[:, 0] + np.random.normal(size=n_obs)
    df = pd.DataFrame(
        np.column_stack([x, y, d0, d1, z]),
        columns=[f"X{i + 1}" for i in range(dim_x)] + ["y", "d1", "d2", "Z1"],
    )
    return dml.DoubleMLData(
        df,
        y_col="y",
        d_cols=["d1", "d2"],
        x_cols=[f"X{i + 1}" for i in range(dim_x)],
        z_cols="Z1",
    )


@pytest.mark.ci
def test_exception_data_type():
    """Non-DoubleMLData input is rejected with a TypeError."""
    msg = r"The data must be of DoubleMLData type\."
    with pytest.raises(TypeError, match=msg):
        PLRVector(pd.DataFrame())


@pytest.mark.ci
def test_exception_instrument():
    """Data carrying instrumental variables (z_cols) is rejected."""
    msg = r"Incompatible data\. .* have been set as instrumental variable\(s\)\."
    with pytest.raises(ValueError, match=msg):
        PLRVector(_make_iv_data())


@pytest.mark.ci
def test_exception_invalid_score():
    """Unknown score string is rejected at construction."""
    msg = r"Invalid score 'invalid'\."
    with pytest.raises(ValueError, match=msg):
        PLRVector(_make_bivariate_data(), score="invalid")


@pytest.mark.ci
def test_exception_iv_type_binary_outcome():
    """IV-type score with binary outcome is rejected."""
    msg = r"For score = 'IV-type', additive probability models \(binary outcomes\) are not supported\."
    with pytest.raises(ValueError, match=msg):
        PLRVector(_make_binary_outcome_bivariate_data(), score="IV-type")


@pytest.mark.ci
def test_exception_n_folds():
    """draw_sample_splitting rejects n_folds < 2."""
    dml_obj = PLRVector(_make_bivariate_data())
    msg = r"n_folds must be an integer >= 2\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=1)


@pytest.mark.ci
def test_exception_n_rep():
    """draw_sample_splitting rejects n_rep < 1."""
    dml_obj = PLRVector(_make_bivariate_data())
    msg = r"n_rep must be an integer >= 1\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_rep=0)


@pytest.mark.ci
def test_exception_missing_learner():
    """fit() fails when no learners are registered."""
    dml_obj = PLRVector(_make_bivariate_data())
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_l' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_exception_missing_partial_learner():
    """fit() fails when ml_m is missing."""
    dml_obj = PLRVector(_make_bivariate_data())
    dml_obj.set_learners(ml_l=Lasso(alpha=0.1))
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_m' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_exception_invalid_learner_class():
    """Passing a class instead of an instance raises TypeError."""
    dml_obj = PLRVector(_make_bivariate_data())
    msg = r"Invalid learner provided for ml_l: provide an instance"
    with pytest.raises(TypeError, match=msg):
        dml_obj.set_learners(ml_l=Lasso)


@pytest.mark.ci
def test_warning_ml_g_partialling_out():
    """Passing ml_g with score='partialling out' triggers a UserWarning."""
    dml_obj = PLRVector(_make_bivariate_data(), score="partialling out")
    with pytest.warns(UserWarning, match=r"not required for score.*ignored"):
        dml_obj.set_learners(ml_l=Lasso(alpha=0.1), ml_m=Lasso(alpha=0.1), ml_g=Lasso(alpha=0.1))


@pytest.mark.ci
def test_cate_not_implemented():
    """cate() raises NotImplementedError on multi-treatment PLR."""
    dml_obj = PLRVector(_make_bivariate_data())
    dml_obj.set_learners(ml_l=Lasso(alpha=0.1), ml_m=Lasso(alpha=0.1))
    dml_obj.fit(n_folds=3)
    with pytest.raises(NotImplementedError, match=r"cate\(\) is not defined for multi-treatment PLR"):
        dml_obj.cate(pd.DataFrame({"const": np.ones(200)}))


@pytest.mark.ci
def test_gate_not_implemented():
    """gate() raises NotImplementedError on multi-treatment PLR."""
    dml_obj = PLRVector(_make_bivariate_data())
    dml_obj.set_learners(ml_l=Lasso(alpha=0.1), ml_m=Lasso(alpha=0.1))
    dml_obj.fit(n_folds=3)
    with pytest.raises(NotImplementedError, match=r"gate\(\) is not defined for multi-treatment PLR"):
        dml_obj.gate(pd.DataFrame({"g": np.ones(200, dtype=bool)}))
