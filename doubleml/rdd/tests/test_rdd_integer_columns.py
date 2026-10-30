"""RDFlex gives the same results whether outcome and treatment are stored as integers or as floats."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data


def _rdd_frame(fuzzy):
    np.random.seed(3141)
    data = make_simple_rdd_data(n_obs=1000, fuzzy=fuzzy)
    x_cols = ["x" + str(i) for i in range(data["X"].shape[1])]
    return pd.DataFrame(
        np.column_stack((data["Y"], data["D"], data["score"], data["X"])),
        columns=["y", "d", "score"] + x_cols,
    )


def _fit_rdflex(df, ml_g, ml_m=None, fuzzy=False):
    dml_data = dml.DoubleMLRDDData(df, y_col="y", d_cols="d", score_col="score")
    np.random.seed(42)
    return RDFlex(dml_data, ml_g=ml_g, ml_m=ml_m, fuzzy=fuzzy).fit()


@pytest.mark.ci_rdd
def test_rdd_fuzzy_integer_treatment():
    """An integer-coded treatment gives the same fuzzy estimate as the same treatment stored as float."""
    df = _rdd_frame(fuzzy=True)
    rdflex_float = _fit_rdflex(df, LinearRegression(), LogisticRegression(), fuzzy=True)
    rdflex_int = _fit_rdflex(df.astype({"d": int}), LinearRegression(), LogisticRegression(), fuzzy=True)

    assert np.allclose(rdflex_int.coef, rdflex_float.coef)
    assert np.allclose(rdflex_int.se, rdflex_float.se)


@pytest.mark.ci_rdd
def test_rdd_sharp_integer_binary_outcome():
    """An integer-coded binary outcome with a classifier gives the same sharp estimate as the outcome stored as float."""
    df = _rdd_frame(fuzzy=False)
    df["y"] = (df["y"] > df["y"].median()).astype(float)
    rdflex_float = _fit_rdflex(df, LogisticRegression())
    rdflex_int = _fit_rdflex(df.astype({"y": int}), LogisticRegression())

    assert np.allclose(rdflex_int.coef, rdflex_float.coef)
    assert np.allclose(rdflex_int.se, rdflex_float.se)
