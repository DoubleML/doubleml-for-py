import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data

np.random.seed(3141)

n_obs = 300
data = make_simple_rdd_data(n_obs=n_obs)
data["Y_bin"] = (data["Y"] < np.median(data["Y"])).astype("int")


df = pd.DataFrame(
    np.column_stack((data["Y_bin"], data["D"], data["score"], data["X"])),
    columns=["y", "d", "score"] + ["x" + str(i) for i in range(data["X"].shape[1])],
)
dml_data = dml.DoubleMLData(df, y_col="y", d_cols="d", s_col="score")


@pytest.mark.ci_rdd
def test_rdd_classifier():
    dml_rdflex = RDFlex(dml_data, ml_g=LogisticRegression(), ml_m=LogisticRegression(), fuzzy=True)
    dml_rdflex.fit()
