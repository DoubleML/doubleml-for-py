import pytest
import numpy as np
import pandas as pd

import doubleml as dml
from doubleml.rdd.datasets import make_simple_rdd_data

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)

data_rdd = make_simple_rdd_data(n_obs=300)
data_rdd = pd.DataFrame(data_rdd).drop(columns=["oracle_values"])
dml_data_rdd = dml.DoubleMLData(data_rdd, y_col="Y", d_cols="D", s_col="score")

dml_rdflex = dml.rdd.RDFlex(dml_data_rdd, ml_g=Lasso(), ml_m=LogisticRegression())


def _assert_resampling_default_settings(dml_obj):
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.cutoff == 0
    assert dml_obj.h_fs is None
    assert dml_obj.fs_kernel == "uniform"


@pytest.mark.ci
def test_rdd_defaults():
    _assert_resampling_default_settings(dml_rdflex)
