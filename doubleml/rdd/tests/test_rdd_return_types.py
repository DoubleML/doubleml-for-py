import pytest
import numpy as np
import pandas as pd

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)

n_obs = 300
data_rdd = make_simple_rdd_data(n_obs=n_obs)
data_rdd = pd.DataFrame(data_rdd).drop(columns=["oracle_values"])
dml_data_rdd = dml.DoubleMLData(data_rdd, y_col="Y", d_cols="D", s_col="score")

dml_rdflex = RDFlex(dml_data_rdd, ml_g=Lasso(), ml_m=LogisticRegression())


def _assert_return_types(dml_obj):
    assert isinstance(dml_obj.n_folds, int)
    assert isinstance(dml_obj.n_rep, int)
    assert (isinstance(dml_obj.cutoff, float) | isinstance(dml_obj.cutoff, int))
    assert isinstance(dml_obj.fuzzy, bool)
    assert isinstance(dml_obj.fs_kernel, str)
    assert isinstance(dml_obj.weights, np.ndarray)
    assert dml_obj.weights.shape == (n_obs,)
    assert dml_obj.weights.dtype == float
    assert isinstance(dml_obj.w_mask, np.ndarray)
    assert dml_obj.w_mask.shape == (n_obs,)
    assert dml_obj.w_mask.dtype == bool
    assert isinstance(dml_obj.__str__, str)


def _assert_return_types_after_fit(dml_obj):
    assert isinstance(dml_obj.fit(), RDFlex)
    assert isinstance(dml_obj.__str__, str)
    assert isinstance(dml_obj.n_folds, int)
    assert isinstance(dml_obj.n_rep, int)
    assert (isinstance(dml_obj.cutoff, float) | isinstance(dml_obj.cutoff, int))
    assert isinstance(dml_obj.fuzzy, bool)
    assert isinstance(dml_obj.fs_kernel, str)
    assert isinstance(dml_obj.weights, np.ndarray)
    assert dml_obj.weights.shape == (n_obs,)
    assert dml_obj.weights.dtype == float
    assert isinstance(dml_obj.w_mask, np.ndarray)
    assert dml_obj.w_mask.shape == (n_obs,)
    assert dml_obj.w_mask.dtype == bool
    # TODO: Add Coefficient tests



@pytest.mark.ci
def test_rdd_returntypes():
    _assert_return_types(dml_rdflex)
    _assert_return_types_after_fit(dml_rdflex)
