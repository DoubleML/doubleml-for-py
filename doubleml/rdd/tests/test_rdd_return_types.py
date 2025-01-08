import pytest
import numpy as np
import pandas as pd

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)

n_obs = 300
data = make_simple_rdd_data(n_obs=n_obs)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
)
dml_data = dml.DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

dml_rdflex = RDFlex(dml_data, ml_g=Lasso(), ml_m=LogisticRegression())


def _assert_return_types(dml_obj):
    assert isinstance(dml_obj.n_folds, int)
    assert isinstance(dml_obj.n_rep, int)
    assert (isinstance(dml_obj.cutoff, float) | isinstance(dml_obj.cutoff, int))
    assert isinstance(dml_obj.fuzzy, bool)
    assert isinstance(dml_obj.fs_kernel, str)
    assert isinstance(dml_obj.w, np.ndarray)
    assert isinstance(dml_obj.h, np.ndarray)
    assert dml_obj.w.shape == (n_obs,)
    assert dml_obj.w.dtype == float
    assert isinstance(dml_obj.__str__(), str)


def _assert_return_types_after_fit(dml_obj):
    assert isinstance(dml_obj.fit(), RDFlex)
    assert isinstance(dml_obj.__str__(), str)
    assert isinstance(dml_obj.n_folds, int)
    assert isinstance(dml_obj.n_rep, int)
    assert (isinstance(dml_obj.cutoff, float) | isinstance(dml_obj.cutoff, int))
    assert isinstance(dml_obj.fuzzy, bool)
    assert isinstance(dml_obj.fs_kernel, str)
    assert isinstance(dml_obj.w, np.ndarray)
    assert isinstance(dml_obj.h, np.ndarray)
    assert dml_obj.w.shape == (n_obs,)
    assert dml_obj.w.dtype == float
    assert isinstance(dml_obj.confint(), pd.DataFrame)
    # TODO: Add Coefficient tests


@pytest.mark.ci_rdd
def test_rdd_returntypes():
    _assert_return_types(dml_rdflex)
    _assert_return_types_after_fit(dml_rdflex)
