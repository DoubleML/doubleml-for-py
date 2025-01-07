import pytest
import numpy as np
import pandas as pd

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)

n_obs = 300
data = make_simple_rdd_data(n_obs=n_obs, fuzzy=False)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
)
dml_data = dml.DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

dml_rdflex = RDFlex(dml_data, ml_g=Lasso(), ml_m=LogisticRegression())


def _assert_resampling_default_settings(dml_obj):
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.cutoff == 0
    assert dml_obj.h_fs is not None
    assert dml_obj.fs_kernel == "triangular"
    assert dml_obj.fuzzy is False


@pytest.mark.ci_rdd
def test_rdd_defaults():
    _assert_resampling_default_settings(dml_rdflex)
