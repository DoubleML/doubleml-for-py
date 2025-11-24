import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLLPLR
from doubleml.plm.datasets import make_lplr_LZZ2020

np.random.seed(3141)

dml_data_lplr = make_lplr_LZZ2020(n_obs=10)
n_obs = dml_data_lplr.n_obs
ml_M = LogisticRegression()
ml_t = Lasso()
ml_m = Lasso()
dml_lplr = DoubleMLLPLR(dml_data_lplr, ml_M, ml_t, ml_m, n_folds=7, n_rep=8, draw_sample_splitting=False)


@pytest.mark.ci
def test_doubleml_exceptions_double_sample_splitting():
    smpls = (np.arange(n_obs), np.arange(n_obs))
    msg = "set_sample_splitting not supported for double sample splitting."
    with pytest.raises(ValueError, match=msg):
        dml_lplr.set_sample_splitting(smpls)

    dml_lplr._is_cluster_data = True
    msg = "Cluster data not supported for double sample splitting."
    with pytest.raises(ValueError, match=msg):
        dml_lplr.draw_sample_splitting()
