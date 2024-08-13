import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLData
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.rdd import RDFlex

from sklearn.linear_model import Lasso, LogisticRegression

n = 100
data = make_simple_rdd_data(n_obs=n)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['x'].shape[1])]
)

dml_data = DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

ml_g = Lasso()
ml_m = LogisticRegression()


@pytest.mark.ci
def test_rdd_exception_learner():
    msg = 'This test will fail!'
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m=None)
