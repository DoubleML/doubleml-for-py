import pytest
import pandas as pd
import numpy as np
import copy

from doubleml import DoubleMLData
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.rdd import RDFlex

from sklearn.linear_model import Lasso, LogisticRegression

n = 500
data = make_simple_rdd_data(n_obs=n)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
)

dml_data = DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

ml_g = Lasso()
ml_m = LogisticRegression()


@pytest.mark.ci
def test_rdd_exception_data():
    # DoubleMLData
    msg = r"The data must be of DoubleMLData type. \[\] of type <class 'list'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex([], ml_g)

    # existing instruments
    msg = r'Incompatible data. x0 have been set as instrumental variable\(s\). '
    with pytest.raises(ValueError, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._z_cols = ['x0']
        _ = RDFlex(tmp_dml_data, ml_g)

    # treatment exceptions
    msg = ('Incompatible data. '
           'To fit an RDFlex model with DML '
           'exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    # multiple treatment variables
    with pytest.raises(ValueError, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._d_cols = ['d', 'x0']
        _ = RDFlex(tmp_dml_data, ml_g)
    # non-binary treatment
    with pytest.raises(ValueError, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data.x_cols = ['x1']  # reset x to only x1 to enable setting d to x0
        tmp_dml_data.d_cols = ['x0']
        _ = RDFlex(tmp_dml_data, ml_g)


@pytest.mark.ci
def test_rdd_exception_learner():
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier but the outcome variable is not'
           ' binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g=LogisticRegression())

    msg = r'Invalid learner provided for ml_m: Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m=Lasso())


