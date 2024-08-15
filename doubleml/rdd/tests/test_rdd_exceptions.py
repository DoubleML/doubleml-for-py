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

    # score column
    msg = 'Incompatible data. Score variable has not been set. '
    with pytest.raises(ValueError, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._s_col = None
        _ = RDFlex(tmp_dml_data, ml_g)
    msg = 'Incompatible data. Score variable has to be continuous. '
    with pytest.raises(ValueError, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._s = tmp_dml_data._d
        _ = RDFlex(tmp_dml_data, ml_g)

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
def test_rdd_exception_cutoff():
    msg = "Cutoff value has to be a float or int. Object of type <class 'list'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, cutoff=[2])

    msg = 'Cutoff value is not within the range of the score variable. '
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, cutoff=2)


@pytest.mark.ci
def test_rdd_exception_learner():

    # ml_g
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier but the outcome variable is not'
           ' binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g=LogisticRegression())

    # ml_m
    msg = r'Invalid learner provided for ml_m: Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m=Lasso())
    msg = 'Fuzzy design requires a classifier ml_m for treatment assignment.'
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g)

    msg = ('A learner ml_m has been provided for for a sharp design but will be ignored. '
           'A learner ml_m is not required for estimation.')
    with pytest.warns(UserWarning, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._data['sharp_d'] = (tmp_dml_data.s >= 0)
        tmp_dml_data.d_cols = 'sharp_d'
        _ = RDFlex(tmp_dml_data, ml_g, ml_m)


@pytest.mark.ci
def test_rdd_exception_resampling():
    # n_folds
    msg = r"The number of folds must be of int type. \[1\] of type <class 'list'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, n_folds=[1])
    msg = 'The number of folds greater or equal to 2. 1 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, n_folds=1)

    # n_rep
    msg = r"The number of repetitions for the sample splitting must be of int type. \[0\] of type <class 'list'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, n_rep=[0])
    msg = 'The number of repetitions for the sample splitting has to be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, n_rep=0)
