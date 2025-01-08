import pytest
import pandas as pd
import numpy as np
import copy

from doubleml import DoubleMLData
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.rdd import RDFlex

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Lasso, LogisticRegression

n = 500
data = make_simple_rdd_data(n_obs=n, fuzzy=False)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
)

dml_data = DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

ml_g = Lasso()
ml_m = LogisticRegression()


# dummy learners for testing
class DummyRegressorNoSampleWeight(BaseEstimator, RegressorMixin):
    """
    A dummy regressor that predicts the mean of the target values,
    and does not support sample weights.
    """
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.mean_)


class DummyClassifierNoSampleWeight(BaseEstimator, ClassifierMixin):
    """
    A dummy classifier that predicts the most frequent class,
    and does not support sample weights.
    """
    def fit(self, X, y):
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        self.most_frequent_ = self.classes_[np.argmax(self.counts_)]
        return self

    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.most_frequent_)

    def predict_proba(self, X):
        return np.column_stack(
            (np.full(shape=(X.shape[0],), fill_value=1),
             np.full(shape=(X.shape[0],), fill_value=0))
        )


@pytest.mark.ci_rdd
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


@pytest.mark.ci_rdd
def test_rdd_exception_cutoff():
    msg = "Cutoff value has to be a float or int. Object of type <class 'list'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, cutoff=[200])

    msg = 'Cutoff value is not within the range of the score variable. '
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, cutoff=200)


@pytest.mark.ci_rdd
def test_rdd_warning_fuzzy():
    msg = 'A sharp RD design is being estimated, but the data indicate that the design is fuzzy.'
    with pytest.warns(UserWarning, match=msg):
        _ = RDFlex(dml_data, ml_g, cutoff=0.1)


@pytest.mark.ci_rdd
def test_rdd_warning_treatment_assignment():
    msg = ("Treatment probability within bandwidth left from cutoff higher than right from cutoff.\n"
           "Treatment assignment might be based on the wrong side of the cutoff.")
    with pytest.warns(UserWarning, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._s = -1.0*tmp_dml_data._s
        _ = RDFlex(tmp_dml_data, ml_g, ml_m, fuzzy=True)


@pytest.mark.ci_rdd
def test_rdd_exception_learner():

    # ml_g
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier but the outcome variable is not'
           ' binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g=LogisticRegression())
    msg = (r"The ml_g learner DummyRegressorNoSampleWeight\(\) does not support sample weights. Please choose a learner"
           " that supports sample weights.")
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g=DummyRegressorNoSampleWeight(), ml_m=ml_m)

    # ml_m
    msg = r'Invalid learner provided for ml_m: Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m=Lasso(), fuzzy=True)
    msg = 'Fuzzy design requires a classifier ml_m for treatment assignment.'
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, fuzzy=True)
    msg = (r"The ml_m learner DummyClassifierNoSampleWeight\(\) does not support sample weights. Please choose a learner"
           " that supports sample weights.")
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m=DummyClassifierNoSampleWeight(), fuzzy=True)

    msg = ('A learner ml_m has been provided for for a sharp design but will be ignored. '
           'A learner ml_m is not required for estimation.')
    with pytest.warns(UserWarning, match=msg):
        tmp_dml_data = copy.deepcopy(dml_data)
        tmp_dml_data._data['sharp_d'] = (tmp_dml_data.s >= 0)
        tmp_dml_data.d_cols = 'sharp_d'
        _ = RDFlex(tmp_dml_data, ml_g, ml_m, fuzzy=False)


@pytest.mark.ci_rdd
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


@pytest.mark.ci_rdd
def test_rdd_exception_kernel():
    msg = "fs_kernel must be either a string or a callable. 2 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, fs_kernel=2)
    msg = r"Invalid kernel 'rbf'. Valid kernels are \['uniform', 'triangular', 'epanechnikov'\]."
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, fs_kernel='rbf')


@pytest.mark.ci_rdd
def test_rdd_exception_h_fs():
    msg = "Initial bandwidth 'h_fs' has to be a float. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, h_fs=1)


@pytest.mark.ci_rdd
def test_rdd_exception_fs_specification():
    msg = "fs_specification must be a string. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, fs_specification=1)

    msg = ("Invalid fs_specification 'local_constant'. "
           r"Valid specifications are \['cutoff', 'cutoff and score', 'interacted cutoff and score'\].")
    with pytest.raises(ValueError, match=msg):
        _ = RDFlex(dml_data, ml_g, ml_m, fs_specification='local_constant')


@pytest.mark.ci_rdd
def test_rdd_exception_fit():
    rdd_model = RDFlex(dml_data, ml_g, ml_m)
    msg = (r"The number of iterations for the iterative bandwidth fitting must be of int type. \[0\] of type <class 'list'> "
           "was passed.")
    with pytest.raises(TypeError, match=msg):
        rdd_model.fit(n_iterations=[0])

    msg = 'The number of iterations for the iterative bandwidth fitting has to be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        rdd_model.fit(n_iterations=0)
