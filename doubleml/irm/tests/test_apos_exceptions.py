import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLAPOS, DoubleMLData
from doubleml.datasets import make_irm_data_discrete_treatments, make_iivm_data

from sklearn.linear_model import Lasso, LogisticRegression

n = 100
data = make_irm_data_discrete_treatments(n_obs=n)
df = pd.DataFrame(
    np.column_stack((data['y'], data['d'], data['x'])),
    columns=['y', 'd'] + ['x' + str(i) for i in range(data['x'].shape[1])]
)

dml_data = DoubleMLData(df, 'y', 'd')

ml_g = Lasso()
ml_m = LogisticRegression()


@pytest.mark.ci
def test_apos_exception_data():
    msg = 'The data must be of DoubleMLData or DoubleMLClusterData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPOS(pd.DataFrame(), ml_g, ml_m, treatment_levels=0)

    msg = 'The data must not contain instrumental variables.'
    with pytest.raises(ValueError, match=msg):
        dml_data_z = make_iivm_data()
        _ = DoubleMLAPOS(dml_data_z, ml_g, ml_m, treatment_levels=0)

    msg = ('Invalid reference_levels. reference_levels has to be an iterable subset or '
           'a single element of the unique treatment levels in the data.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=[1.1])
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=1.1)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=[1, 2.2])


@pytest.mark.ci
def test_apos_exception_scores():
    msg = 'Invalid score MAR. Valid score APO.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=0, score='MAR')


@pytest.mark.ci
def test_apos_exception_trimming_rule():
    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=0, trimming_rule='discard')

    # check the trimming_threshold exceptions
    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=0,
                         trimming_rule='truncate', trimming_threshold="0.1")

    msg = 'Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=0,
                         trimming_rule='truncate', trimming_threshold=0.6)


@pytest.mark.ci
def test_apos_exception_ipw_normalization():
    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=0, normalize_ipw=1)


@pytest.mark.ci
def test_causal_contrast_exceptions():
    msg = r"Apply fit\(\) before causal_contrast\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_obj = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=[0, 1])
        dml_obj.causal_contrast(reference_levels=0)

    dml_obj = DoubleMLAPOS(dml_data, ml_g, ml_m, treatment_levels=[0, 1])
    dml_obj.fit()
    msg = ('Invalid reference_levels. reference_levels has to be an iterable subset of treatment_levels or '
           'a single treatment level.')
    with pytest.raises(ValueError, match=msg):
        dml_obj.causal_contrast(reference_levels=2)
    with pytest.raises(ValueError, match=msg):
        dml_obj.causal_contrast(reference_levels=[2])
    with pytest.raises(ValueError, match=msg):
        dml_obj.causal_contrast(reference_levels=[0, 2])
