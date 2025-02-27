import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

df = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

valid_arguments = {
    "obj_dml_data": dml_data,
    "ml_g": LinearRegression(),
    "ml_m": LogisticRegression(),
    "gt_combinations": [(1, 0, 1)],
}


@pytest.mark.ci
def test_input():
    # data
    msg = r"The data has to be a DoubleMLPanelData object. 0 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"obj_dml_data": 0}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    invalid_data = dml.data.DoubleMLPanelData(
        df, y_col="y", d_cols="d", id_col="id", t_col="t", z_cols=["Z4"], x_cols=["Z1", "Z2", "Z3"]
    )
    msg = r"Incompatible data. Z4 have been set as instrumental variable\(s\)."
    with pytest.raises(NotImplementedError, match=msg):
        invalid_arguments = {"obj_dml_data": invalid_data}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    # control group
    msg = r"The control group has to be one of \['never_treated', 'not_yet_treated'\]. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"control_group": 0}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    # propensity score adjustments
    msg = "in_sample_normalization indicator has to be boolean. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"in_sample_normalization": "test"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    # score
    msg = 'Invalid score test. Valid score observational or experimental.'
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"score": "test"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))
