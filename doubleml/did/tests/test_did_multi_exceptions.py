import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

df = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])
# df_binary_outcome = df.copy()
# df_binary_outcome["y"] = (df_binary_outcome["y"] > df_binary_outcome["y"].median()).astype(int)
# dml_data_binary_outcome = dml.data.DoubleMLPanelData(
#     df_binary_outcome, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
# )

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
    msg = "Invalid score test. Valid score observational or experimental."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"score": "test"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    # trimming
    msg = "Invalid trimming_rule discard. Valid trimming_rule truncate."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"trimming_rule": "discard"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"trimming_threshold": "test"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"trimming_threshold": 0.6}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))


@pytest.mark.ci
def test_exception_learners():
    msg = (
        r"The ml_g learner LogisticRegression\(\) was identified as classifier but "
        + "the outcome variable is not binary with values 0 and 1."
    )
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"ml_g": LogisticRegression()}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))


@pytest.mark.ci
def test_exception_gt_combinations():
    msg = r"gt_combinations must be one of \['standard', 'all'\]. test was passed."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"gt_combinations": "test"}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "gt_combinations must be a list. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"gt_combinations": 1}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "gt_combinations must not be empty."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"gt_combinations": []}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "gt_combinations must be a list of tuples. At least one element is not a tuple."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"gt_combinations": [1]}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))

    msg = "gt_combinations must be a list of tuples with 3 elements. At least one tuple has not 3 elements."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"gt_combinations": [(1, 0)]}
        _ = dml.did.DoubleMLDIDMulti(**(valid_arguments | invalid_arguments))


@pytest.mark.ci
def test_exceptions_aggregate():
    dml_obj = dml.did.DoubleMLDIDMulti(**valid_arguments)
    # test without fit()
    msg = r"Apply fit\(\) before aggregate\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_obj.aggregate()

    dml_obj.fit()

    # Test non-string input
    msg = "aggregation must be a string or dictionary. 123 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_obj.aggregate(aggregation=123)

    # Test invalid string value
    msg = "aggregation must be one of \\['group'\\]. invalid was passed."
    with pytest.raises(ValueError, match=msg):
        dml_obj.aggregate(aggregation="invalid")
