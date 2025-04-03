from unittest.mock import patch

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

    msg = (
        'A learner ml_m has been provided for score = "experimental" but will be ignored. '
        "A learner ml_m is not required for estimation."
    )
    with pytest.warns(UserWarning, match=msg):
        invalid_arguments = {"score": "experimental"}
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
    msg = "aggregation must be one of \\['group', 'time', 'eventstudy'\\]. invalid was passed."
    with pytest.raises(ValueError, match=msg):
        dml_obj.aggregate(aggregation="invalid")


@pytest.mark.ci
def test_check_external_predictions():
    # Create DID instance
    model = dml.did.DoubleMLDIDMulti(**valid_arguments)

    # Test 1: Invalid type (not a dictionary)
    invalid_pred = ["not a dict"]
    with pytest.raises(TypeError, match="external_predictions must be a dictionary"):
        model.fit(external_predictions=invalid_pred)

    # Test 2: Invalid keys in top-level dictionary
    invalid_keys = {"invalid_key": {}}
    with pytest.raises(ValueError, match="external_predictions must be a subset of all gt_combinations"):
        model.fit(external_predictions=invalid_keys)

    # Test 3: Invalid type for nested prediction dictionary
    invalid_nested = {model.gt_labels[0]: "not a dict"}
    msg = r"external_predictions\[ATT\(1,0,1\)\] must be a dictionary\. Object of type <class 'str'> passed\."
    with pytest.raises(TypeError, match=msg):
        model.fit(external_predictions=invalid_nested)

    # Test 4: Invalid keys in nested prediction dictionary
    invalid_learner = {model.gt_labels[0]: {"invalid_learner": None}}
    with pytest.raises(ValueError, match="must be a subset of "):
        model.fit(external_predictions=invalid_learner)

    # Test 5: Valid external predictions should not raise
    valid_pred = {model.gt_labels[0]: {"ml_g0": None, "ml_g1": None, "ml_m": None}}
    model._check_external_predictions(valid_pred)


@pytest.mark.ci
def test_exceptions_before_fit():
    """Test exception handling for confint() and p_adjust() methods when fit() hasn't been called."""
    dml_obj = dml.did.DoubleMLDIDMulti(**valid_arguments)

    msg = r"Apply fit\(\) before {}."
    with pytest.raises(ValueError, match=msg.format("confint")):
        dml_obj.confint()

    with pytest.raises(ValueError, match=msg.format("p_adjust")):
        dml_obj.p_adjust()

    with pytest.raises(ValueError, match=msg.format("bootstrap")):
        dml_obj.bootstrap()

    with pytest.raises(ValueError, match=msg.format("sensitivity_analysis")):
        dml_obj.sensitivity_analysis()

    with pytest.raises(ValueError, match=msg.format("sensitivity_plot")):
        dml_obj.sensitivity_plot()

    with pytest.raises(ValueError, match=msg.format("aggregate")):
        dml_obj.aggregate()

    msg = r"Apply sensitivity_analysis\(\) before sensitivity_summary."
    with pytest.raises(ValueError, match=msg):
        _ = dml_obj.sensitivity_summary


@pytest.mark.ci
def test_exceptions_sensitivity_benchmark():
    """Test exception handling for sensitivity_benchmark() method."""
    dml_obj = dml.did.DoubleMLDIDMulti(**valid_arguments)
    dml_obj.fit()

    # Test 1: sensitivity_elements is None
    with patch.object(dml_obj.__class__, "sensitivity_elements", property(lambda self: None)):
        msg = "Sensitivity analysis not yet implemented for"
        with pytest.raises(NotImplementedError, match=msg):
            dml_obj.sensitivity_benchmark(benchmarking_set=["Z1"])

    # Test 2: benchmarking_set is not a list
    invalid_types = [123, "string", {"dict": "value"}, (1, 2, 3)]
    for invalid_type in invalid_types:
        msg = "benchmarking_set must be a list."
        with pytest.raises(TypeError, match=msg):
            dml_obj.sensitivity_benchmark(benchmarking_set=invalid_type)

    # Test 3: benchmarking_set is an empty list
    msg = "benchmarking_set must not be empty."
    with pytest.raises(ValueError, match=msg):
        dml_obj.sensitivity_benchmark(benchmarking_set=[])

    # Test 4: benchmarking_set is not a subset of features
    msg = (
        r"benchmarking_set must be a subset of features \['Z1', 'Z2', 'Z3', 'Z4'\]. \['Z5', 'NonExistentFeature'\] was passed."
    )
    with pytest.raises(ValueError, match=msg):
        dml_obj.sensitivity_benchmark(benchmarking_set=["Z5", "NonExistentFeature"])

    # Test 5: fit_args is not None and not a dictionary
    invalid_types = [123, "string", ["list"], (1, 2, 3)]
    for invalid_type in invalid_types:
        msg = "fit_args must be a dict."
        with pytest.raises(TypeError, match=msg):
            dml_obj.sensitivity_benchmark(benchmarking_set=["Z1"], fit_args=invalid_type)
