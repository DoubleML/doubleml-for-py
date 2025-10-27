from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

dml_data = dml.did.datasets.make_did_SZ2020(n_obs=500, dgp_type=1, return_type="DoubleMLPanelData")

valid_arguments = {
    "obj_dml_data": dml_data,
    "ml_g": LinearRegression(),
    "ml_m": LogisticRegression(),
    "g_value": 1,
    "t_value_pre": 0,
    "t_value_eval": 1,
    "score": "observational",
    "n_rep": 1,
    "draw_sample_splitting": True,
}


@pytest.mark.ci
def test_input():
    # control group
    msg = r"The control group has to be one of \['never_treated', 'not_yet_treated'\]. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"control_group": 0}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    # g value
    msg = r"The value test is not in the set of treatment group values \[0 1\]."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": "test"}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    msg = r"The never treated group is not allowed as treatment group \(g_value=0\)."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": 0}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    msg = r"The never treated group is not allowed as treatment group \(g_value=0\)."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": 0.0}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    # t values
    msg = r"The value test is not in the set of evaluation period values \[0 1\]."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"t_value_pre": "test"}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"t_value_eval": "test"}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    # in-sample normalization
    msg = "in_sample_normalization indicator has to be boolean. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        invalid_arguments = {"in_sample_normalization": "test"}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))

    # ml_g classifier
    msg = r"The ml_g learner LogisticRegression\(\) was identified as"
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"ml_g": LogisticRegression()}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))


@pytest.mark.ci
def test_no_control_group_exception():
    msg = "No observations in the control group."
    with pytest.raises(ValueError, match=msg):
        invalid_data = dml.did.datasets.make_did_SZ2020(n_obs=500, dgp_type=1, return_type="DoubleMLPanelData")
        invalid_data.data["d"] = 1.0
        invalid_arguments = {"obj_dml_data": invalid_data, "control_group": "not_yet_treated"}
        _ = dml.did.DoubleMLDIDCSBinary(**(valid_arguments | invalid_arguments))


@pytest.mark.ci
def test_check_data_exceptions():
    """Test exception handling for _check_data method in DoubleMLDIDCSBinary"""
    df = pd.DataFrame(np.random.normal(size=(10, 5)), columns=[f"Col_{i}" for i in range(5)])

    # Test 1: Data has to be DoubleMLPanelData
    invalid_data_types = [
        dml.data.DoubleMLDIDData(df, y_col="Col_0", d_cols="Col_1"),
    ]

    for invalid_data in invalid_data_types:
        msg = r"For repeated outcomes the data must be of DoubleMLPanelData type\."
        with pytest.raises(TypeError, match=msg):
            _ = dml.did.DoubleMLDIDCSBinary(
                obj_dml_data=invalid_data,
                ml_g=LinearRegression(),
                ml_m=LogisticRegression(),
                g_value=1,
                t_value_pre=0,
                t_value_eval=1,
            )

    # Test 2: Data cannot have instrumental variables
    df_with_z = dml_data.data.copy()
    dml_data_with_z = dml.data.DoubleMLPanelData(
        df_with_z, y_col="y", d_cols="d", id_col="id", t_col="t", z_cols=["Z1"], x_cols=["Z2", "Z3", "Z4"]
    )

    msg = r"Incompatible data. Z1 have been set as instrumental variable\(s\)."
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.did.DoubleMLDIDCSBinary(
            obj_dml_data=dml_data_with_z,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            g_value=1,
            t_value_pre=0,
            t_value_eval=1,
        )

    # Test 3: Data must have exactly one treatment variable (using mock)
    with patch.object(dml_data.__class__, "n_treat", property(lambda self: 2)):
        msg = (
            "Incompatible data. To fit an DID model with DML exactly one variable needs to be specified as treatment variable."
        )
        with pytest.raises(ValueError, match=msg):
            _ = dml.did.DoubleMLDIDCSBinary(
                obj_dml_data=dml_data,
                ml_g=LinearRegression(),
                ml_m=LogisticRegression(),
                g_value=1,
                t_value_pre=0,
                t_value_eval=1,
            )


@pytest.mark.ci
def test_benchmark_warning():
    """Test warning when sensitivity_benchmark is called with experimental score"""
    args = {
        "obj_dml_data": dml_data,
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(),
        "g_value": 1,
        "t_value_pre": 0,
        "t_value_eval": 1,
        "n_rep": 1,
    }
    # Create a DID model with experimental score
    did_model = dml.did.DoubleMLDIDCSBinary(**args, score="experimental")
    did_model.fit()
    with pytest.warns(UserWarning, match="Sensitivity benchmarking for experimental score may not be meaningful"):
        did_model.sensitivity_benchmark(["Z1", "Z2"])
