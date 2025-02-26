import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

dml_data = dml.did.datasets.make_did_SZ2020(n_obs=500, dgp_type=1, return_type="DoubleMLPanelData")

df = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")

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
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))

    # g value
    msg = r"The value test is not in the set of treatment group values \[0 1\]."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": "test"}
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))

    msg = r"The never treated group is not allowed as treatment group \(g_value=0\)."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": 0}
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))

    # t values
    msg = r"The value test is not in the set of evaluation period values \[0 1\]."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"t_value_pre": "test"}
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"t_value_eval": "test"}
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))
