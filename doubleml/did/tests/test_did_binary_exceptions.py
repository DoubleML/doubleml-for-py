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
    msg = r"The never treated group is not allowed as treatment group \(g_value=0\)."
    with pytest.raises(ValueError, match=msg):
        invalid_arguments = {"g_value": 0}
        _ = dml.did.DoubleMLDIDBinary(**(valid_arguments | invalid_arguments))
