import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

df = dml.did.datasets.make_did_cs_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=2, n_periods=4, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

args = {
    "obj_dml_data": dml_data,
    "ml_g": LinearRegression(),
    "ml_m": LogisticRegression(),
    "g_value": 2,
    "t_value_pre": 0,
    "t_value_eval": 1,
    "score": "observational",
    "n_rep": 1,
}


@pytest.mark.ci
def test_control_groups_different():
    dml_did_never_treated = dml.did.DoubleMLDIDCSBinary(control_group="never_treated", **args)
    dml_did_not_yet_treated = dml.did.DoubleMLDIDCSBinary(control_group="not_yet_treated", **args)

    assert dml_did_never_treated.n_obs_subset != dml_did_not_yet_treated.n_obs_subset
    # same treatment group
    assert dml_did_never_treated.data_subset["G_indicator"].sum() == dml_did_not_yet_treated.data_subset["G_indicator"].sum()

    dml_did_never_treated.fit()
    dml_did_not_yet_treated.fit()

    assert dml_did_never_treated.coef != dml_did_not_yet_treated.coef
