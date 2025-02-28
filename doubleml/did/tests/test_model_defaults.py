import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did import DoubleMLDIDBinary, DoubleMLDIDMulti
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

df_panel = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
dml_panel_data = dml.data.DoubleMLPanelData(
    df_panel, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)

dml_did_multi_obj = DoubleMLDIDMulti(dml_panel_data, LinearRegression(), LogisticRegression(), [(1, 0, 1)])
dml_did_binary_obj = DoubleMLDIDBinary(
    dml_panel_data, g_value=1, t_value_pre=0, t_value_eval=1, ml_g=LinearRegression(), ml_m=LogisticRegression()
)


@pytest.mark.ci
def test_did_binary_defaults():
    _check_basic_defaults_before_fit(dml_did_binary_obj)
    _fit_bootstrap(dml_did_binary_obj)
    _check_basic_defaults_after_fit(dml_did_binary_obj)


@pytest.mark.ci
def test_did_multi_defaults():
    _check_basic_defaults_before_fit(dml_did_multi_obj)

    # coefs and se
    assert dml_did_multi_obj.coef is None
    assert dml_did_multi_obj.se is None
    assert dml_did_multi_obj.all_coef is None
    assert dml_did_multi_obj.all_se is None
    assert dml_did_multi_obj.t_stat is None
    assert dml_did_multi_obj.pval is None

    _fit_bootstrap(dml_did_multi_obj)
