import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did import DoubleMLDIDBinary, DoubleMLDIDMulti
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

df_panel = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=2, n_periods=5, time_type="float")
dml_panel_data = dml.data.DoubleMLPanelData(
    df_panel, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)

dml_did_multi_obj = DoubleMLDIDMulti(dml_panel_data, LinearRegression(), LogisticRegression(), [(2, 0, 1)])
dml_did_binary_obj = DoubleMLDIDBinary(
    dml_panel_data, g_value=2, t_value_pre=0, t_value_eval=1, ml_g=LinearRegression(), ml_m=LogisticRegression()
)


@pytest.mark.ci
def test_did_binary_defaults():
    _check_basic_defaults_before_fit(dml_did_binary_obj)

    # specific parameters
    assert dml_did_binary_obj.control_group == "never_treated"
    assert dml_did_binary_obj.anticipation_periods == 0

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

    # specific parameters
    assert dml_did_binary_obj.control_group == "never_treated"
    assert dml_did_binary_obj.anticipation_periods == 0

    _fit_bootstrap(dml_did_multi_obj)
    _check_basic_defaults_after_fit(dml_did_multi_obj)


@pytest.mark.ci
def test_did_multi_str():
    # Test the string representation before fitting
    dml_str = str(dml_did_multi_obj)

    # Check that all important sections are present
    assert "================== DoubleMLDIDMulti Object ==================" in dml_str
    assert "------------------ Data summary      ------------------" in dml_str
    assert "------------------ Score & algorithm ------------------" in dml_str
    assert "------------------ Machine learner   ------------------" in dml_str
    assert "------------------ Resampling        ------------------" in dml_str
    assert "------------------ Fit summary       ------------------" in dml_str

    # Check specific content before fitting
    assert "Score function: observational" in dml_str
    assert "No. folds: 5" in dml_str
    assert "No. repeated sample splits: 1" in dml_str
    assert "Learner ml_g:" in dml_str
    assert "Learner ml_m:" in dml_str

    # Fit the model
    dml_did_multi_obj_fit = dml_did_multi_obj.fit()
    dml_str_after_fit = str(dml_did_multi_obj_fit)

    # Check that additional information is present after fitting
    assert "ATT(2,0,1)" in dml_str_after_fit
    assert "coef" in dml_str_after_fit
    assert "std err" in dml_str_after_fit
    assert "t" in dml_str_after_fit
    assert "P>|t|" in dml_str_after_fit
    assert "Out-of-sample Performance:" in dml_str_after_fit
