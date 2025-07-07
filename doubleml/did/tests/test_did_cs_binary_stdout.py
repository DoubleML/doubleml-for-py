import io
from contextlib import redirect_stdout

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

dml_data = dml.did.datasets.make_did_SZ2020(n_obs=500, dgp_type=1, return_type="DoubleMLPanelData")


@pytest.mark.ci
def test_print_periods():
    """Test that print_periods parameter correctly controls output printing."""

    # Create test data
    dml_data = dml.did.datasets.make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")

    # Test 1: Default case (print_periods=False) - should not print anything
    f = io.StringIO()
    with redirect_stdout(f):
        _ = dml.did.DoubleMLDIDCSBinary(
            obj_dml_data=dml_data,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            g_value=1,
            t_value_pre=0,
            t_value_eval=1,
            print_periods=False,  # Default
        )
    output_default = f.getvalue()
    assert output_default.strip() == "", "Expected no output with print_periods=False"

    # Test 2: With print_periods=True - should print information
    f = io.StringIO()
    with redirect_stdout(f):
        _ = dml.did.DoubleMLDIDCSBinary(
            obj_dml_data=dml_data,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            g_value=1,
            t_value_pre=0,
            t_value_eval=1,
            print_periods=True,
        )
    output_print = f.getvalue()
    assert "Evaluation of ATT(1, 1), with pre-treatment period 0" in output_print
    assert "post-treatment: True" in output_print
    assert "Control group: never_treated" in output_print
