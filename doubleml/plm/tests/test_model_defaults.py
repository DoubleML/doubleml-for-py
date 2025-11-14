import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLLPLR
from doubleml.plm.datasets import make_lplr_LZZ2020
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

dml_data_lplr = make_lplr_LZZ2020(n_obs=100)

dml_lplr_obj = DoubleMLLPLR(dml_data_lplr, LogisticRegression(), LinearRegression(), LinearRegression())


@pytest.mark.ci
def test_lplr_defaults():
    _check_basic_defaults_before_fit(dml_lplr_obj)

    _fit_bootstrap(dml_lplr_obj)

    _check_basic_defaults_after_fit(dml_lplr_obj)


@pytest.mark.ci
def test_did_multi_str():
    # Test the string representation before fitting
    dml_str = str(dml_lplr_obj)

    # Check that all important sections are present
    assert "================== DoubleMLLPLR Object ==================" in dml_str
    assert "------------------ Data Summary      ------------------" in dml_str
    assert "------------------ Score & Algorithm ------------------" in dml_str
    assert "------------------ Machine Learner   ------------------" in dml_str
    assert "------------------ Resampling        ------------------" in dml_str
    assert "------------------ Fit Summary       ------------------" in dml_str

    # Check specific content before fitting
    assert "No. folds: 5" in dml_str
    assert "No. repeated sample splits: 1" in dml_str
    assert "Learner ml_M:" in dml_str
    assert "Learner ml_m:" in dml_str
    assert "Learner ml_t:" in dml_str

    # Fit the model
    dml_lplr_obj_fit = dml_lplr_obj.fit()
    dml_str_after_fit = str(dml_lplr_obj_fit)

    # Check that additional information is present after fitting
    assert "coef" in dml_str_after_fit
    assert "std err" in dml_str_after_fit
    assert "t" in dml_str_after_fit
    assert "P>|t|" in dml_str_after_fit
    assert "Out-of-sample Performance:" in dml_str_after_fit
