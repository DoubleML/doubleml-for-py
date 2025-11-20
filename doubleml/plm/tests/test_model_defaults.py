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
