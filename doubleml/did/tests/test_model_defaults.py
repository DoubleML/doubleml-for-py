import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from doubleml.did import DoubleMLDIDMulti

from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

df = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

dml_multi_obj = DoubleMLDIDMulti(dml_data, LinearRegression(), LogisticRegression(), [(1, 0, 1)])


@pytest.mark.ci
def test_did_multi_defaults():
    _check_basic_defaults_before_fit(dml_multi_obj)
    _fit_bootstrap(dml_multi_obj)

