import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLLPLR, DoubleMLPanelData, DoubleMLPLPR
from doubleml.plm.datasets import make_lplr_LZZ2020, make_plpr_CP2025
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

dml_data_lplr = make_lplr_LZZ2020(n_obs=100)

dml_lplr_obj = DoubleMLLPLR(dml_data_lplr, LogisticRegression(), LinearRegression(), LinearRegression())

plpr_data = make_plpr_CP2025(num_id=100)
dml_data_plpr = DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)

dml_plpr_obj = DoubleMLPLPR(dml_data_plpr, LinearRegression(), LinearRegression())


@pytest.mark.ci
def test_lplr_defaults():
    _check_basic_defaults_before_fit(dml_lplr_obj)

    _fit_bootstrap(dml_lplr_obj)

    _check_basic_defaults_after_fit(dml_lplr_obj)


@pytest.mark.ci
def test_plpr_defaults():
    _check_basic_defaults_before_fit(dml_plpr_obj)
    # TODO: fit for cluster?
