import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from doubleml.did import DoubleMLDIDMulti

df = dml.did.datasets.make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

dml_multi_obj = DoubleMLDIDMulti(dml_data, LinearRegression(), LogisticRegression(), [(1, 0, 1)])


def _assert_is_none(dml_obj):
    assert dml_obj.n_rep_boot is None
    assert dml_obj.boot_method is None
    assert dml_obj.framework is None
    assert dml_obj.sensitivity_params is None
    assert dml_obj.boot_t_stat is None


def _fit_bootstrap(dml_obj):
    dml_obj.fit()


@pytest.mark.ci
def test_plr_defaults():
    # _assert_is_none(dml_multi_obj)
    _fit_bootstrap(dml_multi_obj)
