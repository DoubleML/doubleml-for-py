import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.double_ml import DoubleML
from doubleml.plm.datasets import make_lplr_LZZ2020, make_plpr_CP2025
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

dml_data_lplr = make_lplr_LZZ2020(n_obs=100)

dml_lplr_obj = dml.DoubleMLLPLR(dml_data_lplr, LogisticRegression(), LinearRegression(), LinearRegression())

plpr_data = make_plpr_CP2025(num_id=100, dgp_type="dgp1")
dml_data_plpr = dml.DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)

dml_plpr_obj = dml.DoubleMLPLPR(dml_data_plpr, LinearRegression(), LinearRegression())


@pytest.mark.ci
def test_lplr_defaults():
    _check_basic_defaults_before_fit(dml_lplr_obj)

    _fit_bootstrap(dml_lplr_obj)

    _check_basic_defaults_after_fit(dml_lplr_obj)


@pytest.mark.ci
def test_plpr_defaults():
    _check_basic_defaults_before_fit(dml_plpr_obj)

    # manual fit and default check after fit
    dml_plpr_obj.fit()
    assert dml_plpr_obj.n_folds == 5
    assert dml_plpr_obj.n_rep == 1
    assert dml_plpr_obj.framework is not None

    # coefs and se
    assert isinstance(dml_plpr_obj.coef, np.ndarray)
    assert isinstance(dml_plpr_obj.se, np.ndarray)
    assert isinstance(dml_plpr_obj.all_coef, np.ndarray)
    assert isinstance(dml_plpr_obj.all_se, np.ndarray)
    assert isinstance(dml_plpr_obj.t_stat, np.ndarray)
    assert isinstance(dml_plpr_obj.pval, np.ndarray)

    # bootstrap and p_adjust method skipped

    # sensitivity
    assert dml_plpr_obj.sensitivity_params is None
    if dml_plpr_obj.sensitivity_params is not None:
        assert isinstance(dml_plpr_obj.sensitivity_elements, dict)

    # fit method
    if isinstance(dml_plpr_obj, DoubleML):
        assert dml_plpr_obj.predictions is not None
        assert dml_plpr_obj.models is None

    # confint method
    assert dml_plpr_obj.confint().equals(dml_plpr_obj.confint(joint=False, level=0.95))
