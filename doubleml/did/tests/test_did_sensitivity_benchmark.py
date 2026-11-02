import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021, make_did_cs_CS2021


def _panel_data(panel, x_cols):
    np.random.seed(3141)
    if panel:
        df = make_did_CS2021(n_obs=500, dgp_type=1, time_type="float")
    else:
        df = make_did_cs_CS2021(n_obs=1000, dgp_type=1, time_type="float", lambda_t=0.5)
    return dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols)


def _fit(model_class, dml_data, **kwargs):
    np.random.seed(42)
    model = model_class(dml_data, ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    return model.fit()


@pytest.mark.ci
@pytest.mark.parametrize("panel", [True, False])
def test_did_multi_sensitivity_benchmark_drops_covariates(panel):
    x_long = ["Z1", "Z2", "Z3", "Z4"]
    x_short = ["Z2", "Z3", "Z4"]
    kwargs = {"gt_combinations": "standard", "panel": panel}
    dml_long = _fit(dml.did.DoubleMLDIDMulti, _panel_data(panel, x_long), **kwargs)
    dml_short = _fit(dml.did.DoubleMLDIDMulti, _panel_data(panel, x_short), **kwargs)

    benchmark = dml_long.sensitivity_benchmark(benchmarking_set=["Z1"])
    np.testing.assert_allclose(benchmark["delta_theta"].values, dml_short.coef - dml_long.coef, rtol=1e-9, atol=1e-12)
    assert np.all(benchmark["delta_theta"].values != 0)


@pytest.mark.ci
@pytest.mark.parametrize("panel", [True, False])
def test_did_binary_sensitivity_benchmark_drops_covariates(panel):
    model_class = dml.did.DoubleMLDIDBinary if panel else dml.did.DoubleMLDIDCSBinary
    kwargs = {"g_value": 2.0, "t_value_pre": 1.0, "t_value_eval": 2.0}
    dml_long = _fit(model_class, _panel_data(panel, ["Z1", "Z2", "Z3", "Z4"]), **kwargs)
    dml_short = _fit(model_class, _panel_data(panel, ["Z2", "Z3", "Z4"]), **kwargs)

    benchmark = dml_long.sensitivity_benchmark(benchmarking_set=["Z1"])
    np.testing.assert_allclose(benchmark["delta_theta"].values, dml_short.coef - dml_long.coef, rtol=1e-9, atol=1e-12)
    assert np.all(benchmark["delta_theta"].values != 0)
