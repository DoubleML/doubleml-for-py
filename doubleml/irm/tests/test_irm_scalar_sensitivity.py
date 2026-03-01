"""Score-parametrized sensitivity analysis tests for IRM scalar models."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM

N_OBS = 500
N_FOLDS = 5
N_REP = 2


@pytest.fixture(scope="module")
def irm_data():
    """Shared IRM dataset."""
    np.random.seed(3141)
    return make_irm_data(theta=0.5, n_obs=N_OBS, dim_x=5, return_type="DoubleMLData")


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def fitted_irm(request, irm_data):
    """Fitted IRM model parametrized over both score variants."""
    dml_obj = IRM(irm_data, score=request.param)
    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )
    dml_obj.fit(n_folds=N_FOLDS, n_rep=N_REP)
    return dml_obj


@pytest.mark.ci
def test_sensitivity_elements_positive(fitted_irm):
    """sigma2 >= 0, nu2 > 0, and max_bias >= 0 for each score variant."""
    elems = fitted_irm.sensitivity_elements
    assert np.all(elems["sigma2"] >= 0)
    assert np.all(elems["nu2"] > 0)
    assert np.all(fitted_irm.framework.sensitivity_elements["max_bias"] >= 0)


@pytest.mark.ci
def test_sensitivity_params_structure(fitted_irm):
    """After sensitivity_analysis(), theta/se/ci have lower/upper; rv/rva in [0,1]."""
    fitted_irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params = fitted_irm.framework.sensitivity_params
    for key in ["theta", "se", "ci"]:
        assert "lower" in params[key] and "upper" in params[key]
    for key in ["rv", "rva"]:
        assert np.all(np.isfinite(params[key]))
        assert np.all(params[key] >= 0) and np.all(params[key] <= 1)


@pytest.mark.ci
def test_sensitivity_params_bounds_ordered(fitted_irm):
    """theta lower bound <= estimated coef <= theta upper bound."""
    fitted_irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params = fitted_irm.framework.sensitivity_params
    assert np.all(params["theta"]["lower"] <= fitted_irm.coef)
    assert np.all(fitted_irm.coef <= params["theta"]["upper"])


@pytest.mark.ci
def test_sensitivity_rho0(fitted_irm):
    """With rho=0, se lower and upper bounds equal the unadjusted se."""
    fitted_irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=0.0)
    params = fitted_irm.framework.sensitivity_params
    np.testing.assert_allclose(params["se"]["lower"], fitted_irm.se, rtol=1e-6)
    np.testing.assert_allclose(params["se"]["upper"], fitted_irm.se, rtol=1e-6)


@pytest.mark.ci
def test_sensitivity_monotonicity_cf_y(fitted_irm):
    """Increasing cf_y produces wider theta sensitivity bounds."""
    fitted_irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    width_low = (
        fitted_irm.framework.sensitivity_params["theta"]["upper"] - fitted_irm.framework.sensitivity_params["theta"]["lower"]
    )
    fitted_irm.sensitivity_analysis(cf_y=0.15, cf_d=0.03, rho=1.0)
    width_high = (
        fitted_irm.framework.sensitivity_params["theta"]["upper"] - fitted_irm.framework.sensitivity_params["theta"]["lower"]
    )
    assert np.all(width_high >= width_low)
