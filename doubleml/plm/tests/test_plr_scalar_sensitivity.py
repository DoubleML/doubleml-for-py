"""Score-parametrized sensitivity analysis tests for PLR scalar models."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 500
N_FOLDS = 5
N_REP = 2


@pytest.fixture(scope="module")
def plr_data():
    """Shared PLR dataset."""
    np.random.seed(3141)
    return make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5)


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def fitted_plr(request, plr_data):
    """Fitted PLR model parametrized over both score variants."""
    dml_obj = PLR(plr_data, score=request.param)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.fit(n_folds=N_FOLDS, n_rep=N_REP)
    return dml_obj


@pytest.mark.ci
def test_sensitivity_elements_positive(fitted_plr):
    """sigma2 >= 0, nu2 > 0, and max_bias >= 0 for each score variant."""
    elems = fitted_plr.sensitivity_elements
    assert np.all(elems["sigma2"] >= 0)
    assert np.all(elems["nu2"] > 0)
    assert np.all(fitted_plr.framework.sensitivity_elements["max_bias"] >= 0)


@pytest.mark.ci
def test_sensitivity_params_structure(fitted_plr):
    """After sensitivity_analysis(), theta/se/ci have lower/upper; rv/rva in [0,1]."""
    fitted_plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params = fitted_plr.framework.sensitivity_params
    for key in ["theta", "se", "ci"]:
        assert "lower" in params[key] and "upper" in params[key]
    for key in ["rv", "rva"]:
        assert np.all(np.isfinite(params[key]))
        assert np.all(params[key] >= 0) and np.all(params[key] <= 1)


@pytest.mark.ci
def test_sensitivity_params_bounds_ordered(fitted_plr):
    """theta lower bound <= estimated coef <= theta upper bound."""
    fitted_plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params = fitted_plr.framework.sensitivity_params
    assert np.all(params["theta"]["lower"] <= fitted_plr.coef)
    assert np.all(fitted_plr.coef <= params["theta"]["upper"])


@pytest.mark.ci
def test_sensitivity_rho0(fitted_plr):
    """With rho=0, se lower and upper bounds equal the unadjusted se."""
    fitted_plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=0.0)
    params = fitted_plr.framework.sensitivity_params
    np.testing.assert_allclose(params["se"]["lower"], fitted_plr.se, rtol=1e-6)
    np.testing.assert_allclose(params["se"]["upper"], fitted_plr.se, rtol=1e-6)


@pytest.mark.ci
def test_sensitivity_monotonicity_cf_y(fitted_plr):
    """Increasing cf_y produces wider theta sensitivity bounds."""
    fitted_plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    width_low = (
        fitted_plr.framework.sensitivity_params["theta"]["upper"] - fitted_plr.framework.sensitivity_params["theta"]["lower"]
    )
    fitted_plr.sensitivity_analysis(cf_y=0.15, cf_d=0.03, rho=1.0)
    width_high = (
        fitted_plr.framework.sensitivity_params["theta"]["upper"] - fitted_plr.framework.sensitivity_params["theta"]["lower"]
    )
    assert np.all(width_high >= width_low)
