"""Validate IRM scalar return types and reset behavior."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM

N_OBS = 200
N_FOLDS = 3
N_REP = 2
N_REP_BOOT = 314

np.random.seed(3141)
obj_dml_data = make_irm_data(theta=0.5, n_obs=N_OBS, dim_x=10, return_type="DoubleMLData")


@pytest.fixture(scope="module")
def fitted_dml_obj():
    np.random.seed(3141)
    dml_obj = IRM(obj_dml_data)
    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_coef_type_and_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.coef, np.ndarray)
    assert fitted_dml_obj.coef.shape == (1,)


@pytest.mark.ci
def test_se_type_and_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.se, np.ndarray)
    assert fitted_dml_obj.se.shape == (1,)


@pytest.mark.ci
def test_all_thetas_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_thetas, np.ndarray)
    assert fitted_dml_obj.all_thetas.shape == (1, N_REP)


@pytest.mark.ci
def test_all_coef_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_coef, np.ndarray)
    assert fitted_dml_obj.all_coef.shape == (1, N_REP)


@pytest.mark.ci
def test_all_ses_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_ses, np.ndarray)
    assert fitted_dml_obj.all_ses.shape == (1, N_REP)


@pytest.mark.ci
def test_summary_type(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.summary, pd.DataFrame)
    assert fitted_dml_obj.summary.shape[0] == 1


@pytest.mark.ci
def test_confint_type_and_shape(fitted_dml_obj):
    ci = fitted_dml_obj.confint()
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (1, 2)


@pytest.mark.ci
def test_confint_joint(fitted_dml_obj):
    ci_joint = fitted_dml_obj.confint(joint=True)
    assert isinstance(ci_joint, pd.DataFrame)
    assert ci_joint.shape == (1, 2)


@pytest.mark.ci
def test_psi_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.psi, np.ndarray)
    assert fitted_dml_obj.psi.shape == (N_OBS, 1, N_REP)


@pytest.mark.ci
def test_predictions_type(fitted_dml_obj):
    preds = fitted_dml_obj.predictions
    assert isinstance(preds, dict)
    assert "ml_g0" in preds
    assert "ml_g1" in preds
    assert "ml_m" in preds
    assert preds["ml_g0"].shape == (N_OBS, N_REP)
    assert preds["ml_g1"].shape == (N_OBS, N_REP)
    assert preds["ml_m"].shape == (N_OBS, N_REP)


@pytest.mark.ci
def test_smpls_type(fitted_dml_obj):
    smpls = fitted_dml_obj.smpls
    assert isinstance(smpls, list)
    assert len(smpls) == N_REP
    assert len(smpls[0]) == N_FOLDS


@pytest.mark.ci
def test_n_properties(fitted_dml_obj):
    assert fitted_dml_obj.n_obs == N_OBS
    assert fitted_dml_obj.n_folds == N_FOLDS
    assert fitted_dml_obj.n_rep == N_REP
    assert fitted_dml_obj.score == "ATE"


@pytest.mark.ci
def test_required_learners(fitted_dml_obj):
    assert fitted_dml_obj.required_learners == ["ml_g0", "ml_g1", "ml_m"]
    assert "ml_g0" in fitted_dml_obj.learners
    assert "ml_g1" in fitted_dml_obj.learners
    assert "ml_m" in fitted_dml_obj.learners


@pytest.mark.ci
def test_str_repr(fitted_dml_obj):
    assert isinstance(str(fitted_dml_obj), str)
    assert isinstance(repr(fitted_dml_obj), str)


@pytest.mark.ci
def test_get_params(fitted_dml_obj):
    params = fitted_dml_obj.get_params("ml_g0")
    assert isinstance(params, dict)
    assert "n_estimators" in params


@pytest.mark.ci
def test_set_params(fitted_dml_obj):
    result = fitted_dml_obj.set_params("ml_g0", n_estimators=5)
    assert result is fitted_dml_obj
    params = fitted_dml_obj.get_params("ml_g0")
    assert params["n_estimators"] == 5
    # Reset
    fitted_dml_obj.set_params("ml_g0", n_estimators=10)


@pytest.mark.ci
def test_get_params_invalid_learner(fitted_dml_obj):
    with pytest.raises(ValueError, match="not registered"):
        fitted_dml_obj.get_params("ml_invalid")


@pytest.mark.ci
def test_before_fit_raises():
    """Raise errors when accessing results before fitting."""
    np.random.seed(3141)
    dml_obj = IRM(obj_dml_data)
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef
    with pytest.raises(ValueError, match="Predictions not available. Call fit"):
        _ = dml_obj.predictions


@pytest.mark.ci
def test_irm_properties(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.normalize_ipw, bool)
    assert fitted_dml_obj.normalize_ipw is False
    assert isinstance(fitted_dml_obj.weights, dict)
    assert "weights" in fitted_dml_obj.weights
    assert fitted_dml_obj.ps_processor is not None
    assert fitted_dml_obj.ps_processor_config is not None


@pytest.mark.ci
def test_reset_after_set_learners():
    """Reset fitted state after updating learners."""
    np.random.seed(3141)
    dml_obj = IRM(obj_dml_data)
    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()

    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )

    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef
    with pytest.raises(ValueError, match="Predictions not available. Call fit"):
        _ = dml_obj.predictions


@pytest.mark.ci
def test_reset_after_draw_sample_splitting():
    """Reset fitted state after changing sample splits."""
    np.random.seed(3141)
    dml_obj = IRM(obj_dml_data)
    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()

    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)

    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef
    with pytest.raises(ValueError, match="Predictions not available. Call fit"):
        _ = dml_obj.predictions


@pytest.mark.ci
def test_sensitivity_elements_type_and_shape(fitted_dml_obj):
    """sensitivity_elements has correct keys, types, and shapes after fit."""
    elems = fitted_dml_obj.sensitivity_elements
    assert isinstance(elems, dict)
    for key in ["sigma2", "nu2"]:
        assert key in elems
        assert isinstance(elems[key], np.ndarray)
        assert elems[key].shape == (1, 1, N_REP)
    for key in ["psi_sigma2", "psi_nu2", "riesz_rep"]:
        assert key in elems
        assert isinstance(elems[key], np.ndarray)
        assert elems[key].shape == (N_OBS, 1, N_REP)


@pytest.mark.ci
def test_sensitivity_analysis_runs(fitted_dml_obj):
    """sensitivity_analysis() completes without error and returns self."""
    result = fitted_dml_obj.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    assert result is fitted_dml_obj.framework


@pytest.mark.ci
def test_sensitivity_before_fit_is_none():
    """sensitivity_elements returns None before fit()."""
    dml_obj = IRM(obj_dml_data)
    assert dml_obj.sensitivity_elements is None


@pytest.mark.ci
def test_sensitivity_reset_after_draw_sample_splitting():
    """sensitivity_elements resets to None after draw_sample_splitting()."""
    np.random.seed(3141)
    dml_obj = IRM(obj_dml_data)
    dml_obj.set_learners(
        ml_g=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
        ml_m=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    )
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    assert dml_obj.sensitivity_elements is not None
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    assert dml_obj.sensitivity_elements is None


@pytest.mark.ci
def test_sensitivity_params_structure(fitted_dml_obj):
    """sensitivity_params has expected keys and finite rv/rva after sensitivity_analysis()."""
    fitted_dml_obj.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    params = fitted_dml_obj.framework.sensitivity_params
    for key in ["theta", "se", "ci"]:
        assert "lower" in params[key] and "upper" in params[key]
    for key in ["rv", "rva"]:
        assert np.all(np.isfinite(params[key]))
        assert np.all(params[key] >= 0) and np.all(params[key] <= 1)


@pytest.mark.ci
def test_sensitivity_rho0_se_bounds(fitted_dml_obj):
    """With rho=0, se lower and upper bounds equal the unadjusted se."""
    fitted_dml_obj.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=0.0)
    params = fitted_dml_obj.framework.sensitivity_params
    np.testing.assert_allclose(params["se"]["lower"], fitted_dml_obj.se, rtol=1e-6)
    np.testing.assert_allclose(params["se"]["upper"], fitted_dml_obj.se, rtol=1e-6)


@pytest.mark.ci
def test_sensitivity_monotonicity_cf_y(fitted_dml_obj):
    """Increasing cf_y widens the theta sensitivity bounds."""
    fitted_dml_obj.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params_low = fitted_dml_obj.framework.sensitivity_params
    width_low = params_low["theta"]["upper"] - params_low["theta"]["lower"]
    fitted_dml_obj.sensitivity_analysis(cf_y=0.15, cf_d=0.03, rho=1.0)
    params_high = fitted_dml_obj.framework.sensitivity_params
    width_high = params_high["theta"]["upper"] - params_high["theta"]["lower"]
    assert np.all(width_high >= width_low)
