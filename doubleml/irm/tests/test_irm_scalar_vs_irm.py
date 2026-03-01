"""Compare IRM scalar against the existing DoubleMLIRM implementation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def comparison_fixture(score, n_rep):
    n_folds = 5
    seed = 3141

    np.random.seed(42)
    obj_dml_data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type="DoubleMLData")

    ml_g = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

    # Old IRM
    np.random.seed(seed)
    dml_old = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g,
        ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
    )
    dml_old.fit()

    # New IRM scalar — share sample splits from old model for exact comparison
    dml_new = IRM(obj_dml_data, score=score)
    dml_new.set_learners(ml_g=ml_g, ml_m=ml_m)
    # Copy sample splits directly to ensure identical cross-fitting structure
    dml_new._n_folds = n_folds
    dml_new._n_rep = n_rep
    dml_new._smpls = dml_old.smpls
    dml_new.fit()

    return {"old": dml_old, "new": dml_new}


@pytest.mark.ci
def test_coef_equal(comparison_fixture):
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.coef, old.coef, rtol=1e-9)


@pytest.mark.ci
def test_se_equal(comparison_fixture):
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.se, old.se, rtol=1e-9)


@pytest.mark.ci
def test_all_coef_equal(comparison_fixture):
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.all_thetas, old.all_coef, rtol=1e-9)


@pytest.mark.ci
def test_all_se_equal(comparison_fixture):
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.all_ses, old.all_se, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_sigma2_equal(comparison_fixture):
    """IRM scalar sigma2 matches DoubleMLIRM sensitivity_elements['sigma2']."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    # Legacy shape: (1, n_rep, 1); scalar shape: (1, 1, n_rep). Transpose to align.
    old_sigma2 = np.transpose(old.sensitivity_elements["sigma2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["sigma2"], old_sigma2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_nu2_equal(comparison_fixture):
    """IRM scalar nu2 matches DoubleMLIRM sensitivity_elements['nu2']."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    old_nu2 = np.transpose(old.sensitivity_elements["nu2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["nu2"], old_nu2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_max_bias_equal(comparison_fixture):
    """IRM scalar framework max_bias matches DoubleMLIRM framework max_bias."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(
        new.framework.sensitivity_elements["max_bias"],
        old.framework.sensitivity_elements["max_bias"],
        rtol=1e-9,
    )
