"""Compare PLR against the existing DoubleMLPLR implementation."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LinearRegression

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR


@pytest.fixture(scope="module", params=[LinearRegression(), Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def comparison_fixture(learner, score, n_rep):
    n_folds = 5
    seed = 3141

    np.random.seed(42)
    obj_dml_data = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5)

    # Old PLR
    np.random.seed(seed)
    dml_old = dml.DoubleMLPLR(
        obj_dml_data,
        learner,
        learner,
        learner,
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
    )
    dml_old.fit()

    # New PLR
    np.random.seed(seed)
    dml_new = PLR(obj_dml_data, score=score)
    dml_new.set_learners(ml_l=learner, ml_m=learner, ml_g=learner)
    dml_new.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
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
    """PLR scalar sigma2 matches DoubleMLPLR sensitivity_elements['sigma2']."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    # Legacy shape: (1, n_rep, 1); scalar shape: (1, 1, n_rep). Transpose to align.
    old_sigma2 = np.transpose(old.sensitivity_elements["sigma2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["sigma2"], old_sigma2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_nu2_equal(comparison_fixture):
    """PLR scalar nu2 matches DoubleMLPLR sensitivity_elements['nu2']."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    old_nu2 = np.transpose(old.sensitivity_elements["nu2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["nu2"], old_nu2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_max_bias_equal(comparison_fixture):
    """PLR scalar framework max_bias matches DoubleMLPLR framework max_bias."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(
        new.framework.sensitivity_elements["max_bias"],
        old.framework.sensitivity_elements["max_bias"],
        rtol=1e-9,
    )
