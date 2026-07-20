"""Compare PLRVector against the legacy DoubleMLPLR implementation for multi-treatment data."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression

import doubleml as dml
from doubleml.plm.plr_vector import PLRVector


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
def comparison_fixture(generate_data_bivariate, learner, score, n_rep):
    n_folds = 3
    seed = 3141
    data = generate_data_bivariate
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()
    d_cols = data.columns[data.columns.str.startswith("d")].tolist()

    obj_dml_data = dml.DoubleMLData(data, y_col="y", d_cols=d_cols, x_cols=x_cols)

    ml_g_arg = clone(learner) if score == "IV-type" else None

    # Legacy DoubleMLPLR draws splits in __init__
    np.random.seed(seed)
    dml_old = dml.DoubleMLPLR(
        obj_dml_data,
        clone(learner),
        clone(learner),
        ml_g_arg,
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
    )
    dml_old.fit()

    # New PLRVector draws splits explicitly via draw_sample_splitting
    np.random.seed(seed)
    dml_new = PLRVector(obj_dml_data, score=score)
    dml_new.set_learners(ml_l=clone(learner), ml_m=clone(learner), ml_g=ml_g_arg)
    dml_new.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_new.fit()

    return {"old": dml_old, "new": dml_new}


@pytest.mark.ci
def test_coef_equal(comparison_fixture):
    """PLRVector.coef matches legacy DoubleMLPLR.coef per treatment."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.coef, old.coef, rtol=1e-9)


@pytest.mark.ci
def test_se_equal(comparison_fixture):
    """PLRVector.se matches legacy DoubleMLPLR.se per treatment."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.se, old.se, rtol=1e-9)


@pytest.mark.ci
def test_all_coef_equal(comparison_fixture):
    """PLRVector.all_thetas matches legacy DoubleMLPLR.all_coef."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.all_thetas, old.all_coef, rtol=1e-9)


@pytest.mark.ci
def test_all_se_equal(comparison_fixture):
    """PLRVector.all_ses matches legacy DoubleMLPLR.all_se."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(new.all_ses, old.all_se, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_sigma2_equal(comparison_fixture):
    """PLRVector sigma2 matches legacy DoubleMLPLR sensitivity_elements['sigma2'] after axis swap."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    # Legacy shape: (1, n_rep, n_treat); vector shape: (1, n_treat, n_rep). Transpose to align.
    old_sigma2 = np.transpose(old.sensitivity_elements["sigma2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["sigma2"], old_sigma2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_nu2_equal(comparison_fixture):
    """PLRVector nu2 matches legacy DoubleMLPLR sensitivity_elements['nu2'] after axis swap."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    old_nu2 = np.transpose(old.sensitivity_elements["nu2"], (0, 2, 1))
    np.testing.assert_allclose(new.sensitivity_elements["nu2"], old_nu2, rtol=1e-9)


@pytest.mark.ci
def test_sensitivity_max_bias_equal(comparison_fixture):
    """PLRVector framework max_bias matches legacy DoubleMLPLR framework max_bias."""
    old = comparison_fixture["old"]
    new = comparison_fixture["new"]
    np.testing.assert_allclose(
        new.framework.sensitivity_elements["max_bias"],
        old.framework.sensitivity_elements["max_bias"],
        rtol=1e-9,
    )
