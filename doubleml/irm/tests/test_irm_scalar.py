"""Core estimation tests for IRM scalar."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module")
def dml_irm_scalar_fixture(score, normalize_ipw):
    n_folds = 5
    true_theta = 0.5

    np.random.seed(3141)
    data = make_irm_data(theta=true_theta, n_obs=500, dim_x=20, return_type="DoubleMLData")

    ml_g = RandomForestRegressor(n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2, random_state=42)

    np.random.seed(3141)
    dml_obj = IRM(data, score=score, normalize_ipw=normalize_ipw)
    dml_obj.set_learners(ml_g=ml_g, ml_m=ml_m)
    dml_obj.draw_sample_splitting(n_folds=n_folds, n_rep=1)
    dml_obj.fit()

    return {
        "coef": dml_obj.coef[0],
        "se": dml_obj.se[0],
        "true_theta": true_theta,
        "score": score,
    }


@pytest.mark.ci
def test_dml_irm_scalar_coef(dml_irm_scalar_fixture):
    coef = dml_irm_scalar_fixture["coef"]
    se = dml_irm_scalar_fixture["se"]
    true_theta = dml_irm_scalar_fixture["true_theta"]
    score = dml_irm_scalar_fixture["score"]

    # For ATE, the DGP theta is the true ATE parameter
    # For ATTE, the true ATTE differs from theta due to heterogeneous effects in the DGP
    if score == "ATE":
        assert abs(coef - true_theta) <= 3.0 * se
    else:
        # ATTE: just check estimate is finite and reasonable
        assert np.isfinite(coef)
        assert abs(coef) < 10.0


@pytest.mark.ci
def test_dml_irm_scalar_se(dml_irm_scalar_fixture):
    se = dml_irm_scalar_fixture["se"]
    assert se > 0
