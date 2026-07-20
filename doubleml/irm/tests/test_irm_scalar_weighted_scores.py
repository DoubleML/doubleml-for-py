"""Test weighted score computation for IRM scalar, including dict weights with n_rep > 1.

With constant dict weights c * ones (weights = c, weights_bar = c):
  psi_a = -weights / mean(weights) = -c/c = -1  (same as unweighted)
  psi_b = c * psi_b_unweighted
  theta = -mean(psi_b) / mean(psi_a) = c * theta_unweighted
  se    = c * se_unweighted  (psi scales by c, psi_a unchanged)
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM

_N_FOLDS = 5
_N_OBS = 500
_DIM_X = 10
_WEIGHT_CONST = 0.5

ml_g = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
ml_m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    """Number of repetitions — covers single- and multi-rep cases."""
    return request.param


@pytest.fixture(scope="module")
def irm_data():
    """Shared IRM dataset."""
    np.random.seed(42)
    return make_irm_data(theta=0.5, n_obs=_N_OBS, dim_x=_DIM_X, return_type="DoubleMLData")


@pytest.fixture(scope="module")
def constant_weights_fixture(irm_data, n_rep):
    """Pair of IRM scalar models sharing sample splits: unweighted and constant-0.5-weighted.

    With weights = weights_bar = 0.5 * ones:
      theta_weighted = 0.5 * theta_unweighted
      se_weighted    = 0.5 * se_unweighted
    """
    n_obs = irm_data.n_obs
    const_weights = {
        "weights": np.full(n_obs, _WEIGHT_CONST),
        "weights_bar": np.full((n_obs, n_rep), _WEIGHT_CONST),
    }

    # Unweighted reference
    dml_ref = IRM(irm_data, score="ATE")
    dml_ref.set_learners(ml_g=clone(ml_g), ml_m=clone(ml_m))
    dml_ref.draw_sample_splitting(n_folds=_N_FOLDS, n_rep=n_rep)
    dml_ref.fit_nuisance_models()
    dml_ref.estimate_causal_parameters()

    # Constant-weighted — share exact sample splits for identical nuisance predictions
    dml_weighted = IRM(irm_data, score="ATE", weights=const_weights)
    dml_weighted.set_learners(ml_g=clone(ml_g), ml_m=clone(ml_m))
    dml_weighted._n_folds = _N_FOLDS
    dml_weighted._n_rep = n_rep
    dml_weighted._smpls = dml_ref.smpls
    dml_weighted.fit_nuisance_models()
    dml_weighted.estimate_causal_parameters()

    return {"ref": dml_ref, "weighted": dml_weighted}


@pytest.mark.ci
def test_dict_weights_n_rep_gt1_succeeds():
    """IRM scalar with weights_bar shape (n_obs, 3) and n_rep=3 fits without error."""
    np.random.seed(42)
    obj_dml_data = make_irm_data(theta=0.5, n_obs=200, dim_x=5, return_type="DoubleMLData")
    n_obs = obj_dml_data.n_obs
    n_rep = 3

    dict_weights = {
        "weights": np.full(n_obs, _WEIGHT_CONST),
        "weights_bar": np.full((n_obs, n_rep), _WEIGHT_CONST),
    }
    dml_obj = IRM(obj_dml_data, score="ATE", weights=dict_weights)
    dml_obj.set_learners(ml_g=clone(ml_g), ml_m=clone(ml_m))
    dml_obj.draw_sample_splitting(n_folds=3, n_rep=n_rep)
    dml_obj.fit_nuisance_models()
    dml_obj.estimate_causal_parameters()


@pytest.mark.ci
def test_constant_weights_coef(constant_weights_fixture):
    """theta (coef) with constant weights c equals c * theta_unweighted."""
    np.testing.assert_allclose(
        constant_weights_fixture["weighted"].coef,
        _WEIGHT_CONST * constant_weights_fixture["ref"].coef,
        rtol=1e-9,
    )


@pytest.mark.ci
def test_constant_weights_se(constant_weights_fixture):
    """se with constant weights c equals c * se_unweighted.

    psi_weighted = c * psi_unweighted, psi_a unchanged (-1), so
    se_weighted = sqrt(mean(c² * psi²)) / sqrt(n) = c * se_unweighted.
    """
    np.testing.assert_allclose(
        constant_weights_fixture["weighted"].se,
        _WEIGHT_CONST * constant_weights_fixture["ref"].se,
        rtol=1e-9,
    )
