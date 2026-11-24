"""Test the near-zero denominator check of the scalar LinearScoreMixin (vehicle: PLR scalar)."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 200


@pytest.fixture(scope="module")
def plr_with_smpls():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5, alpha=0.5)
    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=1)
    return dml_obj


@pytest.mark.ci
def test_linear_score_cancelling_psi_a_raises(plr_with_smpls):
    """A psi_a with mixed signs whose mean cancels to zero raises ValueError."""
    psi_a = np.tile([1.0, -1.0], N_OBS // 2).reshape(-1, 1)
    psi_b = np.random.normal(size=(N_OBS, 1))
    msg = r"Division by near-zero detected in linear score estimation"
    with pytest.raises(ValueError, match=msg):
        plr_with_smpls._est_causal_pars_and_se({"psi_a": psi_a, "psi_b": psi_b})


@pytest.mark.ci
def test_linear_score_scale_invariant(plr_with_smpls):
    """Rescaling psi_a by a tiny factor does not raise and rescales theta and se inversely."""
    np.random.seed(42)
    psi_a = -np.random.uniform(0.5, 1.5, size=(N_OBS, 1))
    psi_b = np.random.normal(size=(N_OBS, 1))
    scale = 1e-20

    plr_with_smpls._est_causal_pars_and_se({"psi_a": psi_a, "psi_b": psi_b})
    theta_ref, se_ref = plr_with_smpls._all_thetas.copy(), plr_with_smpls._all_ses.copy()

    plr_with_smpls._est_causal_pars_and_se({"psi_a": scale * psi_a, "psi_b": psi_b})
    np.testing.assert_allclose(plr_with_smpls._all_thetas * scale, theta_ref, rtol=1e-12)
    np.testing.assert_allclose(plr_with_smpls._all_ses * scale, se_ref, rtol=1e-12)
