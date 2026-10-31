"""Check that DoubleMLSSM normalizes the inverse probability weights to mean one when normalize_ipw=True."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.irm.datasets import make_irm_data, make_ssm_data


@pytest.mark.ci
def test_ssm_normalize_ipw_hajek_weights():
    """With normalize_ipw=True, the inverse probability weights (including s and pi) average to one in each arm."""
    np.random.seed(3141)
    dml_data = make_ssm_data(n_obs=500, dim_x=5)
    ssm = dml.DoubleMLSSM(dml_data, LinearRegression(), LogisticRegression(), LogisticRegression(), normalize_ipw=True)

    rng = np.random.default_rng(42)
    n_obs = 500
    d = rng.binomial(1, 0.4, size=n_obs)
    s = rng.binomial(1, 0.7, size=n_obs)
    y = rng.normal(size=n_obs)
    pi = rng.uniform(0.2, 0.9, size=n_obs)
    m = rng.uniform(0.1, 0.9, size=n_obs)
    dtreat, dcontrol = d == 1, d == 0

    # with y - g_d1 = 1 and y - g_d0 = 0, psi_b + 1 equals the normalized weights of the treated
    _, psi_b = ssm._score_elements(dtreat, dcontrol, y - 1.0, y, pi, m, s, y)
    np.testing.assert_allclose(np.mean(psi_b + 1.0), 1.0, rtol=1e-12)
    # with y - g_d1 = 0 and y - g_d0 = 1, 1 - psi_b equals the normalized weights of the controls
    _, psi_b = ssm._score_elements(dtreat, dcontrol, y, y - 1.0, pi, m, s, y)
    np.testing.assert_allclose(np.mean(1.0 - psi_b), 1.0, rtol=1e-12)


@pytest.mark.ci
def test_ssm_normalize_ipw_matches_irm():
    """Without selection (s = 1, pi = 1), the normalized SSM score equals the normalized IRM ATE score."""
    np.random.seed(3141)
    n_obs = 500
    ssm = dml.DoubleMLSSM(
        make_ssm_data(n_obs=n_obs, dim_x=5), LinearRegression(), LogisticRegression(), LogisticRegression(), normalize_ipw=True
    )
    irm = dml.DoubleMLIRM(
        make_irm_data(n_obs=n_obs, dim_x=5), LinearRegression(), LogisticRegression(), score="ATE", normalize_ipw=True
    )

    rng = np.random.default_rng(42)
    d = rng.binomial(1, 0.4, size=n_obs)
    y = rng.normal(size=n_obs)
    g_d1, g_d0 = rng.normal(size=(2, n_obs))
    m = rng.uniform(0.1, 0.9, size=n_obs)
    ones = np.ones(n_obs)

    psi_a_ssm, psi_b_ssm = ssm._score_elements(d == 1, d == 0, g_d1, g_d0, ones, m, ones, y)
    psi_a_irm, psi_b_irm = irm._score_elements(y, d, g_d0, g_d1, m, smpls=None)

    np.testing.assert_allclose(psi_a_irm, psi_a_ssm, rtol=1e-12)
    np.testing.assert_allclose(psi_b_ssm, psi_b_irm, rtol=1e-12)
