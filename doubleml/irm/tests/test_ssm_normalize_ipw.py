"""Check that DoubleMLSSM normalizes the inverse probability weights to mean one when normalize_ipw=True."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.irm.datasets import make_ssm_data


@pytest.mark.ci
def test_ssm_normalize_ipw_hajek_weights():
    # With normalize_ipw=True, the inverse probability weights d * s / (m * pi) and (1 - d) * s / ((1 - m) * pi)
    # are divided by their sample means, so that the normalized weights average to one in each treatment arm.
    np.random.seed(3141)
    dml_data = make_ssm_data(n_obs=500, dim_x=5)
    ssm = dml.DoubleMLSSM(dml_data, LinearRegression(), LogisticRegression(), LogisticRegression(), normalize_ipw=True)

    rng = np.random.default_rng(42)
    n_obs = 500
    d = rng.binomial(1, 0.4, size=n_obs)
    s = rng.binomial(1, 0.7, size=n_obs)
    y = rng.normal(size=n_obs)
    g_d1, g_d0 = rng.normal(size=(2, n_obs))
    pi = rng.uniform(0.2, 0.9, size=n_obs)
    m = rng.uniform(0.1, 0.9, size=n_obs)
    dtreat, dcontrol = d == 1, d == 0

    _, psi_b = ssm._score_elements(dtreat, dcontrol, g_d1, g_d0, pi, m, s, y)
    w_treat = dtreat * s / (m * pi)
    w_control = dcontrol * s / ((1 - m) * pi)
    psi_b_exp = (g_d1 + w_treat / np.mean(w_treat) * (y - g_d1)) - (g_d0 + w_control / np.mean(w_control) * (y - g_d0))
    np.testing.assert_allclose(psi_b, psi_b_exp, rtol=1e-12)

    # with y - g_d1 = 1 and y - g_d0 = 0, psi_b + 1 equals the normalized weights of the treated
    _, psi_b = ssm._score_elements(dtreat, dcontrol, y - 1.0, y, pi, m, s, y)
    np.testing.assert_allclose(np.mean(psi_b + 1.0), 1.0, rtol=1e-12)
    # with y - g_d1 = 0 and y - g_d0 = 1, 1 - psi_b equals the normalized weights of the controls
    _, psi_b = ssm._score_elements(dtreat, dcontrol, y, y - 1.0, pi, m, s, y)
    np.testing.assert_allclose(np.mean(1.0 - psi_b), 1.0, rtol=1e-12)
