import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.irm.datasets import make_iivm_data
from doubleml.utils._propensity_score import _normalize_ipw


@pytest.mark.ci
def test_iivm_normalize_ipw_uses_instrument():
    # The inverse probability weights in the IIVM score are z / m(x) and (1 - z) / (1 - m(x)),
    # with m(x) = P(Z = 1 | X), so normalize_ipw has to normalize them with the instrument z.
    np.random.seed(3141)
    data = make_iivm_data(n_obs=500, dim_x=5, theta=0.5, alpha_x=1.0, return_type="DataFrame")
    dml_data = dml.DoubleMLData(data, "y", "d", z_cols="z")

    def iivm(normalize_ipw):
        return dml.DoubleMLIIVM(
            dml_data, LinearRegression(), LogisticRegression(), LogisticRegression(), normalize_ipw=normalize_ipw
        )

    y, z, d = data["y"].values, data["z"].values, data["d"].values
    assert np.any(z != d)
    rng = np.random.default_rng(42)
    g_hat0, g_hat1 = rng.normal(size=(2, len(y)))
    r_hat0, r_hat1 = rng.uniform(0.1, 0.9, size=(2, len(y)))
    m_hat = rng.uniform(0.1, 0.9, size=len(y))

    psi_a, psi_b = iivm(True)._score_elements(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls=None)
    m_hat_adj = _normalize_ipw(m_hat, z)
    psi_a_exp, psi_b_exp = iivm(False)._score_elements(y, z, d, g_hat0, g_hat1, m_hat_adj, r_hat0, r_hat1, smpls=None)

    np.testing.assert_allclose(psi_a, psi_a_exp, rtol=1e-12)
    np.testing.assert_allclose(psi_b, psi_b_exp, rtol=1e-12)
    # the normalized weights average to one in both instrument arms
    np.testing.assert_allclose(np.mean(z / m_hat_adj), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.mean((1.0 - z) / (1.0 - m_hat_adj)), 1.0, rtol=1e-12)
