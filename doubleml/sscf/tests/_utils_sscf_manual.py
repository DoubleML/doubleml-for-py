"""Manual (non object-oriented) implementation of the SSCF estimator.

The unit tests compare the output of :class:`doubleml.sscf.DoubleMLSSCF` against
this reference implementation, which follows Algorithm 1 of the paper line by line.
"""

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import train_test_split


def imr(propensity, trimming_threshold=1e-2):
    pi = np.clip(propensity, trimming_threshold, 1 - trimming_threshold)
    return norm.pdf(norm.ppf(pi)) / pi


def fit_sscf_manual(y, d, x, z, smpls, ml_pi, ml_g, ml_m, nested=True, trimming_threshold=1e-2):
    """Cross-fitted nuisances and score elements, computed by an explicit fold loop."""
    n_obs = y.shape[0]
    z_sel = x if z is None else np.column_stack((x, z))

    pi_hat = np.full(n_obs, np.nan)
    h_hat = np.full(n_obs, np.nan)
    g_hat = np.full(n_obs, np.nan)
    m_hat = np.full(n_obs, np.nan)

    for train_inds, test_inds in smpls:
        if nested:
            train_sel, train_out = train_test_split(train_inds, test_size=0.5, random_state=42, stratify=d[train_inds])
        else:
            train_sel, train_out = train_inds, train_inds

        pi_learner = clone(ml_pi).fit(z_sel[train_sel, :], d[train_sel])
        pi_fold = pi_learner.predict_proba(z_sel)[:, 1]
        h_fold = imr(pi_fold, trimming_threshold)
        pi_hat[test_inds] = np.clip(pi_fold[test_inds], trimming_threshold, 1 - trimming_threshold)
        h_hat[test_inds] = h_fold[test_inds]

        train_out_sel = train_out[d[train_out] == 1]
        g_learner = clone(ml_g).fit(x[train_out_sel, :], y[train_out_sel])
        g_hat[test_inds] = g_learner.predict(x[test_inds, :])

        m_learner = clone(ml_m).fit(x[train_out_sel, :], h_fold[train_out_sel])
        m_hat[test_inds] = m_learner.predict(x[test_inds, :])

    h_tilde = h_hat - m_hat
    psi_a = -d * h_hat * h_tilde
    psi_b = d * (y - g_hat) * h_tilde

    theta = -np.mean(psi_b) / np.mean(psi_a)
    psi = psi_a * theta + psi_b
    jac = np.mean(psi_a)
    se = np.sqrt(np.mean(psi**2) / jac**2 / n_obs)

    # equivalent representation on the selected subsample only
    sel = d == 1
    n_sel = int(sel.sum())
    theta_selected = np.sum((y[sel] - g_hat[sel]) * h_tilde[sel]) / np.sum(h_hat[sel] * h_tilde[sel])
    jac_selected = -np.mean(h_hat[sel] * h_tilde[sel])
    se_selected = np.sqrt(np.mean(psi[sel] ** 2) / jac_selected**2 / n_sel)

    score_null = psi_b[sel]
    stat = n_sel * np.mean(score_null) ** 2 / np.mean((score_null - np.mean(score_null)) ** 2)

    return {
        "theta": theta,
        "se": se,
        "theta_selected": theta_selected,
        "se_selected": se_selected,
        "psi_a": psi_a,
        "psi_b": psi_b,
        "pi_hat": pi_hat,
        "h_hat": h_hat,
        "g_hat": g_hat,
        "m_hat": m_hat,
        "score_stat": stat,
    }
