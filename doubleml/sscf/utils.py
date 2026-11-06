"""Utilities for the sample selection control function model."""

import numpy as np
from scipy.stats import norm


def generalized_inverse_mills_ratio(propensity, eps=1e-12):
    r"""Inverse Mills ratio implied by a participation propensity score.

    Under joint normality of the error terms the inverse Mills ratio of the selected
    subpopulation is :math:`h_i = \varphi(Z_i'\beta)/\Phi(Z_i'\beta)`. Since
    :math:`\pi_i = P(D_i = 1 \mid Z_i) = \Phi(Z_i'\beta)`, the linear index can be
    recovered from the propensity score through the probit link,
    :math:`Z_i'\beta = \Phi^{-1}(\pi_i)`, which gives

    .. math::

        h_i = \frac{\varphi\big(\Phi^{-1}(\pi_i)\big)}{\pi_i}.

    This representation is link free: any consistent estimator of the participation
    propensity score (logit, probit, random forest, boosting, ...) can be mapped into
    the corresponding inverse Mills ratio, which is what makes the control function
    correction usable with generic machine learners.

    Parameters
    ----------
    propensity : array-like
        Estimated participation propensities :math:`\hat\pi_i \in (0, 1)`.

    eps : float
        Lower bound used to guard against division by zero. Default is ``1e-12``.
        Note that meaningful trimming has to be done before calling this function
        (strict overlap), ``eps`` only prevents numerical overflow.

    Returns
    -------
    h : numpy.ndarray
        Inverse Mills ratios.
    """
    pi = np.clip(np.asarray(propensity, dtype=float), eps, 1 - eps)
    return norm.pdf(norm.ppf(pi)) / pi


def selection_bias_score_test(y, d, g_hat, h_hat, m_hat):
    r"""Orthogonal score test of :math:`H_0: \theta_0 = 0` from nuisance predictions.

    Stand-alone version of :meth:`DoubleMLSSCF.selection_bias_test` which can be applied
    to externally computed (cross-fitted) nuisance predictions.

    Returns
    -------
    res : dict
        Dictionary with ``'statistic'`` and ``'p_value'``.
    """
    from scipy.stats import chi2

    selected = np.asarray(d) == 1
    psi_null = (np.asarray(y)[selected] - np.asarray(g_hat)[selected]) * (
        np.asarray(h_hat)[selected] - np.asarray(m_hat)[selected]
    )
    n_sel = psi_null.shape[0]
    score_mean = np.mean(psi_null)
    score_var = np.mean(np.square(psi_null - score_mean))
    statistic = n_sel * score_mean**2 / score_var
    return {"statistic": float(statistic), "p_value": float(chi2.sf(statistic, df=1)), "n_selected": n_sel}
