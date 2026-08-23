r"""Adaptive kernel group lasso (KG-lasso) with sample selection bias correction.

This module implements the third step of the three-step procedure: post variable
selection in the outcome equation once the sample selection bias coefficient
:math:`\theta_0` has been estimated by the orthogonal DML score.

The estimator solves

.. math::

    \min_{b} \tfrac{1}{2}\big\|Y - \hat\theta \hat h - \mathbf{k} b\big\|^2
             + \lambda \sum_{j=1}^{J} w_j \|b_j\|_{\mathsf{K}_j},
    \qquad \|b_j\|_{\mathsf{K}_j} = (b_j' \mathsf{K}_j b_j)^{1/2},

on the selected subsample, with adaptive weights

.. math::

    w_j = \left(\frac{1 + |\hat\beta_j|}{\|\hat b_j^{\text{init}}\|_2}\right)^{\gamma},
    \qquad \gamma > 1 ,

which implement the data driven (soft) exclusion restriction: characteristics with a
strong signal in the selection equation are penalized more heavily in the outcome
equation, unless they also carry a strong outcome signal.
"""

import numpy as np
from scipy.optimize import brentq


def _spectral_positive_part(mat):
    """Spectral truncation: replace negative eigenvalues of a symmetric matrix by zero."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals) @ eigvecs.T


def _group_norm(vec, k_mat):
    """Kernel group norm :math:`\\|v\\|_{K} = (v' K v)^{1/2}`."""
    return float(np.sqrt(max(vec @ k_mat @ vec, 0.0)))


def _dual_group_norm(vec, k_mat_inv):
    """Dual norm :math:`\\|v\\|_{K,*} = (v' K^{-1} v)^{1/2}`."""
    return float(np.sqrt(max(vec @ k_mat_inv @ vec, 0.0)))


def _prox_kernel_group(v, k_mat, k_mat_inv, tau, method="exact", eig=None):
    r"""Proximal operator of :math:`\tau \|\cdot\|_{\mathsf{K}}` at ``v``.

    Both variants first apply the stationarity condition of the kernel group lasso,
    :math:`\hat b_j = 0` if :math:`\|v\|_{\mathsf{K}_j, *} \le \tau`.

    ``method='exact'`` (default) solves the proximal problem

    .. math::

        \hat b_j = \arg\min_b \tfrac{1}{2}\|b - v\|^2 + \tau \|b\|_{\mathsf{K}_j}

    exactly. Its first order condition is
    :math:`(\mathbf{I} + \tau \mathsf{K}_j / s)\hat b_j = v` with
    :math:`s = \|\hat b_j\|_{\mathsf{K}_j}`, so with the spectral decomposition
    :math:`\mathsf{K}_j = U \Lambda U'` and :math:`w = U'v` the solution is
    :math:`\hat b_j = U \operatorname{diag}((1 + \tau \lambda_\ell / s)^{-1}) w`,
    where :math:`s` solves the scalar fixed point equation
    :math:`s^2 = \sum_\ell \lambda_\ell w_\ell^2 (1 + \tau\lambda_\ell/s)^{-2}`.

    ``method='paper'`` uses the closed form of equation (31) of the paper,

    .. math::

        \hat b_j = \Big(\mathbf{I}_L - \tau \frac{\mathsf{K}_j}{\|v\|_{\mathsf{K}_j}}\Big)_+ v,

    with :math:`(\cdot)_+` denoting spectral truncation. The two variants coincide
    whenever :math:`\mathsf{K}_j \propto \mathbf{I}_L` (in particular they both reduce
    to the standard group lasso soft-thresholding operator
    :math:`(1 - \tau/\|v\|_2)_+ v`), but differ otherwise: the closed form replaces the
    norm of the solution by the norm of its argument and linearizes the resolvent
    :math:`(\mathbf{I} + \tau\mathsf{K}/s)^{-1} \approx \mathbf{I} - \tau\mathsf{K}/s`.
    Because of the linearization ``method='paper'`` is not guaranteed to decrease the
    objective, which is why the exact operator is the default.
    """
    if _dual_group_norm(v, k_mat_inv) <= tau:
        return np.zeros_like(v)

    if method == "paper":
        norm_v = _group_norm(v, k_mat)
        if norm_v <= 0.0:
            return np.zeros_like(v)
        shrink = _spectral_positive_part(np.eye(k_mat.shape[0]) - tau * k_mat / norm_v)
        return shrink @ v

    if method != "exact":
        raise ValueError("method must be either 'exact' or 'paper'.")

    if eig is None:
        eig = np.linalg.eigh(k_mat)
    eigvals, eigvecs = eig
    eigvals = np.clip(eigvals, 0.0, None)
    w = eigvecs.T @ v

    def _fixed_point(s):
        shrunk = w / (1.0 + tau * eigvals / s)
        return np.sqrt(max(np.sum(eigvals * shrunk**2), 0.0)) - s

    upper = _group_norm(v, k_mat)
    if upper <= 0.0:
        return np.zeros_like(v)
    lower = 1e-12 * upper
    if _fixed_point(lower) <= 0.0:  # numerically indistinguishable from zero
        return np.zeros_like(v)
    s_star = brentq(_fixed_point, lower, upper, xtol=1e-14, rtol=1e-12)
    return eigvecs @ (w / (1.0 + tau * eigvals / s_star))


class AdaptiveKernelGroupLasso:
    r"""Adaptive kernel group lasso estimator.

    Parameters
    ----------
    lam : float
        Penalty parameter :math:`\lambda`.

    weights : numpy.ndarray or None
        Adaptive group weights :math:`w_j` of length ``n_groups``. ``None`` corresponds
        to the non-adaptive KG-lasso (:math:`w_j = 1`).

    max_iter : int
        Maximum number of block coordinate descent sweeps. Default is ``500``.

    tol : float
        Relative tolerance for the objective. Default is ``1e-7``.

    fit_intercept : bool
        Whether an unpenalized intercept is included. Default is ``True``.

    prox : str
        Proximal operator of the kernel group norm, ``'exact'`` (default) or
        ``'paper'`` for the closed form of equation (31). See
        :func:`_prox_kernel_group`; the two coincide for
        :math:`\mathsf{K}_j \propto \mathbf{I}_L`.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Estimated coefficient vector :math:`\hat b`.

    intercept_ : float
        Estimated intercept.

    active_groups_ : numpy.ndarray
        Indices of the selected groups
        :math:`\hat{\mathcal{A}} = \{j : \|\hat b_j\|_{\mathsf{K}_j} > 0\}`.

    Notes
    -----
    The optimization is a block proximal gradient (block coordinate descent) algorithm.
    For each group the step size is the inverse of the block Lipschitz constant
    :math:`\Lambda_j = \lambda_{\max}(\mathbf{k}(j)'\mathbf{k}(j))`, so that the update
    coincides with the closed form solution of the paper whenever the blocks are
    orthonormalized (:math:`\mathbf{k}(j)'\mathbf{k}(j) = \mathbf{I}_L`, hence
    :math:`\Lambda_j = 1`), and remains valid otherwise. The objective is convex, and
    each sweep is monotonically decreasing.
    """

    def __init__(self, lam, weights=None, max_iter=500, tol=1e-7, fit_intercept=True, prox="exact"):
        self.lam = lam
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.prox = prox

    def _objective(self, y, k, b, k_mats, weights, group_slices):
        resid = y - k @ b
        pen = 0.0
        for j, sl in enumerate(group_slices):
            pen += weights[j] * _group_norm(b[sl], k_mats[j])
        return 0.5 * float(resid @ resid) + self.lam * pen

    def fit(self, k, y, groups, k_mats, b_init=None):
        """Fit the adaptive kernel group lasso.

        Parameters
        ----------
        k : numpy.ndarray
            Kernel design matrix of dimension ``(n_obs, n_features)``.
        y : numpy.ndarray
            Response vector (already corrected for the selection bias, if applicable).
        groups : numpy.ndarray
            Group index of length ``n_features``.
        k_mats : list of numpy.ndarray
            Kernel matrices :math:`\\mathsf{K}_j`, one per group.
        b_init : numpy.ndarray or None
            Warm start.
        """
        k = np.asarray(k, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        n_features = k.shape[1]
        group_labels = np.unique(groups)
        n_groups = group_labels.shape[0]

        if len(k_mats) != n_groups:
            raise ValueError("The number of kernel matrices must equal the number of groups.")
        weights = np.ones(n_groups) if self.weights is None else np.asarray(self.weights, dtype=float)
        if weights.shape[0] != n_groups:
            raise ValueError("The number of weights must equal the number of groups.")

        group_slices = [np.where(groups == g)[0] for g in group_labels]

        if self.fit_intercept:
            k_mean = k.mean(axis=0)
            y_mean = y.mean()
            kc = k - k_mean
            yc = y - y_mean
        else:
            k_mean = np.zeros(n_features)
            y_mean = 0.0
            kc, yc = k, y

        lipschitz = np.array([np.linalg.eigvalsh(kc[:, sl].T @ kc[:, sl]).max() for sl in group_slices])
        lipschitz = np.clip(lipschitz, 1e-12, None)
        k_mats_inv = [np.linalg.pinv(mat) for mat in k_mats]
        k_mats_eig = [np.linalg.eigh(mat) for mat in k_mats]

        b = np.zeros(n_features) if b_init is None else np.asarray(b_init, dtype=float).copy()
        resid = yc - kc @ b
        obj_old = self._objective(yc, kc, b, k_mats, weights, group_slices)
        self.n_iter_ = 0

        for it in range(self.max_iter):
            for j, sl in enumerate(group_slices):
                if not np.isfinite(weights[j]):
                    if np.any(b[sl] != 0.0):
                        resid = resid + kc[:, sl] @ b[sl]
                        b[sl] = 0.0
                    continue
                b_old_j = b[sl].copy()
                v = b_old_j + (kc[:, sl].T @ resid) / lipschitz[j]
                tau = self.lam * weights[j] / lipschitz[j]
                b_new_j = _prox_kernel_group(v, k_mats[j], k_mats_inv[j], tau, method=self.prox, eig=k_mats_eig[j])
                if not np.allclose(b_new_j, b_old_j):
                    resid = resid - kc[:, sl] @ (b_new_j - b_old_j)
                    b[sl] = b_new_j
            obj_new = self._objective(yc, kc, b, k_mats, weights, group_slices)
            self.n_iter_ = it + 1
            if abs(obj_old - obj_new) <= self.tol * max(abs(obj_old), 1.0):
                break
            obj_old = obj_new

        self.coef_ = b
        self.intercept_ = float(y_mean - k_mean @ b) if self.fit_intercept else 0.0
        self.objective_ = obj_new
        self.group_slices_ = group_slices
        self.group_norms_ = np.array([_group_norm(b[sl], k_mats[j]) for j, sl in enumerate(group_slices)])
        self.active_groups_ = group_labels[self.group_norms_ > 0]
        return self

    def predict(self, k):
        """Predicted values."""
        return self.intercept_ + np.asarray(k, dtype=float) @ self.coef_


def lambda_max(k, y, groups, k_mats, weights=None, fit_intercept=True):
    r"""Smallest penalty parameter for which all groups are zero.

    Given the stationarity conditions the solution is :math:`\hat b = 0` if and only if
    :math:`\|\mathbf{k}(j)'y\|_{\mathsf{K}_j,*} \le \lambda w_j` for all :math:`j`.
    """
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    group_labels = np.unique(groups)
    weights = np.ones(group_labels.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if fit_intercept:
        k = k - k.mean(axis=0)
        y = y - y.mean()
    lam = 0.0
    for j, g in enumerate(group_labels):
        if not np.isfinite(weights[j]) or weights[j] <= 0:
            continue
        sl = np.where(groups == g)[0]
        z_j = k[:, sl].T @ y
        lam = max(lam, _dual_group_norm(z_j, np.linalg.pinv(k_mats[j])) / weights[j])
    return lam


def kg_lasso_path(
    k,
    y,
    groups,
    k_mats,
    weights=None,
    n_lambda=50,
    lambda_min_ratio=0.01,
    gamma_ebic=0.5,
    max_iter=500,
    tol=1e-7,
    prox="exact",
):
    r"""Fit an adaptive KG-lasso path and select :math:`\lambda` by EBIC.

    The extended BIC of Chen and Chen (2008) is used with degrees of freedom given by
    the number of non-zero coefficients,

    .. math::

        \mathrm{EBIC}(\lambda) = n \log(\mathrm{RSS}_\lambda / n)
            + \mathrm{df}_\lambda \log(n) + 2 \gamma\, \mathrm{df}_\lambda \log(J).

    Returns
    -------
    res : dict
        Dictionary with the selected model (``'model'``), the penalty grid
        (``'lambdas'``), the EBIC values (``'ebic'``), the selected penalty
        (``'lambda'``) and the path of active sets (``'active_sets'``).
    """
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    n_obs = k.shape[0]
    n_groups = np.unique(groups).shape[0]

    lam_max = lambda_max(k, y, groups, k_mats, weights=weights)
    lambdas = np.exp(np.linspace(np.log(lam_max), np.log(lam_max * lambda_min_ratio), n_lambda))

    models, ebic, active_sets = [], [], []
    b_warm = None
    for lam in lambdas:
        model = AdaptiveKernelGroupLasso(lam=lam, weights=weights, max_iter=max_iter, tol=tol, prox=prox)
        model.fit(k, y, groups, k_mats, b_init=b_warm)
        b_warm = model.coef_.copy()
        resid = y - model.predict(k)
        rss = float(resid @ resid)
        df = int(np.sum(model.coef_ != 0.0)) + 1
        crit = n_obs * np.log(max(rss, 1e-12) / n_obs) + df * np.log(n_obs) + 2.0 * gamma_ebic * df * np.log(max(n_groups, 2))
        models.append(model)
        ebic.append(crit)
        active_sets.append(model.active_groups_)

    ebic = np.array(ebic)
    i_best = int(np.argmin(ebic))
    return {
        "model": models[i_best],
        "lambda": float(lambdas[i_best]),
        "lambdas": lambdas,
        "ebic": ebic,
        "active_sets": active_sets,
        "models": models,
        "index": i_best,
    }


def adaptive_weights(beta_hat, b_init, groups, gamma=2.0, large=1e8):
    r"""Adaptive weights :math:`w_j = ((1 + |\hat\beta_j|) / \|\hat b_j^{init}\|_2)^{\gamma}`.

    Parameters
    ----------
    beta_hat : numpy.ndarray
        Selection equation coefficients of the characteristics, length ``n_groups``.
        Only the coefficients of the characteristics :math:`X_j` are used; the
        coefficients of the excluded variables :math:`U` play no role here since these
        never enter the outcome equation.

    b_init : numpy.ndarray
        Initial (non-adaptive) group lasso estimate of the outcome coefficients.

    groups : numpy.ndarray
        Group index of length ``n_features``.

    gamma : float
        Adaptivity parameter :math:`\gamma > 1`. Default is ``2.0``.

    large : float
        Weight assigned to groups with a zero initial estimate. Default is ``1e8``.
    """
    groups = np.asarray(groups)
    group_labels = np.unique(groups)
    beta_hat = np.asarray(beta_hat, dtype=float)
    if beta_hat.shape[0] != group_labels.shape[0]:
        raise ValueError("beta_hat must have one entry per group (characteristic).")
    b_init = np.asarray(b_init, dtype=float)

    w = np.empty(group_labels.shape[0])
    for j, g in enumerate(group_labels):
        sl = np.where(groups == g)[0]
        norm_init = float(np.linalg.norm(b_init[sl]))
        if norm_init <= 0.0:
            w[j] = large
        else:
            w[j] = ((1.0 + abs(beta_hat[j])) / norm_init) ** gamma
    return w


def post_selection_refit(k, y, groups, active_groups):
    """Ordinary least squares refit on the columns of the selected groups."""
    groups = np.asarray(groups)
    cols = np.where(np.isin(groups, active_groups))[0]
    n_obs = k.shape[0]
    design = np.column_stack((np.ones(n_obs), k[:, cols]))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    b_full = np.zeros(k.shape[1])
    b_full[cols] = coef[1:]
    resid = y - design @ coef
    dof = max(n_obs - design.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    return {"intercept": float(coef[0]), "coef": b_full, "columns": cols, "sigma2": sigma2}


def variable_selection(
    y,
    k,
    groups,
    k_mats,
    beta_hat,
    theta_hat=0.0,
    imr=None,
    gamma=2.0,
    n_lambda=50,
    lambda_min_ratio=0.01,
    gamma_ebic=0.5,
):
    r"""Post variable selection after sample selection bias correction.

    Implements the third step: the response is corrected by the estimated bias term
    :math:`\hat\theta \hat h_i`, an initial KG-lasso provides the denominator of the
    adaptive weights, and the adaptive KG-lasso selects the active characteristics.

    Parameters
    ----------
    y, k, groups, k_mats
        Outcome, kernel design, group index and kernel matrices of the **selected**
        subsample (:math:`D_i = 1`).

    beta_hat : numpy.ndarray
        Selection equation coefficients of the characteristics (one per group).

    theta_hat : float
        Estimated sample selection bias coefficient. Setting ``theta_hat=0``
        reproduces the uncorrected benchmark (M2).

    imr : numpy.ndarray or None
        Estimated inverse Mills ratios of the selected subsample. Required whenever
        ``theta_hat`` is non-zero.

    Returns
    -------
    res : dict
        Dictionary with the corrected response (``'y_corrected'``), the initial fit
        (``'b_init'``), the adaptive weights (``'weights'``), the selected groups
        (``'active_groups'``), the adaptive KG-lasso path (``'path'``), the KG-lasso
        coefficients (``'coef'``) and the post-selection OLS refit (``'post'``).
    """
    y = np.asarray(y, dtype=float)
    if theta_hat != 0.0:
        if imr is None:
            raise ValueError("imr has to be provided whenever theta_hat is non-zero.")
        y_corrected = y - theta_hat * np.asarray(imr, dtype=float)
    else:
        y_corrected = y.copy()

    init_path = kg_lasso_path(
        k,
        y_corrected,
        groups,
        k_mats,
        weights=None,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        gamma_ebic=gamma_ebic,
    )
    b_init = init_path["model"].coef_

    w = adaptive_weights(beta_hat, b_init, groups, gamma=gamma)

    path = kg_lasso_path(
        k,
        y_corrected,
        groups,
        k_mats,
        weights=w,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        gamma_ebic=gamma_ebic,
    )
    model = path["model"]
    post = post_selection_refit(k, y_corrected, groups, model.active_groups_)

    return {
        "y_corrected": y_corrected,
        "b_init": b_init,
        "weights": w,
        "active_groups": model.active_groups_,
        "group_norms": model.group_norms_,
        "coef": model.coef_,
        "intercept": model.intercept_,
        "path": path,
        "init_path": init_path,
        "post": post,
    }


def selection_metrics(active_estimated, active_true, n_groups):
    """True/false positive rates and exact recovery of the group selection."""
    est = np.zeros(n_groups, dtype=bool)
    est[np.asarray(active_estimated, dtype=int)] = True
    true = np.zeros(n_groups, dtype=bool)
    true[np.asarray(active_true, dtype=int)] = True

    tp = int(np.sum(est & true))
    fp = int(np.sum(est & ~true))
    fn = int(np.sum(~est & true))
    tn = int(np.sum(~est & ~true))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "exact_recovery": bool(np.array_equal(est, true)),
        "n_selected": int(np.sum(est)),
    }
