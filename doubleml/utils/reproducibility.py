"""Conditional Monte Carlo diagnostics for means across independent partitions."""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state


def _check_target(tolerance, error_rate):
    for name, value in (("tolerance", tolerance), ("error_rate", error_rate)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if error_rate >= 1:
        raise ValueError("error_rate must be less than 1.")
    threshold = tolerance / (np.sqrt(2) * norm.isf(error_rate / 2))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The requested target is outside floating-point precision.")
    return threshold


def reproducibility_check(values, tolerance, error_rate=0.05):
    """Assess split-induced Monte Carlo error of a scalar mean.

    Parameters
    ----------
    values : array-like of shape (n_rep,)
        At least two finite scalar estimates from independent, identically
        distributed partitions of the same data. For a fitted DoubleML model,
        use ``model.all_coef[index, :]`` for one prespecified coefficient.
        Independence cannot be checked from these values.
    tolerance : float
        Positive absolute tolerance for the difference between two independent
        mean aggregates, in the units of the coefficient.
    error_rate : float, default=0.05
        Nominal probability of exceeding the tolerance, strictly between 0 and 1.

    Returns
    -------
    report : dict
        ``mean``, ``mcse`` (sample standard deviation divided by sqrt(n_rep)),
        ``threshold`` (tolerance / (sqrt(2) * normal critical value)),
        ``criterion_met``, ``repetitions``, ``tolerance``, and ``error_rate``.

    Notes
    -----
    This is a plug-in diagnostic for the arithmetic mean, not DoubleML's
    median coefficient or its sampling standard error, confidence interval,
    or p-value. A satisfied criterion is not a finite-sample probability
    guarantee. In particular, zero observed variance need not imply zero
    conditional variance. Repeatedly inspecting this diagnostic alone does
    not implement a complete sequential procedure.

    The stopping rule in Algorithm 1 of Ritzwoller and Romano [1]_ uses this
    criterion after a prespecified burn-in. Its reproducibility interpretation
    is asymptotic as the tolerance decreases, conditional on the observed data.

    References
    ----------
    .. [1] Ritzwoller, D. M. and Romano, J. P. (2026), Reproducible Aggregation
       of Sample-Split Statistics. https://doi.org/10.1257/aer.20231651

    Examples
    --------
    >>> from doubleml.utils import reproducibility_check
    >>> report = reproducibility_check([0.10, 0.12, 0.11], tolerance=0.02)
    >>> report["criterion_met"]
    True
    """
    threshold = _check_target(tolerance, error_rate)
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2 or values.dtype.kind not in "fiu":
        raise ValueError("values must be a one-dimensional numeric array with at least two estimates.")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite estimates.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
        mcse = float(np.std(values, ddof=1) / np.sqrt(values.size))
    if not np.isfinite(mean) or not np.isfinite(mcse):
        raise ValueError("The estimates overflow the mean or variance calculation; rescale them.")
    return {
        "mean": mean,
        "mcse": mcse,
        "threshold": float(threshold),
        "criterion_met": bool(mcse <= threshold),
        "repetitions": values.size,
        "tolerance": float(tolerance),
        "error_rate": float(error_rate),
    }


def fit_until_reproducible(model, tolerance, error_rate=0.05, min_reps=20, max_reps=1000, random_state=None, n_jobs_cv=None):
    """Sequentially average one PLR coefficient over fresh K-fold partitions.

    Parameters
    ----------
    model : DoubleMLPLR
        Template with one treatment, independent observations, ``n_rep=1``,
        at least two folds, and a built-in score. Existing splits and fitted
        results are not reused. Fold-specific tuned parameters are unsupported;
        set common hyperparameters on the learners instead. All learner
        ``random_state`` parameters must be integers so learner randomness is
        fixed and only partition randomness is assessed. Custom learners must
        also be deterministic conditional on their inputs; this cannot be
        established automatically from their parameter names.
    tolerance : float
        Positive absolute tolerance for the scalar arithmetic mean.
    error_rate : float, default=0.05
        Nominal reproducibility error, strictly between zero and one.
    min_reps : int, default=20
        Prespecified burn-in, at least two. This is not a calibration guarantee.
    max_reps : int, default=1000
        Maximum number of complete repetitions, at least ``min_reps``.
    random_state : int, RandomState instance or None, default=None
        Random state used to generate successive independent shuffled K-fold
        partitions. An integer makes the partition sequence reproducible.
    n_jobs_cv : int or None, default=None
        Number of parallel jobs for fitting nuisance learners within each fold.

    Returns
    -------
    report : dict
        Fields from :func:`reproducibility_check`, plus ``status`` (either
        ``'target_reached'`` or ``'budget_exhausted'``) and ``values``, the
        per-repetition coefficient estimates. No sampling inference is attached
        to this mean. The input model and its existing inference are unchanged.

    Notes
    -----
    Fits a private copy once per complete partition and checks after every
    repetition following burn-in. Earlier repetitions are never refitted.
    Memory for nuisance estimates is bounded by one repetition. This first
    implementation deliberately excludes clustered data, multiple treatments,
    custom scores, and externally supplied predictions or partitions.

    The nominal error is asymptotic, not a finite-sample guarantee; see
    :func:`reproducibility_check`. Choose the tolerance before running the
    algorithm, preferably informed by an independent pilot. Small repetition
    counts can be poorly calibrated. Budget exhaustion does not certify the
    target. A diagnostic for the mean does not certify median-based inference.
    """
    from doubleml.plm.plr import DoubleMLPLR

    _check_target(tolerance, error_rate)
    for name, value in (("min_reps", min_reps), ("max_reps", max_reps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
    if min_reps < 2 or max_reps < min_reps:
        raise ValueError("Require 2 <= min_reps <= max_reps.")
    if not isinstance(model, DoubleMLPLR):
        raise TypeError("model must be a DoubleMLPLR instance.")
    if model._is_cluster_data or model._dml_data.n_treat != 1 or model.n_rep != 1 or model.n_folds < 2:
        raise ValueError("Require independent observations, one treatment, n_rep=1, and n_folds >= 2.")
    if not isinstance(model.score, str):
        raise ValueError("Only built-in PLR scores are supported.")
    if any(value is not None for params in model.params.values() for reps in params.values() for value in reps):
        raise ValueError("Fold-specific parameters are unsupported; configure the learners directly.")
    for learner in model.learner.values():
        for name, value in learner.get_params(deep=True).items():
            if name.split("__")[-1] == "random_state" and (isinstance(value, bool) or not isinstance(value, Integral)):
                raise ValueError("Set every learner random_state to an integer to isolate partition randomness.")
    rng = check_random_state(random_state)
    worker = deepcopy(model)
    splitter = KFold(n_splits=model.n_folds, shuffle=True, random_state=rng)
    values = []
    for _ in range(max_reps):
        worker.set_sample_splitting(list(splitter.split(np.empty(model._dml_data.n_obs))))
        worker.fit(n_jobs_cv=n_jobs_cv, store_predictions=False, store_models=False)
        value = float(worker.all_coef[0, 0])
        if not np.isfinite(value):
            raise ValueError("A repetition produced a non-finite coefficient.")
        values.append(value)
        if len(values) >= min_reps:
            report = reproducibility_check(values, tolerance, error_rate)
            if report["criterion_met"]:
                break
    report["status"] = "target_reached" if report["criterion_met"] else "budget_exhausted"
    report["values"] = np.array(values)
    return report
