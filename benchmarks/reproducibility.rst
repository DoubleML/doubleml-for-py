Reproducibility-controlled PLR aggregation
========================================

The diagnostic operates on one scalar statistic specified before inspecting
the repetitions. For an existing fitted model::

    from doubleml.utils import reproducibility_check

    report = reproducibility_check(model.all_coef[0], tolerance=0.01)

This reports the arithmetic mean and its split-induced Monte Carlo standard
error. It does not describe the precision of the median in ``model.coef``.
The input repetitions must be independent draws from the same partition law;
repeating one fixed partition does not establish reproducibility.

For automatic repetition, supply a single-treatment, nonclustered
``DoubleMLPLR`` template constructed with ``n_rep=1``::

    from doubleml.utils import fit_until_reproducible

    report = fit_until_reproducible(
        model, tolerance=0.01, error_rate=0.05,
        min_reps=20, max_reps=1000, random_state=42,
    )
    print(report["mean"], report["mcse"], report["status"])

The driver fits a private copy on fresh complete partitions. It stores only
the scalar estimates across repetitions. Existing coefficients, standard
errors, confidence intervals, and predictions on the template are preserved.
Common hyperparameters belong on the nuisance learners; fold-specific tuned
parameters, external predictions, custom scores, and clustered data are
outside this initial implementation. Set learner random states to integers.

The stopping criterion is MCSE <= tolerance / (sqrt(2) * z), where z is the
standard normal quantile at 1 - error_rate/2. It is checked after every complete
repetition following burn-in. ``budget_exhausted`` means the criterion was not
satisfied. Even ``target_reached`` reports a satisfied numerical criterion,
not an exact finite-sample probability guarantee. A constant observed sequence
can conceal rare outcomes, and small stopping counts can be poorly calibrated.

The nominal interpretation is asymptotic as the tolerance decreases, conditional
on the observed data. An independent pilot can inform the tolerance; the
minimum repetition count alone cannot establish calibration. No sampling
standard error or p-value is attached to the new mean.

Validation
----------

``reproducibility.py`` runs the entire adaptive driver independently twice per
pair on one fixed synthetic dataset. It reports exceedance frequency, a Wilson
95% Monte Carlo interval, budget exhaustion, actual repetition counts, nuisance
fit counts, and wall-clock time. Separate fixed-budget runs at 10, 50, 100,
and 500 repetitions measure MCSE and cost. They do not claim an empirical
exceedance rate from only 20 runs.

Run from the repository root::

    python benchmarks/reproducibility.py --pairs 1000 --fixed-runs 20 --n-jobs 3
    python benchmarks/reproducibility.py --pairs 1000 --fixed-runs 20 --tolerance 0.01 --n-jobs 3
    pytest doubleml/utils/tests/test_reproducibility.py

Runtime depends on concurrent work and hardware; fit counts are the more
portable comparison. This is a calibration example, not evidence of uniform
validity across datasets, nuisance learners, or treatment models.

Reference: Ritzwoller and Romano (2026),
https://doi.org/10.1257/aer.20231651, Algorithm 1.

Recorded calibration
--------------------

On the fixed dataset with seed 412, using 1,000 independent pairs for each
tolerance, nominal error 0.05, burn-in 20, and maximum 1,000 repetitions:

.. list-table:: Fixed-data calibration
   :header-rows: 1

   * - Tolerance
     - Mean repeats
     - Exceedance
     - 95% Monte Carlo CI
   * - 0.02
     - 29.953
     - 6.0%
     - 4.69%--7.65%
   * - 0.01
     - 122.261
     - 6.2%
     - 4.87%--7.87%

There was no budget exhaustion. Both point estimates exceed the nominal 5%;
both intervals include it. These runs do not establish exact calibration or
demonstrate a monotone improvement in empirical error with tighter tolerance.
The tolerances share the same seeds, so the two experiments are correlated.

Raw results, including timings and fixed-budget comparisons, are in
``results/plr-tolerance-0.02.json`` and ``results/plr-tolerance-0.01.json``.
The environment used Python 3.12.12, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1, and DoubleML based on upstream revision
``469daf260d4ea3520b8265acf17a9d29e800c5e0``. Three worker processes were used.

This pull request includes code written with the assistance of AI.
The code has not yet been reviewed by a human.
