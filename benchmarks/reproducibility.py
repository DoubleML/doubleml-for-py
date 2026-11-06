"""Fixed-data calibration and cost benchmark for scalar PLR aggregation.

Run: python benchmarks/reproducibility.py --pairs 1000 --n-jobs 4
No assertion of universal calibration is made from a single fixed dataset.
"""

import argparse
import json
from copy import deepcopy
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.utils import fit_until_reproducible, reproducibility_check


def make_model():
    rng = np.random.RandomState(412)
    x = rng.normal(size=(81, 4))
    d = x[:, 0] + rng.normal(size=len(x))
    y = 0.5 * d + x[:, 1] + rng.normal(size=len(x))
    data = DoubleMLData.from_arrays(x, y, d)
    return DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=3, draw_sample_splitting=False)


def run_pair(model, pair, tolerance, max_reps):
    with threadpool_limits(limits=1):
        start = perf_counter()
        a, b = [fit_until_reproducible(model, tolerance, max_reps=max_reps, random_state=2 * pair + i) for i in range(2)]
        return {
            "exceeded": abs(a["mean"] - b["mean"]) > tolerance,
            "reps": [a["repetitions"], b["repetitions"]],
            "exhausted": [a["status"] == "budget_exhausted", b["status"] == "budget_exhausted"],
            "seconds": perf_counter() - start,
        }


def fixed_run(model, seed, repetitions, tolerance):
    with threadpool_limits(limits=1):
        worker = deepcopy(model)
        splitter = KFold(model.n_folds, shuffle=True, random_state=np.random.RandomState(seed))
        start = perf_counter()
        worker.set_sample_splitting([list(splitter.split(np.empty(model.n_obs))) for _ in range(repetitions)])
        worker.fit(store_predictions=False)
        report = reproducibility_check(worker.all_coef[0], tolerance)
        return {"mean": report["mean"], "mcse": report["mcse"], "seconds": perf_counter() - start}


def benchmark(pairs, fixed_runs, tolerance, max_reps, n_jobs):
    model = make_model()
    start = perf_counter()
    rows = Parallel(n_jobs=n_jobs)(delayed(run_pair)(model, i, tolerance, max_reps) for i in range(pairs))
    exceedances = binomtest(sum(r["exceeded"] for r in rows), pairs)
    interval = exceedances.proportion_ci(method="wilson")
    nuisance_fits_per_rep = model.n_folds * len(model.learner)
    result = {
        "pairs": pairs,
        "data_seed": 412,
        "tolerance": tolerance,
        "error_rate": 0.05,
        "min_reps": 20,
        "max_reps": max_reps,
        "exceedance_rate": exceedances.statistic,
        "exceedance_interval_95": [float(interval.low), float(interval.high)],
        "mean_repetitions": float(np.mean([r["reps"] for r in rows])),
        "budget_exhaustion_rate": float(np.mean([r["exhausted"] for r in rows])),
        "total_nuisance_fits": int(nuisance_fits_per_rep * np.sum([r["reps"] for r in rows])),
        "elapsed_seconds": perf_counter() - start,
        "mean_run_seconds": float(np.mean([r["seconds"] for r in rows]) / 2),
        "fixed": {},
    }
    for count in (10, 50, 100, 500):
        fixed = Parallel(n_jobs=n_jobs)(delayed(fixed_run)(model, i, count, tolerance) for i in range(fixed_runs))
        result["fixed"][str(count)] = {
            "runs": fixed_runs,
            "mean_mcse": float(np.mean([r["mcse"] for r in fixed])),
            "mean_run_seconds": float(np.mean([r["seconds"] for r in fixed])),
            "nuisance_fits_per_run": nuisance_fits_per_rep * count,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--fixed-runs", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-reps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.pairs < 1 or args.fixed_runs < 1:
        parser.error("pairs and fixed-runs must be positive")
    print(json.dumps(benchmark(args.pairs, args.fixed_runs, args.tolerance, args.max_reps, args.n_jobs), indent=2))
