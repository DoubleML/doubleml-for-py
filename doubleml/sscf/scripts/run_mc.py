"""Time-budgeted, resumable driver for the Monte Carlo parts of the simulation study.

Each call computes as many replications as fit into a wall-clock budget and appends them
to the corresponding csv file, so the study can be run in several passes:

    python run_mc.py --stage sample_size --budget 240
    python run_mc.py --stage theta       --budget 240
    python run_mc.py --stage selection   --budget 240

Once all stages are complete, ``run_simulation.py`` picks up the cached csv files and only
builds the tables and figures.
"""

import argparse
import os
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

from run_simulation import DGP_DEFAULTS, one_replication

warnings.filterwarnings("ignore")


def job_list(stage, n_rep, n_rep_selection, n_obs_grid, theta_grid, n_obs_main):
    if stage == "sample_size":
        return [(seed, n_obs, DGP_DEFAULTS["theta"], False) for n_obs in n_obs_grid for seed in range(n_rep)]
    if stage == "theta":
        return [(10_000 + seed, n_obs_main, theta, False) for theta in theta_grid for seed in range(n_rep)]
    if stage == "selection":
        return [(20_000 + seed, n_obs_main, DGP_DEFAULTS["theta"], True) for seed in range(n_rep_selection)]
    raise ValueError(f"unknown stage {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["sample_size", "theta", "selection"])
    parser.add_argument("--budget", type=float, default=240.0, help="wall clock budget in seconds")
    parser.add_argument("--batch", type=int, default=4, help="replications computed per batch")
    parser.add_argument("--n-rep", type=int, default=15)
    parser.add_argument("--n-rep-selection", type=int, default=10)
    parser.add_argument("--n-obs", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--theta-grid", type=float, nargs="+", default=[0.0, -0.25, -0.5])
    parser.add_argument("--n-obs-main", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"mc_{args.stage}.csv")
    partial = os.path.join(args.out, f"mc_{args.stage}_partial.csv")

    jobs = job_list(
        args.stage, args.n_rep, args.n_rep_selection, args.n_obs, args.theta_grid, args.n_obs_main
    )

    done = pd.DataFrame()
    if os.path.exists(partial):
        done = pd.read_csv(partial)
    completed = set()
    if len(done) > 0:
        completed = {(int(r.seed), int(r.n_obs), float(r.theta_0)) for r in done.itertuples()}

    todo = [job for job in jobs if (job[0], job[1], job[2]) not in completed]
    print(f"stage {args.stage}: {len(jobs) - len(todo)}/{len(jobs)} replications already done", flush=True)

    start = time.time()
    while todo and (time.time() - start) < args.budget:
        batch, todo = todo[: args.batch], todo[args.batch :]
        results = Parallel(n_jobs=args.n_jobs)(delayed(one_replication)(*job) for job in batch)
        done = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
        done.to_csv(partial, index=False)
        print(
            f"  {len(done)}/{len(jobs)} done after {time.time() - start:.0f}s",
            flush=True,
        )

    if not todo:
        done.to_csv(path, index=False)
        print(f"stage {args.stage} complete -> {path}", flush=True)
    else:
        print(f"stage {args.stage} incomplete: {len(todo)} replications remaining", flush=True)


if __name__ == "__main__":
    main()
