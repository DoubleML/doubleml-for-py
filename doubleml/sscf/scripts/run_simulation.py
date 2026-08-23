"""Simulation study for the high-dimensional sample selection control function model.

The script reproduces the four steps of the three-step estimation procedure:

* Step 1: simulate the "green silence" DGP (high-dimensional confounders, endogenous
  disclosure).
* Step 2: estimate the sample selection bias coefficient with the Neyman-orthogonal
  DML score and quantify the implied selection bias.
* Step 3: post variable selection with the adaptive kernel group lasso, with (M1) and
  without (M2) the bias correction.
* Step 4: write the tables (csv and LaTeX) and figures summarizing the results.

Usage
-----
    python run_simulation.py --n-rep 200 --out ../../results

All results are written to the output directory; nothing is printed except progress.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV, LogisticRegression

from doubleml_sscf import (
    DoubleMLSSCF,
    generalized_inverse_mills_ratio,
    make_green_silence_data,
    selection_metrics,
    variable_selection,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- setup

DGP_DEFAULTS = dict(
    dim_x=12,
    dim_z=8,
    n_deciles=6,
    theta=-0.5,
    sigma_epsilon=1.0,
    n_active_outcome=3,
    n_active_selection_x=2,
    n_active_selection_z=4,
    beta_x=0.4,
    beta_z=0.8,
    selection_rate=0.3,
    bandwidth=0.5,
    x_corr=0.5,
)


def make_learners():
    """Learners used throughout the study (sparse DGP, hence l1-penalized learners)."""
    ml_pi = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=5000)
    ml_g = LassoCV(n_alphas=20, max_iter=20000)
    ml_m = LassoCV(n_alphas=20, max_iter=20000)
    return ml_pi, ml_g, ml_m


def cached(path, builder):
    """Run `builder` unless `path` already holds the results (resumable study)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    df = builder()
    df.to_csv(path, index=False)
    return df


def latex_table(df, path, caption, label, float_format="%.4f"):
    """Write a booktabs LaTeX table."""
    body = df.to_latex(
        index=False,
        escape=False,
        float_format=lambda v: float_format % v,
        column_format="l" + "r" * (df.shape[1] - 1),
    )
    with open(path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        handle.write(body)
        handle.write("\\end{table}\n")


# ------------------------------------------------------- benchmark (non-orthogonal)


def naive_heckman(sim, n_folds=5, cross_fit=False):
    r"""Two-step plug-in benchmark without Neyman orthogonalization.

    The inverse Mills ratio is generated from an l1-penalized probit-type propensity
    score and the outcome equation is estimated by lasso on the selected sample; the
    coefficient of the correction term is obtained from the non-orthogonal score
    :math:`\psi = (Y - \hat g - \theta \hat h)\hat h`, i.e. without partialling out
    :math:`\mathbb{E}[h \mid X]`. This is the estimator which suffers from the
    regularization bias that the DML correction removes.
    """
    x = np.column_stack((sim["x"], sim["k"]))
    z_sel = np.column_stack((x, sim["u"]))
    y, d = sim["y"], sim["d"]
    ml_pi, ml_g, _ = make_learners()

    ml_pi.fit(z_sel, d)
    pi_hat = np.clip(ml_pi.predict_proba(z_sel)[:, 1], 1e-2, 1 - 1e-2)
    h_hat = generalized_inverse_mills_ratio(pi_hat)

    sel = d == 1
    ml_g.fit(x[sel], y[sel])
    g_hat = ml_g.predict(x)

    num = np.sum((y[sel] - g_hat[sel]) * h_hat[sel])
    den = np.sum(h_hat[sel] * h_hat[sel])
    theta_hat = num / den

    psi = (y[sel] - g_hat[sel] - theta_hat * h_hat[sel]) * h_hat[sel]
    jac = -np.mean(h_hat[sel] ** 2)
    se = np.sqrt(np.mean(psi**2) / jac**2 / sel.sum())
    return theta_hat, se, h_hat


# --------------------------------------------------------------- one MC replication


def one_replication(seed, n_obs, theta, run_selection=False):
    """One Monte Carlo replication; returns a dict of results."""
    warnings.filterwarnings("ignore")
    np.random.seed(seed)

    params = dict(DGP_DEFAULTS)
    params["theta"] = theta
    sim = make_green_silence_data(n_obs=n_obs, return_type="dict", **params)

    ml_pi, ml_g, ml_m = make_learners()
    dml = DoubleMLSSCF(sim["dml_data"], ml_pi, ml_g, ml_m, n_folds=5)
    dml.fit()
    ci = dml.confint().values[0]
    test = dml.selection_bias_test()

    theta_naive, se_naive, _ = naive_heckman(sim)

    out = {
        "seed": seed,
        "n_obs": n_obs,
        "theta_0": theta,
        "theta_dml": float(dml.coef[0]),
        "se_dml": float(dml.se[0]),
        "covered": bool(ci[0] <= theta <= ci[1]),
        "score_stat": test["statistic"],
        "reject_5pct": test["reject"],
        "theta_naive": theta_naive,
        "se_naive": se_naive,
        "n_selected": int(sim["d"].sum()),
    }

    if run_selection:
        out.update(selection_step(sim, dml))
    return out


# ------------------------------------------------- Step 3: post variable selection


def selection_step(sim, dml, gamma=2.0):
    """Adaptive KG-lasso variable selection with (M1) and without (M2) correction."""
    d = sim["d"]
    sel = d == 1
    k_sel = sim["k"][sel]
    y_sel = sim["y"][sel]
    h_sel = dml.imr[sel]
    theta_hat = float(dml.coef[0])
    groups = sim["groups"]
    k_mats = sim["K_mats"]
    n_groups = len(k_mats)

    # selection equation coefficients of the characteristics (adaptive weights)
    ml_pi, _, _ = make_learners()
    z_sel_design = np.column_stack((sim["x"], sim["k"], sim["u"]))
    ml_pi.fit(z_sel_design, d)
    beta_hat_x = np.abs(ml_pi.coef_.ravel()[: sim["x"].shape[1]])

    res_m1 = variable_selection(y_sel, k_sel, groups, k_mats, beta_hat_x, theta_hat=theta_hat, imr=h_sel, gamma=gamma)
    res_m2 = variable_selection(y_sel, k_sel, groups, k_mats, beta_hat_x, theta_hat=0.0, gamma=gamma)

    met_m1 = selection_metrics(res_m1["active_groups"], sim["active_outcome"], n_groups)
    met_m2 = selection_metrics(res_m2["active_groups"], sim["active_outcome"], n_groups)

    b_true = sim["b"]
    err_m1 = float(np.linalg.norm(res_m1["post"]["coef"] - b_true))
    err_m2 = float(np.linalg.norm(res_m2["post"]["coef"] - b_true))

    # out-of-sample imputation error for the non-disclosing firms
    k_out = sim["k"][~sel]
    truth_out = k_out @ b_true
    pred_m1 = res_m1["post"]["intercept"] + k_out @ res_m1["post"]["coef"]
    pred_m2 = res_m2["post"]["intercept"] + k_out @ res_m2["post"]["coef"]

    out = {
        "m1_tpr": met_m1["tpr"],
        "m1_fpr": met_m1["fpr"],
        "m1_f1": met_m1["f1"],
        "m1_exact": met_m1["exact_recovery"],
        "m1_n_selected": met_m1["n_selected"],
        "m1_l2_error": err_m1,
        "m1_imputation_bias": float(np.mean(pred_m1 - truth_out)),
        "m1_imputation_rmse": float(np.sqrt(np.mean((pred_m1 - truth_out) ** 2))),
        "m2_tpr": met_m2["tpr"],
        "m2_fpr": met_m2["fpr"],
        "m2_f1": met_m2["f1"],
        "m2_exact": met_m2["exact_recovery"],
        "m2_n_selected": met_m2["n_selected"],
        "m2_l2_error": err_m2,
        "m2_imputation_bias": float(np.mean(pred_m2 - truth_out)),
        "m2_imputation_rmse": float(np.sqrt(np.mean((pred_m2 - truth_out) ** 2))),
    }
    for j in range(n_groups):
        out[f"m1_sel_{j}"] = int(j in res_m1["active_groups"])
        out[f"m2_sel_{j}"] = int(j in res_m2["active_groups"])
    return out


# ----------------------------------------------------------------------- summaries


def summarize(df, group_cols):
    """Bias, RMSE, coverage and rejection frequencies by design cell."""
    rows = []
    for keys, sub in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        theta0 = sub["theta_0"].iloc[0]
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "Reps": len(sub),
                "n obs": int(sub["n_obs"].iloc[0]),
                "n selected": int(sub["n_selected"].mean()),
                "Bias (DML)": sub["theta_dml"].mean() - theta0,
                "SD (DML)": sub["theta_dml"].std(ddof=1),
                "RMSE (DML)": np.sqrt(np.mean((sub["theta_dml"] - theta0) ** 2)),
                "Mean SE": sub["se_dml"].mean(),
                "Coverage": sub["covered"].mean(),
                "Rejection": sub["reject_5pct"].mean(),
                "Bias (naive)": sub["theta_naive"].mean() - theta0,
                "RMSE (naive)": np.sqrt(np.mean((sub["theta_naive"] - theta0) ** 2)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rep", type=int, default=200, help="Monte Carlo replications per design cell")
    parser.add_argument("--n-rep-selection", type=int, default=100, help="replications for the variable selection study")
    parser.add_argument("--n-obs", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--theta-grid", type=float, nargs="+", default=[0.0, -0.25, -0.5])
    parser.add_argument("--n-obs-main", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    # ---------------------------------------------------------------- Step 1: DGP
    print("Step 1: data generating process", flush=True)
    np.random.seed(20240101)
    sim = make_green_silence_data(n_obs=args.n_obs_main, return_type="dict", **DGP_DEFAULTS)

    dgp_table = pd.DataFrame(
        {
            "Parameter": [
                "Firms $N$",
                "Characteristics $J$",
                "Excluded variables $\\dim(U)$",
                "Portfolios $L$",
                "Kernel features $JL$",
                "Active outcome groups $|\\mathcal{A}_b|$",
                "Active selection variables $|\\mathcal{A}_\\beta|$",
                "Selection bias coefficient $\\theta_0$",
                "Outcome error s.d. $\\sigma_\\varepsilon$",
                "Error correlation $\\rho$",
                "Kernel bandwidth $\\sigma_k$",
                "Characteristic correlation",
                "Target disclosure rate",
                "Realized disclosure rate",
                "Signal variance $\\mathrm{Var}(\\mathbf{k}b_0)$",
            ],
            "Value": [
                args.n_obs_main,
                DGP_DEFAULTS["dim_x"],
                DGP_DEFAULTS["dim_z"],
                DGP_DEFAULTS["n_deciles"],
                DGP_DEFAULTS["dim_x"] * DGP_DEFAULTS["n_deciles"],
                DGP_DEFAULTS["n_active_outcome"],
                DGP_DEFAULTS["n_active_selection_x"] + DGP_DEFAULTS["n_active_selection_z"],
                DGP_DEFAULTS["theta"],
                DGP_DEFAULTS["sigma_epsilon"],
                DGP_DEFAULTS["theta"] / DGP_DEFAULTS["sigma_epsilon"],
                DGP_DEFAULTS["bandwidth"],
                DGP_DEFAULTS["x_corr"],
                DGP_DEFAULTS["selection_rate"],
                round(float(sim["d"].mean()), 4),
                round(float((sim["k"] @ sim["b"]).var()), 4),
            ],
        }
    )
    dgp_table.to_csv(os.path.join(args.out, "tables", "table1_dgp.csv"), index=False)
    latex_table(
        dgp_table,
        os.path.join(args.out, "tables", "table1_dgp.tex"),
        "Design of the simulated green silence data generating process.",
        "tab:sim_dgp",
        float_format="%.3f",
    )

    # ------------------------------------------- Step 2: estimation on one dataset
    print("Step 2: estimation of the selection bias coefficient", flush=True)
    ml_pi, ml_g, ml_m = make_learners()
    dml = DoubleMLSSCF(sim["dml_data"], ml_pi, ml_g, ml_m, n_folds=5, n_rep=5)
    dml.fit()
    test = dml.selection_bias_test()
    bias = dml.bias_quantification()

    theta_naive, se_naive, h_naive = naive_heckman(sim)
    true_bias_selected = DGP_DEFAULTS["theta"] * sim["imr"][sim["d"] == 1].mean()
    true_bias_non_selected = DGP_DEFAULTS["theta"] * sim["imr"][sim["d"] == 0].mean()

    est_table = pd.DataFrame(
        {
            "Quantity": [
                "$\\theta_0$ (true)",
                "$\\hat\\theta$ (orthogonal DML)",
                "Standard error",
                "95\\% CI lower",
                "95\\% CI upper",
                "Score test $S_n$",
                "$p$-value",
                "$\\hat\\theta$ (naive two-step)",
                "Standard error (naive)",
                "$\\mathbb{E}_n[\\hat h \\mid D=1]$",
                "$\\mathbb{E}_n[h_0 \\mid D=1]$ (true)",
                "$\\mathbb{E}_n[\\hat h \\mid D=0]$",
                "$\\mathbb{E}_n[h_0 \\mid D=0]$ (true)",
                "Estimated bias, disclosers",
                "True bias, disclosers",
                "Estimated bias, non-disclosers",
                "True bias, non-disclosers",
            ],
            "Value": [
                DGP_DEFAULTS["theta"],
                bias["summary"]["theta"],
                bias["summary"]["se"],
                float(dml.confint().values[0][0]),
                float(dml.confint().values[0][1]),
                test["statistic"],
                test["p_value"],
                theta_naive,
                se_naive,
                bias["summary"]["mean_imr_selected"],
                float(sim["imr"][sim["d"] == 1].mean()),
                bias["summary"]["mean_imr_non_selected"],
                float(sim["imr"][sim["d"] == 0].mean()),
                bias["summary"]["mean_bias_selected"],
                float(true_bias_selected),
                bias["summary"]["mean_bias_non_selected"],
                float(true_bias_non_selected),
            ],
        }
    )
    est_table.to_csv(os.path.join(args.out, "tables", "table2_estimation.csv"), index=False)
    latex_table(
        est_table,
        os.path.join(args.out, "tables", "table2_estimation.tex"),
        "Estimation of the sample selection bias coefficient on a single simulated sample "
        "($N=2{,}000$, 5 folds, 5 repetitions of the sample splitting).",
        "tab:sim_estimation",
    )

    # -------------------------------------------------------- Monte Carlo studies
    print(f"Monte Carlo over sample sizes ({args.n_rep} replications per cell)", flush=True)
    def _run_sample_size():
        jobs = []
        for n_obs in args.n_obs:
            jobs += [(seed, n_obs, DGP_DEFAULTS["theta"], False) for seed in range(args.n_rep)]
        return pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(one_replication)(*job) for job in jobs))

    df_n = cached(os.path.join(args.out, "mc_sample_size.csv"), _run_sample_size)

    print(f"Monte Carlo over theta ({args.n_rep} replications per cell)", flush=True)
    def _run_theta():
        jobs = []
        for theta in args.theta_grid:
            jobs += [(10_000 + seed, args.n_obs_main, theta, False) for seed in range(args.n_rep)]
        return pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(one_replication)(*job) for job in jobs))

    df_t = cached(os.path.join(args.out, "mc_theta.csv"), _run_theta)

    tab_n = summarize(df_n, ["n_obs"]).rename(columns={"n_obs": "$N$"}).drop(columns=["n obs"])
    tab_t = summarize(df_t, ["theta_0"]).rename(columns={"theta_0": "$\\theta_0$"})

    tab_n.to_csv(os.path.join(args.out, "tables", "table3_sample_size.csv"), index=False)
    latex_table(
        tab_n,
        os.path.join(args.out, "tables", "table3_sample_size.tex"),
        "Monte Carlo performance of the orthogonal DML estimator of $\\theta_0$ across sample sizes. "
        "Rejection is the frequency of rejecting $H_0:\\theta_0=0$ with the orthogonal score test at the 5\\% level.",
        "tab:sim_sample_size",
    )
    tab_t.to_csv(os.path.join(args.out, "tables", "table4_theta.csv"), index=False)
    latex_table(
        tab_t,
        os.path.join(args.out, "tables", "table4_theta.tex"),
        "Monte Carlo performance across values of the sample selection bias coefficient "
        "($N=2{,}000$). The row $\\theta_0=0$ reports the empirical size of the score test, "
        "the remaining rows its power.",
        "tab:sim_theta",
    )

    # ------------------------------------------------ Step 3: variable selection
    print(f"Step 3: post variable selection ({args.n_rep_selection} replications)", flush=True)
    def _run_selection():
        jobs = [(20_000 + seed, args.n_obs_main, DGP_DEFAULTS["theta"], True) for seed in range(args.n_rep_selection)]
        return pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(one_replication)(*job) for job in jobs))

    df_s = cached(os.path.join(args.out, "mc_selection.csv"), _run_selection)

    sel_table = pd.DataFrame(
        {
            "Statistic": [
                "True positive rate",
                "False positive rate",
                "$F_1$",
                "Exact recovery",
                "Groups selected",
                "$\\|\\hat b - b_0\\|_2$",
                "Imputation bias (non-disclosers)",
                "Imputation RMSE (non-disclosers)",
            ],
            "M1 (corrected)": [
                df_s["m1_tpr"].mean(),
                df_s["m1_fpr"].mean(),
                df_s["m1_f1"].mean(),
                df_s["m1_exact"].mean(),
                df_s["m1_n_selected"].mean(),
                df_s["m1_l2_error"].mean(),
                df_s["m1_imputation_bias"].mean(),
                df_s["m1_imputation_rmse"].mean(),
            ],
            "M2 (uncorrected)": [
                df_s["m2_tpr"].mean(),
                df_s["m2_fpr"].mean(),
                df_s["m2_f1"].mean(),
                df_s["m2_exact"].mean(),
                df_s["m2_n_selected"].mean(),
                df_s["m2_l2_error"].mean(),
                df_s["m2_imputation_bias"].mean(),
                df_s["m2_imputation_rmse"].mean(),
            ],
        }
    )
    sel_table.to_csv(os.path.join(args.out, "tables", "table5_selection.csv"), index=False)
    latex_table(
        sel_table,
        os.path.join(args.out, "tables", "table5_selection.tex"),
        "Post variable selection with the adaptive kernel group lasso, with (M1) and without (M2) "
        "the sample selection bias correction ($N=4{,}000$). Imputation statistics refer to the "
        "predicted outcomes of the non-disclosing firms.",
        "tab:sim_selection",
    )

    n_groups = DGP_DEFAULTS["dim_x"]
    freq_table = pd.DataFrame(
        {
            "Characteristic": [f"$X_{{{j + 1}}}$" for j in range(n_groups)],
            "Active in outcome": [int(j in sim["active_outcome"]) for j in range(n_groups)],
            "Active in selection": [int(j in sim["active_selection_x"]) for j in range(n_groups)],
            "M1 selection freq.": [df_s[f"m1_sel_{j}"].mean() for j in range(n_groups)],
            "M2 selection freq.": [df_s[f"m2_sel_{j}"].mean() for j in range(n_groups)],
        }
    )
    freq_table.to_csv(os.path.join(args.out, "tables", "table6_frequencies.csv"), index=False)
    latex_table(
        freq_table,
        os.path.join(args.out, "tables", "table6_frequencies.tex"),
        "Selection frequencies of the characteristic-sorted portfolio groups across Monte Carlo "
        "replications, with (M1) and without (M2) the bias correction.",
        "tab:sim_frequencies",
        float_format="%.2f",
    )

    # ------------------------------------------------------------ Step 4: figures
    print("Step 4: figures", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta0 = DGP_DEFAULTS["theta"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sub = df_n[df_n["n_obs"] == args.n_obs_main]
    grid = np.linspace(
        min(sub["theta_dml"].min(), sub["theta_naive"].min()) - 0.05,
        max(sub["theta_dml"].max(), sub["theta_naive"].max()) + 0.05,
        200,
    )
    for column, label, style in [("theta_dml", "orthogonal DML", "-"), ("theta_naive", "naive two-step", "--")]:
        values = sub[column].values
        bandwidth = 1.06 * values.std(ddof=1) * len(values) ** (-1 / 5)
        density = np.mean(np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2), axis=1) / (
            bandwidth * np.sqrt(2 * np.pi)
        )
        ax.plot(grid, density, style, label=label)
    ax.axvline(theta0, color="black", linewidth=1, label=r"$\theta_0$")
    ax.set_xlabel(r"$\hat\theta$")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title(rf"Sampling distribution of $\hat\theta$ ($N={args.n_obs_main}$)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig1_theta_density.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    stats = tab_n.copy()
    axes[0].plot(stats["$N$"], stats["Bias (DML)"], "o-", label="orthogonal DML")
    axes[0].plot(stats["$N$"], stats["Bias (naive)"], "s--", label="naive two-step")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$N$")
    axes[0].set_ylabel("bias")
    axes[0].legend(frameon=False)
    axes[0].set_title("Bias")
    axes[1].plot(stats["$N$"], stats["Coverage"], "o-")
    axes[1].axhline(0.95, color="black", linestyle=":", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_xlabel(r"$N$")
    axes[1].set_ylabel("coverage")
    axes[1].set_title("Coverage of the 95\\% confidence interval".replace("\\%", "%"))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig2_bias_coverage.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    index = np.arange(n_groups)
    width = 0.4
    ax.bar(index - width / 2, freq_table["M1 selection freq."], width, label="M1 (corrected)")
    ax.bar(index + width / 2, freq_table["M2 selection freq."], width, label="M2 (uncorrected)")
    for j in sim["active_outcome"]:
        ax.axvspan(j - 0.5, j + 0.5, color="grey", alpha=0.15)
    ax.set_xticks(index)
    ax.set_xticklabels([f"{j + 1}" for j in range(n_groups)], fontsize=8)
    ax.set_xlabel("characteristic $j$ (shaded: truly active)")
    ax.set_ylabel("selection frequency")
    ax.legend(frameon=False)
    ax.set_title("Group selection frequencies of the adaptive KG-lasso")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig3_selection_frequency.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(df_s["m1_imputation_bias"], bins=25, alpha=0.6, label="M1 (corrected)")
    ax.hist(df_s["m2_imputation_bias"], bins=25, alpha=0.6, label="M2 (uncorrected)")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("mean imputation error for non-disclosing firms")
    ax.set_ylabel("frequency")
    ax.legend(frameon=False)
    ax.set_title("Selection bias in imputed outcomes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig4_imputation_bias.png"), dpi=200)
    plt.close(fig)

    print(f"done, results written to {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
