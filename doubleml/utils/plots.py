import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_propensity_score_calibration(
    propensity_score,
    treatment,
    bins=10,
    density=False,
    palette="colorblind",
):
    """
    Plot propensity score distributions and binned calibration curves.

    Parameters
    ----------
    propensity_score : array-like
        Predicted propensity scores of shape (n_samples,).
    treatment : array-like
        Binary treatment indicator of shape (n_samples,).
    bins : int or array-like
        Number of bins or explicit bin edges.
    density : bool
        If True, histogram heights are normalized.
    palette : str or sequence
        Seaborn palette name or explicit colors.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Matplotlib figure.
    axes : :class:`numpy.ndarray`
        2x2 axes array.
    """
    ps = np.asarray(propensity_score, dtype=float).reshape(-1)
    tr = np.asarray(treatment).reshape(-1)

    if ps.shape != tr.shape:
        raise ValueError("propensity_score and treatment must have the same shape.")
    if ps.ndim != 1:
        raise ValueError("propensity_score and treatment must be one-dimensional.")
    if not np.isin(tr, [0, 1]).all():
        raise ValueError("treatment must be binary with values 0 and 1.")
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError("propensity_score must lie in [0, 1].")

    tr = tr.astype(int)

    if isinstance(bins, int):
        if bins < 2:
            raise ValueError("bins must be at least 2.")
        bins = np.linspace(0.0, 1.0, bins + 1)
    else:
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or len(bins) < 2:
            raise ValueError("bins must contain at least two edges.")
        if np.any(np.diff(bins) <= 0):
            raise ValueError("bins must be strictly increasing.")

    x_min, x_max = float(bins[0]), float(bins[-1])
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    treated_frac = []
    control_frac = []
    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            mask = (ps >= bins[i]) & (ps < bins[i + 1])
        else:
            mask = (ps >= bins[i]) & (ps <= bins[i + 1])
        if np.sum(mask) == 0:
            treated_frac.append(np.nan)
            control_frac.append(np.nan)
        else:
            p_treated = np.mean(tr[mask] == 1)
            treated_frac.append(p_treated)
            control_frac.append(1.0 - p_treated)

    colors = sns.color_palette(palette, n_colors=2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

    sns.histplot(
        ps[tr == 1],
        bins=bins,
        stat="density" if density else "count",
        kde=False,
        color=colors[0],
        ax=axes[0, 0],
        label="Treated",
    )
    axes[0, 0].set_title("Treated: Propensity Score Distribution")
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylabel("Density" if density else "Count")
    axes[0, 0].legend()

    sns.histplot(
        ps[tr == 0],
        bins=bins,
        stat="density" if density else "count",
        kde=False,
        color=colors[1],
        ax=axes[0, 1],
        label="Control",
    )
    axes[0, 1].set_title("Control: Propensity Score Distribution")
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_ylabel("Density" if density else "Count")
    axes[0, 1].legend()

    axes[1, 0].bar(centers, treated_frac, width=widths, color=colors[0], alpha=0.7)
    axes[1, 0].plot([x_min, x_max], [x_min, x_max], "k--", label="Ideal calibration")
    axes[1, 0].set_title("Treated: Calibration")
    axes[1, 0].set_xlabel("Predicted propensity score")
    axes[1, 0].set_ylabel("Observed treatment fraction")
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].bar(centers, control_frac, width=widths, color=colors[1], alpha=0.7)
    axes[1, 1].plot([x_min, x_max], [1 - x_min, 1 - x_max], "k--", label="Ideal calibration")
    axes[1, 1].set_title("Control: Calibration")
    axes[1, 1].set_xlabel("Predicted propensity score")
    axes[1, 1].set_ylabel("Observed control fraction")
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    fig.suptitle("Propensity Score Calibration")
    plt.tight_layout()

    return fig, axes
