import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def _sensitivity_contour_plot(
    x,
    y,
    contour_values,
    unadjusted_value,
    scenario_x,
    scenario_y,
    scenario_value,
    include_scenario,
    benchmarks=None,
    fill=True,
):
    if fill:
        text_col = "white"
        contours_coloring = "heatmap"
    else:
        text_col = "black"
        contours_coloring = "lines"

    # create figure
    axis_names = ["cf_d", "cf_y ", "Bound"]
    fig = go.Figure()
    # basic contour plot
    hov_temp = axis_names[0] + ": %{x:.3f}" + "<br>" + axis_names[1] + ": %{y:.3f}" + "</b>" + "<br>" + axis_names[2]
    fig.add_trace(
        go.Contour(
            z=contour_values,
            x=x,
            y=y,
            hovertemplate=hov_temp + ": %{z:.3f}" + "</b>",
            contours=dict(coloring=contours_coloring, showlabels=True, labelfont=dict(size=12, color=text_col)),
            name="Contour",
        )
    )

    if include_scenario:
        fig.add_trace(
            go.Scatter(
                x=[scenario_x],
                y=[scenario_y],
                mode="markers+text",
                marker=dict(size=10, color="red", line=dict(width=2, color=text_col)),
                hovertemplate=hov_temp + f": {round(scenario_value, 3)}" + "</b>",
                name="Scenario",
                textfont=dict(color=text_col, size=14),
                text=["<b>Scenario</b>"],
                textposition="top right",
                showlegend=False,
            )
        )

    # add unadjusted
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            marker=dict(size=10, color="red", line=dict(width=2, color=text_col)),
            hovertemplate=hov_temp + f": {round(unadjusted_value, 3)}" + "</b>",
            name="Unadjusted",
            text=["<b>Unadjusted</b>"],
            textfont=dict(color=text_col, size=14),
            textposition="top right",
            showlegend=False,
        )
    )

    # add benchmarks
    if benchmarks is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmarks["cf_d"],
                y=benchmarks["cf_y"],
                customdata=benchmarks["value"].reshape(-1, 1),
                mode="markers+text",
                marker=dict(size=10, color="red", line=dict(width=2, color=text_col)),
                hovertemplate=hov_temp + ": %{customdata[0]:.3f}" + "</b>",
                name="Benchmarks",
                textfont=dict(color=text_col, size=14),
                text=list(map(lambda s: "<b>" + s + "</b>", benchmarks["name"])),
                textposition="top right",
                showlegend=False,
            )
        )
    fig.update_layout(title=None, xaxis_title=axis_names[0], yaxis_title=axis_names[1])

    fig.update_xaxes(range=[0, np.max(x)])
    fig.update_yaxes(range=[0, np.max(y)])

    return fig


def _propensity_score_overlap_plot(
    ps_scores: np.ndarray,
    treatment: np.ndarray,
    threshold: float = 0.05,
) -> go.Figure:
    """Create an interactive propensity score overlap (common support) plot.

    Visualizes the distribution of estimated propensity scores split by treatment status
    using kernel density estimation. Highlights regions near 0 and 1 where the positivity
    assumption may be violated.

    Parameters
    ----------
    ps_scores : :class:`numpy.ndarray`
        Array of estimated propensity scores.
    treatment : :class:`numpy.ndarray`
        Binary treatment indicator array.
    threshold : float
        Threshold for positivity violation warning zones. Lines are drawn at ``threshold``
        and ``1 - threshold``. Default is ``0.05``.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure`
        Plotly figure with the propensity score overlap plot.
    """
    ps_treated = ps_scores[treatment == 1]
    ps_control = ps_scores[treatment == 0]

    # Compute KDE for both groups
    x_grid = np.linspace(0, 1, 200)

    kde_treated = gaussian_kde(ps_treated)
    kde_control = gaussian_kde(ps_control)
    density_treated = kde_treated(x_grid)
    density_control = kde_control(x_grid)

    fig = go.Figure()

    # Add danger zone shading (near 0 and near 1)
    fig.add_vrect(
        x0=0,
        x1=threshold,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
        annotation_text="Positivity<br>concern",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="red",
    )
    fig.add_vrect(
        x0=1 - threshold,
        x1=1,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
        annotation_text="Positivity<br>concern",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="red",
    )

    # KDE curves for treated and control
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=density_treated,
            mode="lines",
            name="Treated",
            fill="tozeroy",
            line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
            fillcolor="rgba(31, 119, 180, 0.25)",
            hovertemplate="PS: %{x:.3f}<br>Density: %{y:.3f}<extra>Treated</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=density_control,
            mode="lines",
            name="Control",
            fill="tozeroy",
            line=dict(color="rgba(255, 127, 14, 0.9)", width=2),
            fillcolor="rgba(255, 127, 14, 0.25)",
            hovertemplate="PS: %{x:.3f}<br>Density: %{y:.3f}<extra>Control</extra>",
        )
    )

    # Threshold boundary lines
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.7)
    fig.add_vline(x=1 - threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.7)

    # Summary annotation with positivity diagnostics
    n_total = len(ps_scores)
    n_below = np.sum(ps_scores < threshold)
    n_above = np.sum(ps_scores > 1 - threshold)
    pct_violation = (n_below + n_above) / n_total * 100

    annotation_text = (
        f"<b>Positivity diagnostics</b><br>"
        f"PS < {threshold}: {n_below} ({n_below / n_total * 100:.1f}%)<br>"
        f"PS > {1 - threshold}: {n_above} ({n_above / n_total * 100:.1f}%)<br>"
        f"Total violations: {pct_violation:.1f}%"
    )
    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        font=dict(size=11),
        align="left",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255, 255, 255, 0.85)",
    )

    fig.update_layout(
        title="Propensity Score Overlap (Common Support)",
        xaxis_title="Estimated Propensity Score",
        yaxis_title="Density",
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.85, xanchor="left", x=0.02),
        hovermode="x unified",
    )

    return fig

