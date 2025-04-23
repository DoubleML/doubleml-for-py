import warnings
from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from doubleml.double_ml_framework import DoubleMLFramework, concat


class DoubleMLDIDAggregation:
    """
    Class for aggregating multiple difference-in-differences (DID) frameworks.

    This class enables weighted aggregation of multiple DoubleMLFramework objects, allowing for
    both multiple separate aggregations and an overall aggregation across them. It provides
    methods for summarizing and visualizing aggregated treatment effects.

    Parameters
    ----------
    frameworks : list
        List of DoubleMLFramework objects to aggregate. Each framework must be one-dimensional
        (n_thetas = 1).

    aggregation_weights : numpy.ndarray
        2D array of weights for aggregating frameworks. Shape should be (n_aggregations, n_frameworks),
        where each row corresponds to a separate aggregation of the frameworks.

    overall_aggregation_weights : numpy.ndarray, optional
        1D array of weights for aggregating across the aggregated frameworks. Length should
        equal the number of rows in aggregation_weights. If None, equal weights are used.
        Default is None.

    aggregation_names : list of str, optional
        Names for each aggregation. Length should equal the number of rows in aggregation_weights.
        If None, default names like "Aggregation_0", "Aggregation_1", etc. are used.
        Default is None.

    aggregation_method_name : str, optional
        Name describing the aggregation method used.
        Default is "Custom".

    additional_information : dict, optional
        Dictionary containing additional information to display in the string representation.
        Default is None.

    additional_parameters : dict, optional
        Dictionary containing additional parameters used by the class methods.
        For example, can contain 'aggregation_color_idx' for plot_effects().
        Default is None.
    """

    def __init__(
        self,
        frameworks,
        aggregation_weights,
        overall_aggregation_weights=None,
        aggregation_names=None,
        aggregation_method_name="Custom",
        additional_information=None,
        additional_parameters=None,
    ):
        self._base_frameworks = self._check_frameworks(frameworks)

        self._aggregation_weights, self._overall_aggregation_weights = self._check_weights(
            aggregation_weights, overall_aggregation_weights
        )
        self._n_aggregations = self.aggregation_weights.shape[0]

        self._aggregation_names, self._aggregation_method_name = self._check_names(aggregation_names, aggregation_method_name)

        if additional_information is not None and not isinstance(additional_information, dict):
            raise TypeError("'additional_information' must be a dictionary (or None)")
        self._additional_information = additional_information
        if additional_parameters is not None and not isinstance(additional_parameters, dict):
            raise TypeError("'additional_parameters' must be a dictionary (or None)")
        self._additional_parameters = additional_parameters

        agg_frameworks = [None] * self._n_aggregations
        for idx_agg in range(self._n_aggregations):
            weights = self.aggregation_weights[idx_agg, :]
            weighted_frameworks = [w * f for w, f in zip(weights, self.base_frameworks)]
            agg_frameworks[idx_agg] = reduce(add, weighted_frameworks)

        self._aggregated_frameworks = concat(agg_frameworks)
        self._aggregated_frameworks.treatment_names = self._aggregation_names

        # overall framework
        overall_weighted_frameworks = [w * f for w, f in zip(self.overall_aggregation_weights, agg_frameworks)]
        self._overall_aggregated_framework = reduce(add, overall_weighted_frameworks)

    def __str__(self):
        class_name = self.__class__.__name__
        header = (
            f"================== {class_name} Object ==================\n" + f" {self.aggregation_method_name} Aggregation \n"
        )
        overall_summary = self.overall_summary.to_string(index=False)
        aggregated_effects_summary = self.aggregated_summary.to_string(index=True)

        res = (
            header
            + "\n------------------ Overall Aggregated Effects ------------------\n"
            + overall_summary
            + "\n------------------ Aggregated Effects         ------------------\n"
            + aggregated_effects_summary
        )
        if self.additional_information is not None:
            res += "\n------------------ Additional Information     ------------------\n"
            res += self.additional_information

        return res

    @property
    def base_frameworks(self):
        """Underlying frameworks"""
        return self._base_frameworks

    @property
    def aggregated_frameworks(self):
        """Aggregated frameworks"""
        return self._aggregated_frameworks

    @property
    def overall_aggregated_framework(self):
        """Overall aggregated framework"""
        return self._overall_aggregated_framework

    @property
    def aggregation_weights(self):
        """Aggregation weights"""
        return self._aggregation_weights

    @property
    def overall_aggregation_weights(self):
        """Overall aggregation weights"""
        return self._overall_aggregation_weights

    @property
    def n_aggregations(self):
        """Number of aggregations"""
        return self._n_aggregations

    @property
    def aggregation_names(self):
        """Aggregation names"""
        return self._aggregation_names

    @property
    def aggregation_method_name(self):
        """Aggregation method name"""
        return self._aggregation_method_name

    @property
    def aggregated_summary(self):
        """
        A summary for the aggregated effects.
        """
        return self.aggregated_frameworks.summary

    @property
    def overall_summary(self):
        """
        A summary for the overall aggregated effect.
        """
        return self.overall_aggregated_framework.summary

    @property
    def additional_information(self):
        """Additional information"""
        if self._additional_information is None:
            add_info = None
        else:
            add_info = str()
            for key, value in self._additional_information.items():
                add_info += f"{key}: {value}\n"
        return add_info

    @property
    def additional_parameters(self):
        """Additional parameters"""
        return self._additional_parameters

    def plot_effects(
        self,
        level=0.95,
        joint=True,
        figsize=(12, 6),
        sort_by=None,
        color_palette="colorblind",
        title="Aggregated Treatment Effects",
        y_label="Effect",
    ):
        """
        Plot aggregated treatment effect estimates with confidence intervals.

        Parameters
        ----------
        level : float
            Confidence level for the intervals.
            Default is ``0.95``.
        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``True``.
        figsize : tuple
            Figure size as (width, height).
            Default is ``(12, 6)``.
        sort_by : str or None
            How to sort the results - 'estimate', 'name', or None.
            Default is ``None``.
        color_palette : str or list
            Seaborn color palette name or list of colors.
            Default is ``"colorblind"``.
        title : str
            Title for the plot.
            Default is ``"Aggregated Treatment Effects"``.
        y_label : str
            Label for y-axis.
            Default is ``"Effect"``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object.
        ax : matplotlib.axes.Axes
            The axes object for further customization.

        Notes
        -----
        If ``joint=True`` and bootstrapping hasn't been performed, this method will automatically
        perform bootstrapping with default parameters and issue a warning.
        """
        df = self._create_ci_dataframe(level=level, joint=joint)

        # Validate sorting column
        valid_sort_options = {"estimate", "name", None}
        if sort_by not in valid_sort_options:
            raise ValueError(f"Invalid sort_by value. Choose from {valid_sort_options}.")

        # Sort data if requested
        if sort_by == "estimate":
            df = df.sort_values(by="Estimate", ascending=False)
        elif sort_by == "name":
            df = df.sort_values(by="Aggregation_Names", ascending=True)

        # Handle color palette
        colors = sns.color_palette(color_palette) if isinstance(color_palette, str) else color_palette
        selected_colors = [colors[idx] for idx in df["color_idx"]]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot zero reference line
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Zero effect")

        # Calculate asymmetric error bars
        x_positions = np.arange(len(df))
        yerr = np.array([df["Estimate"] - df["CI_Lower"], df["CI_Upper"] - df["Estimate"]])  # lower error  # upper error

        for i, (x, y, color) in enumerate(zip(x_positions, df["Estimate"], selected_colors)):
            ax.errorbar(
                x,
                y,
                yerr=[[yerr[0, i]], [yerr[1, i]]],
                fmt="o",
                capsize=4,
                color=color,
                ecolor=color,
                markersize=8,
                markeredgewidth=1.5,
                linewidth=1.5,
            )

        # Set labels and title
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df["Aggregation_Names"])
        ax.set_ylabel(y_label)
        ax.set_title(title)

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        return fig, ax

    def _check_frameworks(self, frameworks):
        msg = "The 'frameworks' must be a list of DoubleMLFramework objects"
        is_list = isinstance(frameworks, list)
        all_frameworks = all(isinstance(framework, DoubleMLFramework) for framework in frameworks)
        if not is_list or not all_frameworks:
            raise TypeError(msg)

        if not all(framework.n_thetas == 1 for framework in frameworks):
            raise ValueError("All frameworks must be one-dimensional")

        return frameworks

    def _check_weights(self, aggregation_weights, overall_aggregation_weights):

        # aggregation weights
        if not isinstance(aggregation_weights, np.ndarray):
            raise TypeError("'aggregation_weights' must be a numpy array")

        if not aggregation_weights.ndim == 2:
            raise ValueError("'aggregation_weights' must be a 2-dimensional array")

        if not aggregation_weights.shape[1] == len(self.base_frameworks):
            raise ValueError("The number of rows in 'aggregation_weights' must be equal to the number of frameworks")

        n_aggregations = aggregation_weights.shape[0]
        # overall aggregation weights
        if overall_aggregation_weights is None:
            overall_aggregation_weights = np.ones(n_aggregations) / n_aggregations

        if not isinstance(overall_aggregation_weights, np.ndarray):
            raise TypeError("'overall_aggregation_weights' must be a numpy array")
        if not overall_aggregation_weights.ndim == 1:
            raise ValueError("'overall_aggregation_weights' must be a 1-dimensional array")
        if not len(overall_aggregation_weights) == n_aggregations:
            raise ValueError(
                "'overall_aggregation_weights' must have the same length as the number of aggregated frameworks "
                "(number of rows in 'aggregation_weights')."
            )

        return aggregation_weights, overall_aggregation_weights

    def _check_names(self, aggregation_names, aggregation_method_name):
        if aggregation_names is None:
            aggregation_names = [f"Aggregation_{i}" for i in range(self.n_aggregations)]

        if not isinstance(aggregation_names, list):
            raise TypeError("'aggregation_names' must be a list of strings")

        if not all(isinstance(name, str) for name in aggregation_names):
            raise TypeError("'aggregation_names' must be a list of strings")

        if not len(aggregation_names) == self.n_aggregations:
            raise ValueError("'aggregation_names' must have the same length as the number of aggregations")

        if not isinstance(aggregation_method_name, str):
            raise TypeError("'aggregation_method_name' must be a string")

        return aggregation_names, aggregation_method_name

    def _create_ci_dataframe(self, level=0.95, joint=True):
        """
        Create a DataFrame with coefficient estimates and confidence intervals.

        Parameters
        ----------
        level : float, default=0.95
            Confidence level for intervals.
        joint : bool, default=True
            Whether to use joint confidence intervals.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:
            - Aggregation names
            - Coefficient estimates
            - Lower and upper confidence interval bounds
            - Color indices for plotting
        """

        if joint and self.aggregated_frameworks.boot_t_stat is None:
            self.aggregated_frameworks.bootstrap()
            warnings.warn(
                "Joint confidence intervals require bootstrapping which hasn't been performed yet. "
                "Automatically applying '.aggregated_frameworks.bootstrap(method=\"normal\", n_rep_boot=500)' "
                "with default values. For different bootstrap settings, call bootstrap() explicitly before plotting.",
                UserWarning,
            )
        ci = self.aggregated_frameworks.confint(level=level, joint=joint)

        default_color_idx = [0] * self._n_aggregations
        if self.additional_parameters is None:
            color_idx = default_color_idx
        else:
            color_idx = self.additional_parameters.get("aggregation_color_idx", default_color_idx)

        df = pd.DataFrame(
            {
                "Aggregation_Names": self.aggregation_names,
                "Estimate": self.aggregated_frameworks.thetas,
                "CI_Lower": ci.iloc[:, 0],
                "CI_Upper": ci.iloc[:, 1],
                "color_idx": color_idx,
            }
        )
        return df
