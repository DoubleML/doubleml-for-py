from functools import reduce
from operator import add

import numpy as np

from doubleml.double_ml_framework import DoubleMLFramework, concat


class DoubleMLDIDAggregation:
    def __init__(
        self,
        frameworks,
        aggregation_weights,
        overall_aggregation_weights=None,
        aggregation_names=None,
        aggregation_method_name="custom",
        additional_information=None,
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
            f"================== {class_name} Object ==================\n" f" {self.aggregation_method_name} Aggregation \n"
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
