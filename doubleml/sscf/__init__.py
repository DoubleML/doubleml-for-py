"""Double machine learning for high-dimensional sample selection models with a
control function (Heckman-type) correction.

The package extends `DoubleML <https://docs.doubleml.org>`_ by the model class
:class:`DoubleMLSSCF`, which estimates the sample selection bias coefficient
:math:`\\theta_0` with a Neyman-orthogonal score, and by the adaptive kernel group
lasso used for post variable selection in the outcome equation.
"""

from .datasets import (
    decile_grid,
    gaussian_kernel_features,
    kernel_gram_matrices,
    make_green_silence_data,
)
from .kgl import (
    AdaptiveKernelGroupLasso,
    adaptive_weights,
    kg_lasso_path,
    lambda_max,
    post_selection_refit,
    selection_metrics,
    variable_selection,
)
from .sscf import DoubleMLSSCF
from .utils import generalized_inverse_mills_ratio, selection_bias_score_test

__all__ = [
    "AdaptiveKernelGroupLasso",
    "DoubleMLSSCF",
    "adaptive_weights",
    "decile_grid",
    "gaussian_kernel_features",
    "generalized_inverse_mills_ratio",
    "kernel_gram_matrices",
    "kg_lasso_path",
    "lambda_max",
    "make_green_silence_data",
    "post_selection_refit",
    "selection_bias_score_test",
    "selection_metrics",
    "variable_selection",
]

__version__ = "0.1.0"
