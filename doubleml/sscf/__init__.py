"""Double machine learning for high-dimensional sample selection models with a
control function (Heckman-type) correction.

The package extends `DoubleML <https://docs.doubleml.org>`_ by the model class
:class:`DoubleMLSSCF`, which estimates the sample selection bias coefficient
:math:`\\theta_0` with a Neyman-orthogonal score.
"""

from .datasets import (
    decile_grid,
    gaussian_kernel_features,
    kernel_gram_matrices,
    make_green_silence_data,
)
from .sscf import DoubleMLSSCF
from .utils import generalized_inverse_mills_ratio, selection_bias_score_test

__all__ = [
    "DoubleMLSSCF",
    "make_green_silence_data",
    "decile_grid",
    "gaussian_kernel_features",
    "kernel_gram_matrices",
    "generalized_inverse_mills_ratio",
    "selection_bias_score_test",
]

__version__ = "0.1.0"
