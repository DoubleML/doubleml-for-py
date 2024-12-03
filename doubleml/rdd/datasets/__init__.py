"""
The :mod:`doubleml.rdd.datasets` module implements data generating processes for regression discontinuity designs.
"""

from .simple_dgp import make_simple_rdd_data

__all__ = [
    "make_simple_rdd_data",
]
