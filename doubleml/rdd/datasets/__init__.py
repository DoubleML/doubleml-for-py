"""
The :mod:`doubleml.rdd.datasets` module implements data generating processes for regression discontinuity designs.
"""

from .simple_dgp import make_simple_rdd_data
from .area_yield_dgp import dgp_area_yield

__all__ = [
    "make_simple_rdd_data",
    "dgp_area_yield",
]
