"""
The :mod:`doubleml.datasets` module implements data generating processes for double machine learning simulations
and provides access to real datasets.
"""

# Import fetch functions
from .fetch_401K import fetch_401K
from .fetch_bonus import fetch_bonus

__all__ = [
    "fetch_401K",
    "fetch_bonus",
]
