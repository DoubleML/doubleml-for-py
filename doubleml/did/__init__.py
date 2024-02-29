"""
The :mod:`doubleml.did` module implements double machine learning estimates based on difference in differences models.
"""

from .did import DoubleMLDID
from .did_cs import DoubleMLDIDCS

__all__ = [
    "DoubleMLDID",
    "DoubleMLDIDCS",
]
