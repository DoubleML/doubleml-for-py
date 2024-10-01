"""
The :mod:`doubleml.did` module implements double machine learning estimates based on difference in differences models.
"""

from .did import DoubleMLDID
from .did_cs import DoubleMLDIDCS
from .did_binary import DoubleMLDIDBINARY

__all__ = [
    "DoubleMLDID",
    "DoubleMLDIDCS",
    "DoubleMLDIDBINARY"
]
