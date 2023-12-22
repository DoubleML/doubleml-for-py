"""
The :mod:`doubleml.did` module  implements a variety of difference in differences models.
"""

from .did import DoubleMLDID
from .did_cs import DoubleMLDIDCS

__all__ = [
    "DoubleMLDID",
    "DoubleMLDIDCS",
]
