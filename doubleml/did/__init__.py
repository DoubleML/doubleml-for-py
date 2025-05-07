"""
The :mod:`doubleml.did` module implements double machine learning estimates based on difference in differences models.
"""

from .did import DoubleMLDID
from .did_aggregation import DoubleMLDIDAggregation
from .did_binary import DoubleMLDIDBinary
from .did_cs import DoubleMLDIDCS
from .did_multi import DoubleMLDIDMulti

__all__ = [
    "DoubleMLDIDAggregation",
    "DoubleMLDID",
    "DoubleMLDIDCS",
    "DoubleMLDIDBinary",
    "DoubleMLDIDMulti",
]
