"""
The :mod:`doubleml.data` module implements data classes for double machine learning.
"""

from .base_data import DoubleMLData
from .did_data import DoubleMLDIDData
from .panel_data import DoubleMLPanelData
from .rdd_data import DoubleMLRDDData
from .ssm_data import DoubleMLSSMData

__all__ = [
    "DoubleMLData",
    "DoubleMLDIDData",
    "DoubleMLPanelData",
    "DoubleMLRDDData",
    "DoubleMLSSMData",
]
