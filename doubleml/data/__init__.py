"""
The :mod:`doubleml.data` module implements data classes for double machine learning.
"""

from .base_data import DoubleMLData
from .cluster_data import DoubleMLClusterData
from .panel_data import DoubleMLPanelData

__all__ = [
    "DoubleMLData",
    "DoubleMLClusterData",
    "DoubleMLPanelData",
]
