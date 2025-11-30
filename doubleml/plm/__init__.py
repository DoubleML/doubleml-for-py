"""
The :mod:`doubleml.plm` module implements double machine learning estimates based on partially linear models.
"""

from .lplr import DoubleMLLPLR
from .pliv import DoubleMLPLIV
from .plpr import DoubleMLPLPR
from .plr import DoubleMLPLR

__all__ = ["DoubleMLPLR", "DoubleMLPLIV", "DoubleMLLPLR", "DoubleMLPLPR"]
