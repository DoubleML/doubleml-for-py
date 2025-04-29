"""
The :mod:`doubleml.irm` module implements double machine learning estimates based on interactive regression models.
"""

from .apo import DoubleMLAPO
from .apos import DoubleMLAPOS
from .cvar import DoubleMLCVAR
from .iivm import DoubleMLIIVM
from .irm import DoubleMLIRM
from .lpq import DoubleMLLPQ
from .pq import DoubleMLPQ
from .qte import DoubleMLQTE
from .ssm import DoubleMLSSM

__all__ = [
    "DoubleMLIRM",
    "DoubleMLAPO",
    "DoubleMLAPOS",
    "DoubleMLCVAR",
    "DoubleMLIIVM",
    "DoubleMLLPQ",
    "DoubleMLPQ",
    "DoubleMLQTE",
    "DoubleMLSSM",
]
