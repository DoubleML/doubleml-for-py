import importlib.metadata

from .data import DoubleMLClusterData, DoubleMLData, DoubleMLDIDData, DoubleMLPanelData, DoubleMLRDDData, DoubleMLSSMData
from .did.did import DoubleMLDID
from .did.did_cs import DoubleMLDIDCS
from .double_ml_framework import DoubleMLFramework, concat
from .irm.apo import DoubleMLAPO
from .irm.apos import DoubleMLAPOS
from .irm.cvar import DoubleMLCVAR
from .irm.iivm import DoubleMLIIVM
from .irm.irm import DoubleMLIRM
from .irm.lpq import DoubleMLLPQ
from .irm.pq import DoubleMLPQ
from .irm.qte import DoubleMLQTE
from .irm.ssm import DoubleMLSSM
from .plm.lplr import DoubleMLLPLR
from .plm.pliv import DoubleMLPLIV
from .plm.plr import DoubleMLPLR
from .utils.blp import DoubleMLBLP
from .utils.policytree import DoubleMLPolicyTree

__all__ = [
    "concat",
    "DoubleMLFramework",
    "DoubleMLPLR",
    "DoubleMLPLIV",
    "DoubleMLIRM",
    "DoubleMLAPO",
    "DoubleMLAPOS",
    "DoubleMLIIVM",
    "DoubleMLData",
    "DoubleMLClusterData",
    "DoubleMLDIDData",
    "DoubleMLPanelData",
    "DoubleMLRDDData",
    "DoubleMLSSMData",
    "DoubleMLDID",
    "DoubleMLDIDCS",
    "DoubleMLPQ",
    "DoubleMLQTE",
    "DoubleMLLPQ",
    "DoubleMLCVAR",
    "DoubleMLBLP",
    "DoubleMLPolicyTree",
    "DoubleMLSSM",
    "DoubleMLLPLR",
]

__version__ = importlib.metadata.version("doubleml")
