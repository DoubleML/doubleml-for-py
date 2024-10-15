import importlib.metadata

from .double_ml_framework import concat
from .double_ml_framework import DoubleMLFramework
from .plm.plr import DoubleMLPLR
from .plm.pliv import DoubleMLPLIV
from .irm.irm import DoubleMLIRM
from .irm.apo import DoubleMLAPO
from .irm.apos import DoubleMLAPOS
from .irm.iivm import DoubleMLIIVM
from .double_ml_data import DoubleMLData, DoubleMLClusterData, DoubleMLPanelData
from .did.did import DoubleMLDID
from .did.did_cs import DoubleMLDIDCS
from .did.did_binary import DoubleMLDIDBINARY
from .did.did_multi import DoubleMLDIDMULTI
from .irm.qte import DoubleMLQTE
from .irm.pq import DoubleMLPQ
from .irm.lpq import DoubleMLLPQ
from .irm.cvar import DoubleMLCVAR
from .irm.ssm import DoubleMLSSM

from .utils.blp import DoubleMLBLP
from .utils.policytree import DoubleMLPolicyTree

__all__ = [
    'concat',
    'DoubleMLFramework',
    'DoubleMLPLR',
    'DoubleMLPLIV',
    'DoubleMLIRM',
    'DoubleMLAPO',
    'DoubleMLAPOS',
    'DoubleMLIIVM',
    'DoubleMLData',
    'DoubleMLClusterData',
    'DoubleMLPanelData',
    'DoubleMLDID',
    'DoubleMLDIDCS',
    'DoubleMLDIDBINARY',
    'DoubleMLDIDMULTI',
    'DoubleMLPQ',
    'DoubleMLQTE',
    'DoubleMLLPQ',
    'DoubleMLCVAR',
    'DoubleMLBLP',
    'DoubleMLPolicyTree',
    'DoubleMLSSM'
]

__version__ = importlib.metadata.version('doubleml')
