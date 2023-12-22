from pkg_resources import get_distribution

from .plr.plr import DoubleMLPLR
from .iv.pliv import DoubleMLPLIV
from .irm.irm import DoubleMLIRM
from .iv.iivm import DoubleMLIIVM
from .double_ml_data import DoubleMLData, DoubleMLClusterData
from .did.did import DoubleMLDID
from .did.did_cs import DoubleMLDIDCS
from .irm.qte import DoubleMLQTE
from .irm.pq import DoubleMLPQ
from .irm.lpq import DoubleMLLPQ
from .irm.cvar import DoubleMLCVAR

from .utils.blp import DoubleMLBLP
from .utils.policytree import DoubleMLPolicyTree

__all__ = ['DoubleMLPLR',
           'DoubleMLPLIV',
           'DoubleMLIRM',
           'DoubleMLIIVM',
           'DoubleMLData',
           'DoubleMLClusterData',
           'DoubleMLDID',
           'DoubleMLDIDCS',
           'DoubleMLPQ',
           'DoubleMLQTE',
           'DoubleMLLPQ',
           'DoubleMLCVAR',
           'DoubleMLBLP',
           'DoubleMLPolicyTree']

__version__ = get_distribution('doubleml').version
