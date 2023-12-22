from pkg_resources import get_distribution

from .double_ml_plr import DoubleMLPLR
from .iv_model.pliv import DoubleMLPLIV
from .double_ml_irm import DoubleMLIRM
from .iv_model.iivm import DoubleMLIIVM
from .double_ml_data import DoubleMLData, DoubleMLClusterData
from .did_model.did import DoubleMLDID
from .did_model.did_cs import DoubleMLDIDCS
from .double_ml_qte import DoubleMLQTE
from .double_ml_pq import DoubleMLPQ
from .double_ml_lpq import DoubleMLLPQ
from .double_ml_cvar import DoubleMLCVAR

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
