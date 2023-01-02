from pkg_resources import get_distribution

from .double_ml_plr import DoubleMLPLR
from .double_ml_pliv import DoubleMLPLIV
from .double_ml_irm import DoubleMLIRM
from .double_ml_iivm import DoubleMLIIVM
from .double_ml_data import DoubleMLData, DoubleMLClusterData
from .double_ml_qte import DoubleMLQTE
from .double_ml_pq import DoubleMLPQ
from .double_ml_lpq import DoubleMLLPQ
from .double_ml_cvar import DoubleMLCVAR

__all__ = ['DoubleMLPLR',
           'DoubleMLPLIV',
           'DoubleMLIRM',
           'DoubleMLIIVM',
           'DoubleMLData',
           'DoubleMLClusterData',
           'DoubleMLPQ',
           'DoubleMLQTE',
           'DoubleMLLPQ',
           'DoubleMLCVAR']

__version__ = get_distribution('doubleml').version
