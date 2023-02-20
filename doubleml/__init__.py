from pkg_resources import get_distribution

from .double_ml_plr import DoubleMLPLR
from .double_ml_pliv import DoubleMLPLIV
from .double_ml_irm import DoubleMLIRM
from .double_ml_iivm import DoubleMLIIVM
from .double_ml_data import DoubleMLData, DoubleMLClusterData, DoubleMLDIDData
from .double_ml_blp import DoubleMLBLP
from .double_ml_did_pa import DoubleMLDID
from .double_ml_did_cs import DoubleMLDiDCS

__all__ = ['DoubleMLPLR',
           'DoubleMLPLIV',
           'DoubleMLIRM',
           'DoubleMLIIVM',
           'DoubleMLData',
           'DoubleMLClusterData',
           'DoubleMLDIDData',
           'DoubleMLBLP',
           'DoubleMLDID',
           'DoubleMLDiDCS']

__version__ = get_distribution('doubleml').version
