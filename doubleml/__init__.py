from pkg_resources import get_distribution

from .double_ml_plr import DoubleMLPLR
from .double_ml_pliv import DoubleMLPLIV
from .double_ml_irm import DoubleMLIRM
from .double_ml_iivm import DoubleMLIIVM
from .double_ml_pcorr import DoubleMLPartialCorr
from .double_ml_copula import DoubleMLPartialCopula

from .double_ml_data import DoubleMLData, DoubleMLClusterData, DoubleMLPartialDependenceData
from .double_ml_blp import DoubleMLBLP

__all__ = ['DoubleMLPLR',
           'DoubleMLPLIV',
           'DoubleMLIRM',
           'DoubleMLIIVM',
           'DoubleMLPartialCorr',
           'DoubleMLPartialCopula',
           'DoubleMLData',
           'DoubleMLClusterData',
           'DoubleMLPartialDependenceData',
           'DoubleMLBLP']

__version__ = get_distribution('doubleml').version
