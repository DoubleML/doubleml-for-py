"""
The :mod:`doubleml.plm.datasets` module implements data generating processes for partially linear models.
"""

from ._make_pliv_data import _make_pliv_data
from .dgp_confounded_plr_data import make_confounded_plr_data
from .dgp_lplr_LZZ2020 import make_lplr_LZZ2020
from .dgp_pliv_CHS2015 import make_pliv_CHS2015
from .dgp_pliv_multiway_cluster_CKMS2021 import make_pliv_multiway_cluster_CKMS2021
from .dgp_plr_CCDDHNR2018 import make_plr_CCDDHNR2018
from .dgp_plr_turrell2018 import make_plr_turrell2018

__all__ = [
    "make_plr_CCDDHNR2018",
    "make_plr_turrell2018",
    "make_confounded_plr_data",
    "make_pliv_CHS2015",
    "make_pliv_multiway_cluster_CKMS2021",
    "make_lplr_LZZ2020",
    "_make_pliv_data",
]
