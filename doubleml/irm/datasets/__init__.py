"""
The :mod:`doubleml.irm.datasets` module implements data generating processes for interactive regression models.
"""

from .dgp_confounded_irm_data import make_confounded_irm_data
from .dgp_heterogeneous_data import make_heterogeneous_data
from .dgp_iivm_data import make_iivm_data
from .dgp_irm_data import make_irm_data
from .dgp_irm_data_discrete_treatments import make_irm_data_discrete_treatments
from .dgp_ssm_data import make_ssm_data

__all__ = [
    "make_confounded_irm_data",
    "make_heterogeneous_data",
    "make_iivm_data",
    "make_irm_data",
    "make_irm_data_discrete_treatments",
    "make_ssm_data",
]
