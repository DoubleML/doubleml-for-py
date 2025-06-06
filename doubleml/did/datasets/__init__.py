"""
The :mod:`doubleml.did.datasets` module implements data generating processes for difference-in-differences.
"""

from .dgp_did_CS2021 import make_did_CS2021
from .dgp_did_cs_CS2021 import make_did_cs_CS2021
from .dgp_did_SZ2020 import make_did_SZ2020

__all__ = [
    "make_did_SZ2020",
    "make_did_CS2021",
    "make_did_cs_CS2021",
]
