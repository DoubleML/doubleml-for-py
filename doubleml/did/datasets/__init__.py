"""
The :mod:`doubleml.did.datasets` module implements data generating processes for difference-in-differences.
"""


from .dgp_did_SZ2020 import make_did_SZ2020

__all__ = [
    "make_did_SZ2020",
]
