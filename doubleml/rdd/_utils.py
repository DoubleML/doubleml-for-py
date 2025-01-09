import importlib


def _is_rdrobust_available():
    try:
        rdrobust = importlib.import_module("rdrobust")
        return rdrobust
    except ImportError:
        return None
