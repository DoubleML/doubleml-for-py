import importlib


def _is_rdrobust_available():
    try:
        rdrobust = importlib.import_module("rdrobust")
        return rdrobust
    except ImportError:
        msg = (
            "rdrobust is not installed. "
            "Please install it using 'pip install DoubleML[rdd]'")
        raise ImportError(msg)
