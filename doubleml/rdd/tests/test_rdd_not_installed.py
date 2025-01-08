import pytest
from unittest.mock import patch

from doubleml.rdd._utils import _is_rdrobust_available


def test_rdrobust_import_error():
    with patch('importlib.import_module', side_effect=ImportError):
        msg = r"rdrobust is not installed. Please install it using 'pip install DoubleML\[rdd\]'"
        with pytest.raises(ImportError, match=msg):
            _is_rdrobust_available()