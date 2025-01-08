import pytest
from unittest.mock import patch


@pytest.mark.ci
def test_rdrobust_import_error():
    with patch('doubleml.rdd.rdd._rdrobust_available', False):
        msg = r"rdrobust is not installed. Please install it using 'pip install DoubleML\[rdd\]'"
        with pytest.raises(ImportError, match=msg):
            from doubleml.rdd import RDFlex
            RDFlex(None, None)
