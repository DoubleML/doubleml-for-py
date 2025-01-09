import pytest
from unittest.mock import patch

import doubleml as dml


@pytest.mark.ci
def test_rdrobust_import_error():
    with patch('doubleml.rdd.rdd.rdrobust', None):
        msg = r"rdrobust is not installed. Please install it using 'pip install DoubleML\[rdd\]'"
        with pytest.raises(ImportError, match=msg):
            dml.rdd.RDFlex(None, None)
