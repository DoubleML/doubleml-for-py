import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.mark.ci
def test_version_from_version_file():
    """Test version is imported from _version.py when available."""
    import doubleml

    assert hasattr(doubleml, "__version__")
    assert isinstance(doubleml.__version__, str)


@pytest.mark.ci
def test_version_fallback_to_metadata():
    """Test fallback to importlib.metadata when _version.py is missing."""
    with patch.dict(sys.modules, {"doubleml._version": None}):
        with patch("importlib.metadata.version", return_value="1.2.3"):
            # Re-import to trigger the fallback
            importlib.reload(importlib.import_module("doubleml"))
            import doubleml

            assert doubleml.__version__ == "1.2.3"


@pytest.mark.ci
def test_version_fallback_to_unknown():
    """Test fallback to 0.0.0+unknown when package not found."""
    mock_error = importlib.metadata.PackageNotFoundError()
    with patch.dict(sys.modules, {"doubleml._version": None}):
        with patch("importlib.metadata.version", side_effect=mock_error):
            importlib.reload(importlib.import_module("doubleml"))
            import doubleml

            assert doubleml.__version__ == "0.0.0+unknown"
