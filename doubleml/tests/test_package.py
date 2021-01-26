import pytest


@pytest.mark.ci
def test_version_is_string():
    import doubleml
    assert isinstance(doubleml.__version__, str)
