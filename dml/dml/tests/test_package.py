"""Test the dml package."""


def test_version_is_string():
    import dml
    assert isinstance(dml.__version__, str)

