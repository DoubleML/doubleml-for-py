import pytest

from doubleml.utils import PropensityScoreProcessor


@pytest.mark.ci
def test_repr_default_config():
    """Test __repr__ with default configuration."""
    processor = PropensityScoreProcessor()
    expected = (
        "PropensityScoreProcessor(clipping_threshold=0.01, extreme_threshold=0.05, "
        "warn_extreme_values=True, warning_proportion=0.1)"
    )
    assert repr(processor) == expected


@pytest.mark.ci
def test_repr_custom_config():
    """Test __repr__ with custom configuration."""
    processor = PropensityScoreProcessor(clipping_threshold=0.05, warn_extreme_values=False, warning_proportion=0.2)
    expected = (
        "PropensityScoreProcessor(clipping_threshold=0.05, extreme_threshold=0.05, "
        "warn_extreme_values=False, warning_proportion=0.2)"
    )
    assert repr(processor) == expected


@pytest.mark.ci
def test_eq_same_config():
    """Test equality with same configuration."""
    processor1 = PropensityScoreProcessor(clipping_threshold=0.05)
    processor2 = PropensityScoreProcessor(clipping_threshold=0.05)
    assert processor1 == processor2


@pytest.mark.ci
def test_eq_different_config():
    """Test inequality with different configuration."""
    processor1 = PropensityScoreProcessor(clipping_threshold=0.05)
    processor2 = PropensityScoreProcessor(clipping_threshold=0.1)
    assert processor1 != processor2
