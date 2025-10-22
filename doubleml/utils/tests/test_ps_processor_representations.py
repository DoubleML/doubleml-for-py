import pytest

from doubleml.utils import PropensityScoreProcessor


@pytest.mark.ci
def test_repr_default_config():
    """Test __repr__ with default configuration."""
    processor = PropensityScoreProcessor()
    expected = (
        "PropensityScoreProcessor(calibration_method=None, clipping_threshold=0.01, "
        "cv_calibration=False, extreme_threshold=1e-12)"
    )
    assert repr(processor) == expected


@pytest.mark.ci
def test_repr_custom_config():
    """Test __repr__ with custom configuration."""
    processor = PropensityScoreProcessor(clipping_threshold=0.05, extreme_threshold=1e-6)
    expected = (
        "PropensityScoreProcessor(calibration_method=None, clipping_threshold=0.05, "
        "cv_calibration=False, extreme_threshold=1e-06)"
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


@pytest.mark.ci
def test_eq_different_type():
    """Test inequality with different object type."""
    processor = PropensityScoreProcessor()
    assert processor != "NotAPropensityScoreProcessor"
