import pytest

from doubleml.utils import PropensityScoreProcessor

# -------------------------------------------------------------------------
# Tests for __init__ method
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_init_unknown_parameter():
    """Test that unknown parameters raise ValueError during initialization."""
    with pytest.raises(ValueError, match="Unknown parameters: {'invalid_param'}"):
        PropensityScoreProcessor(invalid_param=0.5)


@pytest.mark.ci
def test_init_clipping_threshold_type_error():
    """Test that non-float clipping_threshold raises TypeError."""
    with pytest.raises(TypeError, match="clipping_threshold must be of float type"):
        PropensityScoreProcessor(clipping_threshold="0.01")


@pytest.mark.ci
def test_init_clipping_threshold_value_error():
    """Test that invalid clipping_threshold values raise ValueError."""
    with pytest.raises(ValueError, match="clipping_threshold must be between 0 and 0.5"):
        PropensityScoreProcessor(clipping_threshold=0.0)  # exactly 0

    with pytest.raises(ValueError, match="clipping_threshold must be between 0 and 0.5"):
        PropensityScoreProcessor(clipping_threshold=0.6)  # above 0.5


@pytest.mark.ci
def test_init_extreme_threshold_value_error():
    """Test that invalid extreme_threshold values raise ValueError."""
    with pytest.raises(ValueError, match="extreme_threshold must be between 0 and 0.5"):
        PropensityScoreProcessor(extreme_threshold=0.0)  # exactly 0

    with pytest.raises(ValueError, match="extreme_threshold must be between 0 and 0.5"):
        PropensityScoreProcessor(extreme_threshold=0.6)  # above 0.5


# -------------------------------------------------------------------------
# Tests for update_config method
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_update_config_unknown_parameter():
    """Test that unknown parameters raise ValueError during config update."""
    processor = PropensityScoreProcessor()

    with pytest.raises(ValueError, match="Unknown parameters: {'invalid_param'}"):
        processor.update_config(invalid_param=0.5)


@pytest.mark.ci
def test_update_config_preserves_state_on_failure():
    """Test that failed config updates don't change the processor state."""
    processor = PropensityScoreProcessor(clipping_threshold=0.1)
    original_config = processor.get_config()

    # Try to update with invalid value
    with pytest.raises(ValueError):
        processor.update_config(clipping_threshold=0.6)

    # Verify state hasn't changed
    assert processor.get_config() == original_config
    assert processor.clipping_threshold == 0.1
