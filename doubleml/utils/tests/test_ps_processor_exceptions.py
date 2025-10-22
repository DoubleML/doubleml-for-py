import numpy as np
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


@pytest.mark.ci
def test_update_config_successful_update():
    """Test successful configuration updates."""
    processor = PropensityScoreProcessor(clipping_threshold=0.1)

    processor.update_config(clipping_threshold=0.05)
    assert processor.clipping_threshold == 0.05


@pytest.mark.ci
def test_update_config_defaults():
    """Test updating configuration back to defaults."""
    processor = PropensityScoreProcessor(clipping_threshold=0.1)

    processor.update_config(clipping_threshold=0.01)
    assert processor.clipping_threshold == 0.01

    # Update back to default
    default_config = PropensityScoreProcessor.get_default_config()
    processor.update_config(**default_config)
    assert processor.clipping_threshold == default_config["clipping_threshold"]


# -------------------------------------------------------------------------
# Tests for propensity score validation
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_validate_propensity_scores_type_error_with_learner():
    """Test TypeError includes learner name."""
    processor = PropensityScoreProcessor()
    with pytest.raises(TypeError, match="from learner test_learner"):
        processor.adjust([0.1, 0.2], np.array([0, 1]), learner_name="test_learner")


@pytest.mark.ci
def test_validate_propensity_scores_dimension_error():
    """Test that non-1D propensity scores raise ValueError."""
    processor = PropensityScoreProcessor()
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        processor.adjust(np.array([[0.1, 0.2]]), np.array([0, 1]))


@pytest.mark.ci
def test_validate_propensity_scores_extreme_warning():
    """Test extreme values trigger warnings."""
    processor = PropensityScoreProcessor(extreme_threshold=0.05)
    with pytest.warns(UserWarning, match="close to zero or one"):
        processor.adjust(np.array([0.01, 0.99]), np.array([0, 1]))


@pytest.mark.ci
def test_validate_treatment_type_error():
    """Test that non-numpy array treatment raises TypeError."""
    processor = PropensityScoreProcessor()
    with pytest.raises(TypeError, match="Treatment assignments must be of type np.ndarray"):
        processor.adjust(np.array([0.2, 0.8]), [0, 1])


@pytest.mark.ci
def test_validate_treatment_dimension_error():
    """Test that non-1D treatment raises ValueError."""
    processor = PropensityScoreProcessor()
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        processor.adjust(np.array([0.2, 0.8]), np.array([[0, 1]]))


@pytest.mark.ci
def test_validate_treatment_binary_error():
    """Test that non-binary treatment values raise ValueError."""
    processor = PropensityScoreProcessor()
    with pytest.raises(ValueError, match="must be binary"):
        processor.adjust(np.array([0.2, 0.8]), np.array([0, 2]))
