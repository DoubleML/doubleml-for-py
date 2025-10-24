import numpy as np
import pytest

from doubleml.utils.propensity_score_processing import PSProcessor

# -------------------------------------------------------------------------
# Tests for __init__ method
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_init_clipping_threshold_type_error():
    """Test that non-float clipping_threshold raises TypeError."""
    with pytest.raises(TypeError, match="clipping_threshold must be a float."):
        PSProcessor(clipping_threshold="0.01")


@pytest.mark.ci
def test_init_clipping_threshold_value_error():
    """Test that invalid clipping_threshold values raise ValueError."""
    with pytest.raises(ValueError, match="clipping_threshold must be between 0 and 0.5"):
        PSProcessor(clipping_threshold=0.0)  # exactly 0

    with pytest.raises(ValueError, match="clipping_threshold must be between 0 and 0.5"):
        PSProcessor(clipping_threshold=0.6)  # above 0.5


@pytest.mark.ci
def test_init_extreme_threshold_value_error():
    """Test that invalid extreme_threshold values raise ValueError."""
    with pytest.raises(ValueError, match="extreme_threshold must be between 0 and 0.5"):
        PSProcessor(extreme_threshold=0.0)  # exactly 0

    with pytest.raises(ValueError, match="extreme_threshold must be between 0 and 0.5"):
        PSProcessor(extreme_threshold=0.6)  # above 0.5


@pytest.mark.ci
def test_init_calibration_method_value_error():
    """Test that invalid calibration_method raises ValueError."""
    with pytest.raises(ValueError, match="calibration_method must be one of"):
        PSProcessor(calibration_method="invalid_method")


@pytest.mark.ci
def test_init_cv_calibration_type_error():
    """Test that non-bool cv_calibration raises TypeError."""
    with pytest.raises(TypeError, match="cv_calibration must be of bool type."):
        PSProcessor(cv_calibration="True")


@pytest.mark.ci
def test_init_cv_calibration_value_error():
    """Test that cv_calibration True with None calibration_method raises ValueError."""
    with pytest.raises(ValueError, match="cv_calibration=True requires a calibration_method."):
        PSProcessor(calibration_method=None, cv_calibration=True)


# -------------------------------------------------------------------------
# Tests for propensity score & treatment validation
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_validate_propensity_scores_type_error_with_learner():
    """Test TypeError includes learner name."""
    processor = PSProcessor()
    with pytest.raises(TypeError, match="from learner test_learner"):
        processor.adjust_ps([0.1, 0.2], np.array([0, 1]), learner_name="test_learner")


@pytest.mark.ci
def test_validate_propensity_scores_dimension_error():
    """Test that non-1D propensity scores raise ValueError."""
    processor = PSProcessor()
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        processor.adjust_ps(np.array([[0.1, 0.2]]), np.array([0, 1]))


@pytest.mark.ci
def test_validate_propensity_scores_extreme_warning():
    """Test extreme values trigger warnings."""
    processor = PSProcessor(extreme_threshold=0.05)
    with pytest.warns(UserWarning, match="close to zero or one"):
        processor.adjust_ps(np.array([0.01, 0.99]), np.array([0, 1]))


@pytest.mark.ci
def test_validate_treatment_type_error():
    """Test that non-numpy array treatment raises TypeError."""
    processor = PSProcessor()
    with pytest.raises(TypeError, match="Treatment assignments must be of type np.ndarray"):
        processor.adjust_ps(np.array([0.2, 0.8]), [0, 1])


@pytest.mark.ci
def test_validate_treatment_dimension_error():
    """Test that non-1D treatment raises ValueError."""
    processor = PSProcessor()
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        processor.adjust_ps(np.array([0.2, 0.8]), np.array([[0, 1]]))


@pytest.mark.ci
def test_validate_treatment_binary_error():
    """Test that non-binary treatment values raise ValueError."""
    processor = PSProcessor()
    with pytest.raises(ValueError, match="must be binary"):
        processor.adjust_ps(np.array([0.2, 0.8]), np.array([0, 2]))


# -------------------------------------------------------------------------
# Other exception tests
# -------------------------------------------------------------------------


@pytest.mark.ci
def test_apply_calibration_unsupported_method_error():
    """Test that unsupported calibration method raises ValueError."""
    processor = PSProcessor()
    processor._calibration_method = "unsupported_method"

    propensity_scores = np.array([0.2, 0.8])
    treatment = np.array([0, 1])

    with pytest.raises(ValueError, match="Unsupported calibration method: unsupported_method"):
        processor._apply_calibration(propensity_scores, treatment)
