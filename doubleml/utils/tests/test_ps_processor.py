import numpy as np
import pytest

from doubleml.utils.propensity_score_processing import PropensityScoreProcessor


@pytest.mark.ci
def test_adjust_basic_clipping():
    """Test basic clipping functionality."""
    processor = PropensityScoreProcessor(clipping_threshold=0.1)

    scores = np.array([0.05, 0.2, 0.8, 0.95])
    treatment = np.array([0, 1, 1, 0])
    adjusted = processor.adjust(scores, treatment)

    expected = np.array([0.1, 0.2, 0.8, 0.9])
    np.testing.assert_array_equal(adjusted, expected)


@pytest.mark.ci
def test_adjust_no_clipping_needed():
    """Test when no clipping is needed."""
    processor = PropensityScoreProcessor(clipping_threshold=0.01)

    scores = np.array([0.2, 0.3, 0.7, 0.8])
    treatment = np.array([0, 1, 1, 0])
    adjusted = processor.adjust(scores, treatment)

    np.testing.assert_array_equal(adjusted, scores)
