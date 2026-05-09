"""Tests for LearnerSpec validation in doubleml.utils._learner."""

import pytest

from doubleml.utils._learner import LearnerSpec


@pytest.mark.ci
def test_learner_spec_requires_regressor_or_classifier():
    """LearnerSpec must have at least one of allow_regressor / allow_classifier set to True."""
    msg = r"LearnerSpec 'ml_x': at least one of allow_regressor or allow_classifier must be True\."
    with pytest.raises(ValueError, match=msg):
        LearnerSpec("ml_x", allow_regressor=False, allow_classifier=False)
