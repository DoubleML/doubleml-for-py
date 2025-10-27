import warnings
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict

from doubleml.utils.propensity_score_processing import PSProcessor, PSProcessorConfig, init_ps_processor


# TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
@pytest.mark.ci
def test_init_ps_processor_with_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg, proc = init_ps_processor(None, "truncate", 0.02)
        assert any("deprecated" in str(warn.message) for warn in w)
        assert isinstance(cfg, PSProcessorConfig)
        assert proc.clipping_threshold == 0.02


@pytest.mark.ci
def test_init_ps_processor_with_config():
    config = PSProcessorConfig(clipping_threshold=0.05)
    cfg, proc = init_ps_processor(config, None, None)
    assert isinstance(cfg, PSProcessorConfig)
    assert isinstance(proc, PSProcessor)
    assert proc.clipping_threshold == 0.05


@pytest.mark.ci
def test_from_config_initialization():
    """Test initialization of PSProcessor from PSProcessorConfig."""
    config = PSProcessorConfig(
        clipping_threshold=0.05,
        extreme_threshold=1e-8,
        calibration_method="isotonic",
        cv_calibration=True,
    )
    processor = PSProcessor.from_config(config)
    assert processor.clipping_threshold == 0.05
    assert processor.extreme_threshold == 1e-8
    assert processor.calibration_method == "isotonic"
    assert processor.cv_calibration is True


@pytest.mark.ci
def test_adjust_basic_clipping():
    """Test basic clipping functionality."""
    processor = PSProcessor(clipping_threshold=0.1)

    scores = np.array([0.05, 0.2, 0.8, 0.95])
    treatment = np.array([0, 1, 1, 0])
    adjusted = processor.adjust_ps(scores, treatment)

    expected = np.array([0.1, 0.2, 0.8, 0.9])
    np.testing.assert_array_equal(adjusted, expected)


@pytest.mark.ci
def test_adjust_no_clipping_needed():
    """Test when no clipping is needed."""
    processor = PSProcessor(clipping_threshold=0.01)

    scores = np.array([0.2, 0.3, 0.7, 0.8])
    treatment = np.array([0, 1, 1, 0])
    adjusted = processor.adjust_ps(scores, treatment)

    np.testing.assert_array_equal(adjusted, scores)


@pytest.mark.ci
def test_isotonic_calibration_without_cv():
    """Test isotonic calibration without cross-validation."""
    ps = np.random.uniform(0, 1, size=100)
    treatment = np.random.binomial(1, 0.5, size=100)

    clipping_threshold = 0.01
    processor = PSProcessor(
        calibration_method="isotonic",
        cv_calibration=False,
        clipping_threshold=clipping_threshold,
    )

    isotonic_manual = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_manual.fit(ps.reshape(-1, 1), treatment)
    expected_ps_manual = isotonic_manual.predict(ps.reshape(-1, 1))
    expected_ps_manual = np.clip(expected_ps_manual, clipping_threshold, 1 - clipping_threshold)

    adjusted_ps = processor.adjust_ps(ps, treatment)
    np.testing.assert_array_equal(adjusted_ps, expected_ps_manual)


@pytest.fixture(scope="module", params=[None, 3, "iterable", "splitter"])
def cv(request):
    return request.param


@pytest.mark.ci
def test_isotonic_calibration_with_cv(cv):
    """Test isotonic calibration with cross-validation."""
    n_obs = 100
    ps = np.random.uniform(0, 1, size=n_obs)
    treatment = np.random.binomial(1, 0.5, size=n_obs)
    if cv == "iterable":
        cv = [(train, test) for train, test in KFold(n_splits=3).split(ps)]
    elif cv == "splitter":
        cv = KFold(n_splits=3)

    clipping_threshold = 0.01
    processor = PSProcessor(calibration_method="isotonic", cv_calibration=True, clipping_threshold=clipping_threshold)

    isotonic_manual = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ps_cv = cross_val_predict(isotonic_manual, ps.reshape(-1, 1), treatment, cv=cv)
    expected_ps_manual = np.clip(ps_cv, clipping_threshold, 1 - clipping_threshold)

    adjusted_ps = processor.adjust_ps(ps, treatment, cv=cv)
    np.testing.assert_array_equal(adjusted_ps, expected_ps_manual)


@pytest.mark.ci
def test_no_calibration():
    """Test that no calibration is applied when calibration_method is None."""
    processor = PSProcessor(calibration_method=None, clipping_threshold=0.01)

    scores = np.array([0.2, 0.3, 0.7, 0.8])
    treatment = np.array([0, 1, 1, 0])

    # Should not call any calibration methods
    with patch("sklearn.isotonic.IsotonicRegression") as mock_isotonic:
        adjusted = processor.adjust_ps(scores, treatment)
        mock_isotonic.assert_not_called()

    np.testing.assert_array_equal(adjusted, scores)
