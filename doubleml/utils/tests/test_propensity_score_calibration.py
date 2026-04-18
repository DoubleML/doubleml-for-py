import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from doubleml.utils.plots import plot_propensity_score_calibration


@pytest.mark.ci
class TestInputValidation:
    """Test input validation for plot_propensity_score_calibration."""

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            plot_propensity_score_calibration(np.array([0.5, 0.3]), np.array([0, 1, 0]))

    def test_non_binary_treatment(self):
        with pytest.raises(ValueError, match="binary with values 0 and 1"):
            plot_propensity_score_calibration(np.array([0.5, 0.3, 0.7]), np.array([0, 1, 2]))

    def test_scores_below_zero(self):
        with pytest.raises(ValueError, match="must lie in"):
            plot_propensity_score_calibration(np.array([-0.1, 0.5]), np.array([0, 1]))

    def test_scores_above_one(self):
        with pytest.raises(ValueError, match="must lie in"):
            plot_propensity_score_calibration(np.array([0.5, 1.1]), np.array([0, 1]))

    def test_bins_too_few(self):
        with pytest.raises(ValueError, match="bins must be at least 2"):
            plot_propensity_score_calibration(np.array([0.5, 0.3]), np.array([0, 1]), bins=1)

    def test_bins_array_too_short(self):
        with pytest.raises(ValueError, match="at least two edges"):
            plot_propensity_score_calibration(np.array([0.5, 0.3]), np.array([0, 1]), bins=np.array([0.5]))

    def test_bins_not_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            plot_propensity_score_calibration(np.array([0.5, 0.3]), np.array([0, 1]), bins=np.array([0.5, 0.3, 0.8]))


@pytest.mark.ci
class TestReturnType:
    """Test return type and basic plot structure."""

    def test_returns_figure_and_axes(self):
        ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
        tr = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_density_mode(self):
        ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
        tr = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr, density=True)
        assert axes[0, 0].get_ylabel() == "Density"
        plt.close(fig)

    def test_count_mode(self):
        ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
        tr = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr, density=False)
        assert axes[0, 0].get_ylabel() == "Count"
        plt.close(fig)


@pytest.mark.ci
class TestBinHandling:
    """Test bin handling with int and array bins."""

    def test_int_bins(self):
        np.random.seed(42)
        ps = np.random.uniform(0, 1, 100)
        tr = (ps > 0.5).astype(int)
        fig, axes = plot_propensity_score_calibration(ps, tr, bins=5)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_explicit_bins(self):
        np.random.seed(42)
        ps = np.random.uniform(0, 1, 100)
        tr = (ps > 0.5).astype(int)
        custom_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        fig, axes = plot_propensity_score_calibration(ps, tr, bins=custom_bins)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


@pytest.mark.ci
class TestBoundaryValues:
    """Test boundary values at 0 and 1."""

    def test_all_scores_at_zero(self):
        ps = np.zeros(10)
        tr = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_all_scores_at_one(self):
        ps = np.ones(10)
        tr = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_scores_at_bin_edges(self):
        ps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        tr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr, bins=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


@pytest.mark.ci
class TestEmptyBinBehavior:
    """Test empty-bin behavior (NaN in calibration)."""

    def test_empty_bins_do_not_crash(self):
        # Only scores in [0.4, 0.6], so bins outside that range are empty
        ps = np.array([0.45, 0.5, 0.55, 0.5, 0.45, 0.55])
        tr = np.array([0, 1, 1, 0, 0, 1])
        fig, axes = plot_propensity_score_calibration(ps, tr, bins=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


@pytest.mark.ci
class TestCalibrationContent:
    """Test that calibration subplots have expected properties."""

    def test_calibration_axes_labels(self):
        np.random.seed(42)
        ps = np.random.uniform(0, 1, 200)
        tr = (np.random.uniform(0, 1, 200) < ps).astype(int)
        fig, axes = plot_propensity_score_calibration(ps, tr)

        assert axes[1, 0].get_xlabel() == "Predicted propensity score"
        assert axes[1, 0].get_ylabel() == "Observed treatment fraction"
        assert axes[1, 1].get_xlabel() == "Predicted propensity score"
        assert axes[1, 1].get_ylabel() == "Observed control fraction"
        plt.close(fig)

    def test_titles(self):
        np.random.seed(42)
        ps = np.random.uniform(0, 1, 200)
        tr = (np.random.uniform(0, 1, 200) < ps).astype(int)
        fig, axes = plot_propensity_score_calibration(ps, tr)

        assert axes[0, 0].get_title() == "Treated: Propensity Score Distribution"
        assert axes[0, 1].get_title() == "Control: Propensity Score Distribution"
        assert axes[1, 0].get_title() == "Treated: Calibration"
        assert axes[1, 1].get_title() == "Control: Calibration"
        plt.close(fig)

    def test_suptitle(self):
        np.random.seed(42)
        ps = np.random.uniform(0, 1, 50)
        tr = (np.random.uniform(0, 1, 50) < ps).astype(int)
        fig, axes = plot_propensity_score_calibration(ps, tr)
        assert fig._suptitle.get_text() == "Propensity Score Calibration"
        plt.close(fig)
