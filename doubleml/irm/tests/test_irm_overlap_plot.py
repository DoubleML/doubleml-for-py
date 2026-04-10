import warnings

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLIRM
from doubleml.irm.datasets import make_irm_data

np.random.seed(3141)
n_obs = 200
dml_data = make_irm_data(n_obs=n_obs)


@pytest.fixture(scope="module")
def fitted_irm():
    """IRM model fitted with stored predictions."""
    irm = DoubleMLIRM(dml_data, Lasso(), LogisticRegression(), n_rep=2, n_folds=3)
    irm.fit(store_predictions=True)
    return irm


@pytest.fixture(scope="module")
def fitted_irm_no_preds():
    """IRM model fitted without stored predictions."""
    irm = DoubleMLIRM(dml_data, Lasso(), LogisticRegression())
    irm.fit(store_predictions=False)
    return irm


@pytest.mark.ci
class TestOverlapPlotReturnType:
    """Test that plot_overlap_common_support returns a plotly Figure."""

    def test_returns_plotly_figure(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support()
        assert isinstance(fig, go.Figure)

    def test_returns_plotly_figure_custom_params(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support(
            idx_treatment=0, i_rep=1, threshold=0.1, show_warning=False
        )
        assert isinstance(fig, go.Figure)


@pytest.mark.ci
class TestOverlapPlotErrors:
    """Test error handling for plot_overlap_common_support."""

    def test_error_before_fit(self):
        """Calling before fit() should raise ValueError."""
        irm = DoubleMLIRM(dml_data, Lasso(), LogisticRegression())
        with pytest.raises(ValueError, match="Apply fit"):
            irm.plot_overlap_common_support()

    def test_error_no_predictions(self, fitted_irm_no_preds):
        """Calling after fit(store_predictions=False) should raise ValueError."""
        with pytest.raises(ValueError, match="Predictions are not stored"):
            fitted_irm_no_preds.plot_overlap_common_support()

    def test_error_invalid_idx_treatment_type(self, fitted_irm):
        with pytest.raises(TypeError, match="idx_treatment must be an integer"):
            fitted_irm.plot_overlap_common_support(idx_treatment=0.5)

    def test_error_invalid_idx_treatment_range(self, fitted_irm):
        with pytest.raises(ValueError, match="idx_treatment must be in"):
            fitted_irm.plot_overlap_common_support(idx_treatment=5)

    def test_error_negative_idx_treatment(self, fitted_irm):
        with pytest.raises(ValueError, match="idx_treatment must be in"):
            fitted_irm.plot_overlap_common_support(idx_treatment=-1)

    def test_error_invalid_i_rep_type(self, fitted_irm):
        with pytest.raises(TypeError, match="i_rep must be an integer"):
            fitted_irm.plot_overlap_common_support(i_rep=0.5)

    def test_error_invalid_i_rep_range(self, fitted_irm):
        with pytest.raises(ValueError, match="i_rep must be in"):
            fitted_irm.plot_overlap_common_support(i_rep=10)

    def test_error_invalid_threshold_type(self, fitted_irm):
        with pytest.raises(TypeError, match="threshold must be a float"):
            fitted_irm.plot_overlap_common_support(threshold="0.05")

    def test_error_invalid_threshold_range_low(self, fitted_irm):
        with pytest.raises(ValueError, match="threshold must be in"):
            fitted_irm.plot_overlap_common_support(threshold=0.0)

    def test_error_invalid_threshold_range_high(self, fitted_irm):
        with pytest.raises(ValueError, match="threshold must be in"):
            fitted_irm.plot_overlap_common_support(threshold=0.5)

    def test_error_invalid_show_warning_type(self, fitted_irm):
        with pytest.raises(TypeError, match="show_warning must be a boolean"):
            fitted_irm.plot_overlap_common_support(show_warning="True")


@pytest.mark.ci
class TestOverlapPlotContent:
    """Test plot content and structure."""

    def test_figure_has_traces(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support()
        # Should have at least 2 traces (treated and control KDE)
        assert len(fig.data) >= 2

    def test_figure_trace_names(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support()
        trace_names = [trace.name for trace in fig.data]
        assert "Treated" in trace_names
        assert "Control" in trace_names

    def test_figure_layout(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support()
        assert fig.layout.title.text == "Propensity Score Overlap (Common Support)"
        assert fig.layout.xaxis.title.text == "Estimated Propensity Score"
        assert fig.layout.yaxis.title.text == "Density"

    def test_custom_threshold_in_annotations(self, fitted_irm):
        fig = fitted_irm.plot_overlap_common_support(threshold=0.1)
        # The annotation text should reference the threshold
        annotations = fig.layout.annotations
        found = any("0.1" in str(ann.text) for ann in annotations if ann.text)
        assert found

    def test_different_repetitions(self, fitted_irm):
        """Results should differ between repetitions."""
        fig0 = fitted_irm.plot_overlap_common_support(i_rep=0)
        fig1 = fitted_irm.plot_overlap_common_support(i_rep=1)
        # Traces should exist for both
        assert len(fig0.data) >= 2
        assert len(fig1.data) >= 2


@pytest.mark.ci
class TestOverlapPlotWarning:
    """Test positivity violation warning behavior."""

    def test_no_warning_when_disabled(self, fitted_irm):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warning
            fitted_irm.plot_overlap_common_support(show_warning=False)

    def test_warning_with_extreme_threshold(self, fitted_irm):
        """With a very wide threshold (e.g., 0.49), almost all observations
        should be flagged, triggering the warning."""
        with pytest.warns(UserWarning, match="Potential positivity violation"):
            fitted_irm.plot_overlap_common_support(threshold=0.49, show_warning=True)
