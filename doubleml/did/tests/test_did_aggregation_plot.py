import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from doubleml.did.did_aggregation import DoubleMLDIDAggregation
from doubleml.double_ml_framework import DoubleMLFramework
from doubleml.tests._utils import generate_dml_dict


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture
def mock_framework(n_rep):
    # Create a minimal mock of DoubleMLFramework
    n_obs = 10
    n_thetas = 1
    # generate score samples
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    return DoubleMLFramework(doubleml_dict)


@pytest.fixture
def simple_aggregation(mock_framework):
    """Create a simple DoubleMLDIDAggregation object for testing."""
    # Get two framework instances
    fw1 = mock_framework
    fw2 = mock_framework

    # Set treatment names (important for the test)
    fw1.treatment_names = ["Treatment 1"]
    fw2.treatment_names = ["Treatment 2"]

    # Weights for aggregation
    agg_weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    overall_weights = np.array([0.7, 0.3])

    agg_obj = DoubleMLDIDAggregation(
        frameworks=[fw1, fw2],
        aggregation_weights=agg_weights,
        overall_aggregation_weights=overall_weights,
        aggregation_names=["Group A", "Group B"],
        aggregation_method_name="Test Method",
        additional_information={"Test Info": "Value"},
        additional_parameters={"aggregation_color_idx": [0, 1]},
    )

    agg_obj.aggregated_frameworks.bootstrap(n_rep_boot=10)
    return agg_obj


@pytest.mark.ci
def test_plot_effects_returns_fig_ax(simple_aggregation):
    """Test that plot_effects returns figure and axes objects."""
    fig, ax = simple_aggregation.plot_effects()

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close("all")


@pytest.mark.ci
def test_plot_effects_invalid_sort_by(simple_aggregation):
    """Test that invalid sort_by values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid sort_by value"):
        simple_aggregation.plot_effects(sort_by="invalid")

    # These should not raise
    for valid_value in ["name", "estimate", None]:
        _ = simple_aggregation.plot_effects(sort_by=valid_value)

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_sorting(simple_aggregation):
    """Test that sorting works correctly."""
    # Get the dataframe that would be created inside the method
    df = simple_aggregation._create_ci_dataframe()

    # Test name sorting
    _, ax = simple_aggregation.plot_effects(sort_by="name")
    labels = [text.get_text() for text in ax.get_xticklabels()]
    expected = sorted(df["Aggregation_Names"])
    assert labels == expected

    # Test estimate sorting
    _, ax = simple_aggregation.plot_effects(sort_by="estimate")
    labels = [text.get_text() for text in ax.get_xticklabels()]
    expected = df.sort_values("Estimate", ascending=False)["Aggregation_Names"].tolist()
    assert labels == expected

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_elements(simple_aggregation):
    """Test that the plot contains expected elements."""
    _, ax = simple_aggregation.plot_effects(title="Test Title", y_label="Test Label")

    # Check title and y-label
    assert ax.get_title() == "Test Title"
    assert ax.get_ylabel() == "Test Label"

    # Check that we have the zero line
    zero_lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert len(zero_lines) == 1

    # Check we have scatter points for estimates
    assert len(ax.collections) > 0

    # Check we have the correct number of x-ticks
    assert len(ax.get_xticks()) == 2  # We have 2 groups in our fixture

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_custom_figsize(simple_aggregation):
    """Test that figsize parameter works."""
    custom_figsize = (8, 4)
    fig, _ = simple_aggregation.plot_effects(figsize=custom_figsize)

    # Convert to inches for comparison (matplotlib uses inches)
    width, height = fig.get_size_inches()
    assert (width, height) == custom_figsize

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_custom_colors(simple_aggregation):
    """Test that color_palette parameter works."""
    # Custom color list
    custom_colors = [(1, 0, 0), (0, 1, 0)]  # Red and green
    _, _ = simple_aggregation.plot_effects(color_palette=custom_colors)
    plt.close("all")

    # Named palette
    _, _ = simple_aggregation.plot_effects(color_palette="Set1")
    plt.close("all")


@pytest.mark.ci
def test_joint_ci_bootstrap_warning(mock_framework):
    """Test that requesting joint confidence intervals without bootstrapping issues a warning."""
    # Create a new aggregation object without bootstrapping
    fw1 = mock_framework
    fw2 = mock_framework

    # Set treatment names
    fw1.treatment_names = ["Treatment 1"]
    fw2.treatment_names = ["Treatment 2"]

    # Weights for aggregation
    agg_weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    overall_weights = np.array([0.7, 0.3])

    # Create aggregation without bootstrapping
    aggregation = DoubleMLDIDAggregation(
        frameworks=[fw1, fw2],
        aggregation_weights=agg_weights,
        overall_aggregation_weights=overall_weights,
        aggregation_names=["Group A", "Group B"],
        additional_parameters={"aggregation_color_idx": [0, 1]},
    )

    # Ensure no bootstrapping exists
    aggregation.aggregated_frameworks._boot_t_stat = None

    # Check that a warning is raised with the expected message
    with pytest.warns(UserWarning, match="Joint confidence intervals require bootstrapping"):
        _ = aggregation.plot_effects(joint=True)

    plt.close("all")
