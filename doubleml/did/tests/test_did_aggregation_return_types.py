import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def frameworks(mock_framework):
    # Create a list of 3 frameworks
    return [mock_framework] * 3


@pytest.fixture
def aggregation_weights():
    # Create sample weights for 2 aggregations over 3 frameworks
    return np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])


@pytest.mark.ci
@pytest.mark.parametrize(
    "property_name,expected_value",
    [
        ("overall_aggregation_weights", lambda w: np.array([0.5, 0.5])),  # Equal weights for 2 aggregations
        ("aggregation_names", lambda w: ["Aggregation_0", "Aggregation_1"]),
        ("aggregation_method_name", lambda w: "Custom"),
        ("additional_information", lambda w: None),
        ("additional_parameters", lambda w: None),
    ],
)
def test_default_values(frameworks, aggregation_weights, property_name, expected_value):
    # Test that default values are correctly set when not explicitly provided
    aggregation = DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights)

    expected = expected_value(aggregation_weights)
    actual = getattr(aggregation, property_name)

    if property_name == "overall_aggregation_weights":
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.ci
def test_custom_aggregation_values(frameworks, aggregation_weights):
    # Test all custom values are properly set when provided
    custom_names = ["Custom1", "Custom2"]
    custom_method = "MyMethod"
    custom_overall_weights = np.array([0.7, 0.3])
    custom_info = {"info": "test"}
    custom_params = {"param": 123}

    aggregation = DoubleMLDIDAggregation(
        frameworks=frameworks,
        aggregation_weights=aggregation_weights,
        overall_aggregation_weights=custom_overall_weights,
        aggregation_names=custom_names,
        aggregation_method_name=custom_method,
        additional_information=custom_info,
        additional_parameters=custom_params,
    )

    assert aggregation.aggregation_names == custom_names
    assert aggregation.aggregation_method_name == custom_method
    np.testing.assert_array_equal(aggregation.overall_aggregation_weights, custom_overall_weights)
    assert "info: test" in aggregation.additional_information
    assert aggregation.additional_parameters == custom_params


@pytest.mark.ci
@pytest.mark.parametrize(
    "property_name,expected_type",
    [
        ("base_frameworks", list),
        ("aggregated_frameworks", DoubleMLFramework),
        ("overall_aggregated_framework", DoubleMLFramework),
        ("aggregation_weights", np.ndarray),
        ("overall_aggregation_weights", np.ndarray),
        ("n_aggregations", int),
        ("aggregation_names", list),
        ("aggregation_method_name", str),
        ("aggregated_summary", pd.DataFrame),
        ("overall_summary", pd.DataFrame),
    ],
)
def test_return_types(frameworks, aggregation_weights, property_name, expected_type):
    # Test that properties return the expected types
    aggregation = DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights)

    value = getattr(aggregation, property_name)
    assert isinstance(value, expected_type)


@pytest.mark.ci
def test_additional_info_return_types(frameworks, aggregation_weights):
    # Test additional_information and additional_parameters return types

    # Test when None
    aggregation1 = DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights)
    assert aggregation1.additional_information is None
    assert aggregation1.additional_parameters is None

    # Test when provided
    aggregation2 = DoubleMLDIDAggregation(
        frameworks=frameworks,
        aggregation_weights=aggregation_weights,
        additional_information={"info": "value"},
        additional_parameters={"param": "value"},
    )
    assert isinstance(aggregation2.additional_information, str)
    assert isinstance(aggregation2.additional_parameters, dict)


@pytest.mark.ci
def test_str_representation(frameworks, aggregation_weights):
    # Test string representation without additional information
    aggregation1 = DoubleMLDIDAggregation(
        frameworks=frameworks, aggregation_weights=aggregation_weights, aggregation_method_name="TestMethod"
    )
    str_output = str(aggregation1)

    # Check presence of all required sections
    assert "================== DoubleMLDIDAggregation Object ==================" in str_output
    assert "TestMethod Aggregation" in str_output
    assert "------------------ Overall Aggregated Effects ------------------" in str_output
    assert "------------------ Aggregated Effects         ------------------" in str_output
    assert "------------------ Additional Information     ------------------" not in str_output

    # Test string representation with additional information
    aggregation2 = DoubleMLDIDAggregation(
        frameworks=frameworks,
        aggregation_weights=aggregation_weights,
        aggregation_method_name="TestMethod",
        additional_information={"key": "value"},
    )
    str_output_with_info = str(aggregation2)

    # Check additional information section
    assert "------------------ Additional Information     ------------------" in str_output_with_info
    assert "key: value" in str_output_with_info


@pytest.mark.ci
def test_plot_effects_return_type(frameworks, aggregation_weights):
    """Test that plot_effects method returns matplotlib Figure and Axes objects."""
    aggregation = DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights)
    aggregation.aggregated_frameworks.bootstrap(n_rep_boot=10)

    # Test basic call without parameters
    fig, ax = aggregation.plot_effects()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    # Test with parameters
    fig, ax = aggregation.plot_effects(
        level=0.9,
        joint=False,
        figsize=(10, 5),
        sort_by="estimate",
        color_palette="Set2",
        title="Custom Title",
        y_label="Custom Y-Label",
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)
