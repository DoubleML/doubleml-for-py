import numpy as np
import pytest

from doubleml.did.did_aggregation import DoubleMLDIDAggregation
from doubleml.double_ml_framework import DoubleMLCore, DoubleMLFramework
from doubleml.tests._utils import generate_dml_dict


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[1])
def n_thetas(request):
    return request.param


@pytest.fixture
def mock_framework(n_rep, n_thetas):
    # Create a minimal mock of DoubleMLFramework
    n_obs = 10
    # generate score samples
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_core = DoubleMLCore(**doubleml_dict)
    return DoubleMLFramework(dml_core)


@pytest.fixture
def frameworks(mock_framework):
    # Create a list of 3 frameworks
    return [mock_framework] * 3


@pytest.fixture
def aggregation_weights():
    # Create sample weights for 2 aggregations over 3 frameworks
    return np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])


@pytest.mark.ci
def test_valid_initialization(frameworks, aggregation_weights):
    # Test initialization with valid parameters
    aggregation = DoubleMLDIDAggregation(
        frameworks=frameworks,
        aggregation_weights=aggregation_weights,
        overall_aggregation_weights=np.array([0.6, 0.4]),
        aggregation_names=["agg1", "agg2"],
        aggregation_method_name="custom",
        additional_information={"key": "value"},
    )
    assert isinstance(aggregation.base_frameworks, list)
    assert isinstance(aggregation.aggregation_weights, np.ndarray)
    assert isinstance(aggregation.additional_information, str)


@pytest.mark.ci
def test_invalid_frameworks(aggregation_weights):
    # Test with invalid frameworks type
    with pytest.raises(TypeError, match="The 'frameworks' must be a list of DoubleMLFramework objects"):
        DoubleMLDIDAggregation(frameworks="invalid_frameworks", aggregation_weights=aggregation_weights)


@pytest.mark.ci
def test_invalid_framework_dim():
    psi_a = np.ones(shape=(10, 2, 1))
    psi_b = np.random.normal(size=(10, 2, 1))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_core = DoubleMLCore(**doubleml_dict)
    framework = DoubleMLFramework(dml_core=dml_core)

    # Test with invalid framework dimension
    with pytest.raises(ValueError, match="All frameworks must be one-dimensional"):
        DoubleMLDIDAggregation(frameworks=[framework, framework], aggregation_weights=np.array([[0.5, 0.5], [0.3, 0.7]]))


@pytest.mark.ci
def test_invalid_aggregation_weights(frameworks):
    # Test with invalid aggregation_weights type
    with pytest.raises(TypeError, match="'aggregation_weights' must be a numpy array"):
        DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=[1, 2, 3])  # list instead of numpy array


@pytest.mark.ci
def test_invalid_aggregation_weights_ndim(frameworks):
    # Test with 1D array instead of 2D
    with pytest.raises(ValueError, match="'aggregation_weights' must be a 2-dimensional array"):
        DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=np.array([0.5, 0.3, 0.2]))


@pytest.mark.ci
def test_invalid_aggregation_weights_shape(frameworks):
    # Test with wrong number of columns
    with pytest.raises(
        ValueError, match="The number of rows in 'aggregation_weights' must be equal to the number of frameworks"
    ):
        DoubleMLDIDAggregation(
            frameworks=frameworks, aggregation_weights=np.array([[0.5, 0.5], [0.3, 0.7]])  # Only 2 columns for 3 frameworks
        )


@pytest.mark.ci
def test_invalid_overall_aggregation_weights(frameworks, aggregation_weights):
    # Test with invalid overall_aggregation_weights type
    with pytest.raises(TypeError, match="'overall_aggregation_weights' must be a numpy array"):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            overall_aggregation_weights=[0.5, 0.5],  # list instead of numpy array
        )


@pytest.mark.ci
def test_invalid_overall_weights_ndim(frameworks, aggregation_weights):
    # Test with 2D array instead of 1D
    with pytest.raises(ValueError, match="'overall_aggregation_weights' must be a 1-dimensional array"):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            overall_aggregation_weights=np.array([[0.5], [0.5]]),
        )


@pytest.mark.ci
def test_invalid_overall_weights_length(frameworks, aggregation_weights):
    # Test with wrong length
    with pytest.raises(
        ValueError, match="'overall_aggregation_weights' must have the same length as the number of aggregated frameworks"
    ):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            overall_aggregation_weights=np.array([0.5, 0.3, 0.2]),  # 3 weights for 2 aggregations
        )


@pytest.mark.ci
def test_invalid_aggregation_names_type(frameworks, aggregation_weights):
    # Test with non-list type
    with pytest.raises(TypeError, match="'aggregation_names' must be a list of strings"):
        DoubleMLDIDAggregation(
            frameworks=frameworks, aggregation_weights=aggregation_weights, aggregation_names="invalid_names"
        )


@pytest.mark.ci
def test_invalid_aggregation_names_content(frameworks, aggregation_weights):
    # Test with non-string elements
    with pytest.raises(TypeError, match="'aggregation_names' must be a list of strings"):
        DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights, aggregation_names=[1, 2])


@pytest.mark.ci
def test_invalid_aggregation_names_length(frameworks, aggregation_weights):
    # Test with wrong length
    with pytest.raises(ValueError, match="'aggregation_names' must have the same length as the number of aggregations"):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            aggregation_names=["agg1"],  # Only 1 name for 2 aggregations
        )


@pytest.mark.ci
def test_invalid_method_name_type(frameworks, aggregation_weights):
    # Test with non-string type
    with pytest.raises(TypeError, match="'aggregation_method_name' must be a string"):
        DoubleMLDIDAggregation(frameworks=frameworks, aggregation_weights=aggregation_weights, aggregation_method_name=123)


@pytest.mark.ci
def test_invalid_additional_information(frameworks, aggregation_weights):
    # Test with invalid additional_information type
    with pytest.raises(TypeError, match="'additional_information' must be a dictionary"):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            additional_information=[1, 2, 3],  # list instead of dict
        )


@pytest.mark.ci
def test_additional_parameters(frameworks, aggregation_weights):
    # Test with invalid additional_parameters type
    with pytest.raises(TypeError, match="'additional_parameters' must be a dictionary"):
        DoubleMLDIDAggregation(
            frameworks=frameworks,
            aggregation_weights=aggregation_weights,
            additional_parameters=[1, 2, 3],  # list instead of dict
        )
