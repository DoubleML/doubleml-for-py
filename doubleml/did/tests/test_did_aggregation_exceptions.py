import pytest
import numpy as np
from doubleml.double_ml_framework import DoubleMLFramework
from doubleml.did.did_aggregation import DoubleMLDIDAggregation

from doubleml.tests._utils import generate_dml_dict


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 5])
def n_thetas(request):
    return request.param


@pytest.fixture
def mock_framework(n_rep, n_thetas):
    # Create a minimal mock of DoubleMLFramework
    n_obs = 100

    # generate score samples
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_framework_obj = DoubleMLFramework(doubleml_dict)

    return dml_framework_obj


@pytest.fixture
def weight_masks():
    # Create a sample masked array
    return np.ma.masked_array([1, 2, 3], mask=[False, False, False])


@pytest.mark.ci
def test_valid_initialization(mock_framework, weight_masks):
    # Test initialization with valid parameters
    aggregation = DoubleMLDIDAggregation(
        mock_framework,
        mock_framework,
        weight_masks,
        additional_information={"key": "value"}
    )
    assert isinstance(aggregation._aggregated_frameworks, DoubleMLFramework)
    assert isinstance(aggregation._overall_aggregated_framework, DoubleMLFramework)
    assert isinstance(aggregation._weight_masks, np.ma.MaskedArray)
    assert isinstance(aggregation._additional_information, dict)


@pytest.mark.ci
def test_invalid_aggregated_frameworks(weight_masks, mock_framework):
    # Test with invalid aggregated_frameworks type
    with pytest.raises(TypeError, match="'aggregated_frameworks' must be an instance of DoubleMLFramework"):
        DoubleMLDIDAggregation(
            "invalid_framework",
            mock_framework,
            weight_masks
        )


@pytest.mark.ci
def test_invalid_overall_aggregated_framework(weight_masks, mock_framework):
    # Test with invalid overall_aggregated_framework type
    with pytest.raises(TypeError, match="'overall_aggregated_framework' must be an instance of DoubleMLFramework"):
        DoubleMLDIDAggregation(
            mock_framework,
            "invalid_framework",
            weight_masks
        )


@pytest.mark.ci
def test_invalid_weight_masks(mock_framework):
    # Test with invalid weight_masks type
    with pytest.raises(TypeError, match="'weight_masks' must be an instance of np.ma.MaskedArray"):
        DoubleMLDIDAggregation(
            mock_framework,
            mock_framework,
            [1, 2, 3]  # regular list instead of masked array
        )


@pytest.mark.ci
def test_invalid_additional_information(mock_framework, weight_masks):
    # Test with invalid additional_information type
    with pytest.raises(TypeError, match="'additional_information' must be a dictionary"):
        DoubleMLDIDAggregation(
            mock_framework,
            mock_framework,
            weight_masks,
            additional_information=[1, 2, 3]  # list instead of dict
        )
