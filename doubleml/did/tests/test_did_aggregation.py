import numpy as np
import pytest

from doubleml.did.did_aggregation import DoubleMLDIDAggregation
from doubleml.double_ml_framework import DoubleMLCore, DoubleMLFramework
from doubleml.tests._utils import generate_dml_dict


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[2, 5])
def n_base_fameworks(request):
    return request.param


@pytest.fixture(scope="module")
def base_framework(n_rep):
    # Create a consistent framework for all tests
    n_obs = 10
    n_thetas = 1

    # Generate consistent scores with known effect
    np.random.seed(42)
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))

    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_core = DoubleMLCore(**doubleml_dict)
    return DoubleMLFramework(dml_core=dml_core)


@pytest.fixture(scope="module", params=["ones", "random", "zeros", "mixed"])
def weight_type(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 4, 5])
def n_aggregations(request):
    return request.param


@pytest.fixture
def weights(n_aggregations, n_base_fameworks, weight_type):
    np.random.seed(42)

    if weight_type == "ones":
        aggregation_weights = np.ones(shape=(n_aggregations, n_base_fameworks))
        overall_aggregation_weights = np.ones(shape=n_aggregations)
    elif weight_type == "random":
        aggregation_weights = np.random.rand(n_aggregations, n_base_fameworks)
        overall_aggregation_weights = np.random.rand(n_aggregations)
    elif weight_type == "zeros":
        aggregation_weights = np.zeros(shape=(n_aggregations, n_base_fameworks))
        overall_aggregation_weights = np.zeros(shape=n_aggregations)
    else:  # mixed
        aggregation_weights = np.ones(shape=(n_aggregations, n_base_fameworks))
        aggregation_weights[::2] = 0.5  # Set every other row to 0.5
        overall_aggregation_weights = np.ones(shape=n_aggregations)
        overall_aggregation_weights[::2] = 0.5

    return aggregation_weights, overall_aggregation_weights


@pytest.mark.ci
def test_multiple_equal_frameworks(base_framework, weights):
    """Test that aggregating the same framework with different weights works correctly"""
    agg_weights, overall_agg_weights = weights

    n_aggregations = agg_weights.shape[0]
    n_frameworks = agg_weights.shape[1]
    # Create list of identical frameworks
    frameworks = [base_framework] * n_frameworks

    # Create aggregation
    aggregation = DoubleMLDIDAggregation(
        frameworks=frameworks, aggregation_weights=agg_weights, overall_aggregation_weights=overall_agg_weights
    )

    # Expected results
    scaled_frameworks = [None] * n_aggregations
    for i_agg in range(n_aggregations):
        scaled_frameworks[i_agg] = sum(agg_weights[i_agg]) * base_framework

        # Check individual aggregation results
        np.testing.assert_allclose(aggregation.aggregated_frameworks.all_thetas[i_agg], scaled_frameworks[i_agg].all_thetas[0])
        np.testing.assert_allclose(
            aggregation.aggregated_frameworks.scaled_psi[:, i_agg, :], scaled_frameworks[i_agg].scaled_psi[:, 0, :]
        )
        # ses might differ due to 1/n and 1/n-1 scaling

    # Check overall aggregation results
    overall_weights = sum([overall_agg_weights[i] * sum(agg_weights[i]) for i in range(n_aggregations)])
    overall_scaled_framework = overall_weights * base_framework

    np.testing.assert_allclose(aggregation.overall_aggregated_framework.all_thetas, overall_scaled_framework.all_thetas)
    np.testing.assert_allclose(aggregation.overall_aggregated_framework.scaled_psi, overall_scaled_framework.scaled_psi)
