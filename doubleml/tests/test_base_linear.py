import pytest
import numpy as np

from doubleml.double_ml_base_linear import DoubleMLBaseLinear


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=['mean', 'median'])
def aggregation_method(request):
    return request.param


@pytest.fixture(scope='module')
def dml_base_linear_fixture(n_rep, aggregation_method):
    n_obs = 100
    psi_elements = {
        'psi_a': np.ones(shape=(n_obs, 1, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, 1, n_rep)),
    }

    dml_base_linear_obj = DoubleMLBaseLinear(psi_elements, n_rep=n_rep)
    dml_base_linear_obj.estimate_thetas(aggregation_method=aggregation_method)

    expected_thetas = -1.0 * np.mean(psi_elements['psi_b'], axis=0)
    expected_vars = np.square(expected_thetas + psi_elements['psi_b']).mean(axis=0) / n_obs
    expected_ses = np.sqrt(expected_vars)

    result_dict = {
        'dml_base_linear_obj': dml_base_linear_obj,
        'expected_thetas': expected_thetas,
        'expected_ses': expected_ses,

    }
    return result_dict


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_base_linear_theta(dml_base_linear_fixture):
    assert np.allclose(
        dml_base_linear_fixture['dml_base_linear_obj'].all_thetas,
        dml_base_linear_fixture['expected_thetas']
    )


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_base_linear_se(dml_base_linear_fixture):
    assert np.allclose(
        dml_base_linear_fixture['dml_base_linear_obj'].all_ses,
        dml_base_linear_fixture['expected_ses']
    )
