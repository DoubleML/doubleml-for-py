import pytest
import numpy as np

from doubleml.double_ml_basic_linear import DoubleMLBasicLinear


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_basic_linear_fixture(n_rep):
    n_obs = 100
    psi_elements = {
        'psi_a': np.ones(shape=(n_obs, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, n_rep)),
    }

    dml_basic_linear_obj = DoubleMLBasicLinear(psi_elements)
    dml_basic_linear_obj.estimate_theta()

    expected_thetas = -1.0 * np.mean(psi_elements['psi_b'], axis=0)

    result_dict = {
        'dml_basic_linear_obj': dml_basic_linear_obj,
        'expected_thetas': expected_thetas,

    }

    return result_dict


@pytest.mark.ci
def test_dml_basic_linear_estimation(dml_basic_linear_fixture):
    assert np.allclose(
        dml_basic_linear_fixture['dml_basic_linear_obj'].all_thetas,
        dml_basic_linear_fixture['expected_thetas']
    )
    assert np.allclose(
        dml_basic_linear_fixture['dml_basic_linear_obj'].theta,
        np.median(dml_basic_linear_fixture['expected_thetas'])
    )
