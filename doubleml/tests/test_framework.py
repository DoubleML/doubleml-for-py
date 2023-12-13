import pytest
import numpy as np

from doubleml.double_ml_base_linear import DoubleMLBaseLinear
from doubleml.double_ml_framework import DoubleMLFramework


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_framework_fixture(n_rep):
    n_obs = 100
    psi_elements = {
        'psi_a': np.ones(shape=(n_obs, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, n_rep)),
    }
    dml_obj_1 = DoubleMLBaseLinear(psi_elements)
    dml_obj_2 = DoubleMLBaseLinear(psi_elements)

    dml_framework_obj = DoubleMLFramework([dml_obj_1, dml_obj_2])
    dml_framework_obj.estimate_thetas()

    expected_thetas = -1.0 * np.mean(psi_elements['psi_b'], axis=0)
    expected_ses = np.sqrt(np.square(expected_thetas + psi_elements['psi_b']).mean(axis=0) / n_obs)

    result_dict = {
        'dml_framework_obj': dml_framework_obj,
        'expected_thetas': expected_thetas,
        'expected_ses': expected_ses,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_theta(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_thetas,
        np.transpose(np.tile(dml_framework_fixture['expected_thetas'], (2, 1)))
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].thetas,
        np.tile(np.median(dml_framework_fixture['expected_thetas']), 2)
    )


@pytest.mark.ci
def test_dml_framework_se(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_ses,
        np.transpose(np.tile(dml_framework_fixture['expected_ses'], (2, 1)))
    )
