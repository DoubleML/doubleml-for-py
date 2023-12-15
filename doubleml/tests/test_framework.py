import pytest
import numpy as np
import pandas as pd

from doubleml.double_ml_base_linear import DoubleMLBaseLinear
from doubleml.double_ml_framework import DoubleMLFramework


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_framework_fixture(n_rep):
    n_obs = 100
    psi_elements_1 = {
        'psi_a': np.ones(shape=(n_obs, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, n_rep)),
    }
    psi_elements_2 = {
        'psi_a': psi_elements_1['psi_a'],
        'psi_b': psi_elements_1['psi_b'] + 1.0,
    }
    dml_obj_1 = DoubleMLBaseLinear(psi_elements_1).estimate_theta()
    dml_obj_2 = DoubleMLBaseLinear(psi_elements_2).estimate_theta()

    # combine objects and estimate parameters
    dml_framework_obj_1 = DoubleMLFramework(dml_obj_1)
    dml_framework_obj_2 = DoubleMLFramework(dml_obj_2)

    ci_1 = dml_framework_obj_1.confint(joint=False, level=0.95)
    ci_2 = dml_framework_obj_2.confint(joint=False, level=0.95)
    result_dict = {
        'dml_obj_1': dml_obj_1,
        'dml_obj_2': dml_obj_2,
        'dml_framework_obj_1': dml_framework_obj_1,
        'dml_framework_obj_2': dml_framework_obj_2,
        'ci_1': ci_1,
        'ci_2': ci_2,
    }
    return result_dict


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_theta(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_1'].all_thetas[0, :],
        dml_framework_fixture['dml_obj_1'].all_thetas
    )


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_se(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_1'].all_ses[0, :],
        dml_framework_fixture['dml_obj_1'].all_ses
    )


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_ci(dml_framework_fixture):
    assert isinstance(dml_framework_fixture['ci_1'], pd.DataFrame)


@pytest.fixture(scope='module')
def test_dml_framework_coverage_fixture(n_rep):
    R = 500
    coverage_1 = np.zeros((R, 1))
    coverage_joint_1 = np.zeros((R, 1))
    for r in range(R):
        n_obs = 100
        psi_elements_1 = {
            'psi_a': np.ones(shape=(n_obs, n_rep)),
            'psi_b': np.random.normal(size=(n_obs, n_rep)),
        }
        psi_elements_2 = {
            'psi_a': psi_elements_1['psi_a'],
            'psi_b': psi_elements_1['psi_b'] + 1.0,
        }
        dml_obj_1 = DoubleMLBaseLinear(psi_elements_1).estimate_theta()
        dml_obj_2 = DoubleMLBaseLinear(psi_elements_2).estimate_theta()

        # combine objects and estimate parameters
        dml_framework_obj_1 = DoubleMLFramework(dml_obj_1)
        dml_framework_obj_2 = DoubleMLFramework(dml_obj_2)

        ci_1 = dml_framework_obj_1.confint(joint=False, level=0.95)
        ci_2 = dml_framework_obj_2.confint(joint=False, level=0.95)

        true_thetas = np.array([0.0, -1.0])
        coverage_1[r, :] = (true_thetas[0] >= ci_1['2.5 %'].values) & (true_thetas[0] <= ci_1['97.5 %'].values)

        dml_framework_obj_1.bootstrap(method='normal')
        ci_joint_1 = dml_framework_obj_1.confint(joint=True, level=0.95)
        coverage_joint_1[r, :] = (true_thetas[0] >= ci_joint_1['2.5 %'].values) & \
            (true_thetas[0] <= ci_joint_1['97.5 %'].values)

    result_dict = {
        'dml_framework_obj_1': dml_framework_obj_1,
        'coverage_rate_1': np.mean(coverage_1, axis=0),
        'coverage_rate_joint_1': np.mean(coverage_joint_1, axis=0),
    }
    return result_dict


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_coverage(test_dml_framework_coverage_fixture):
    assert all(test_dml_framework_coverage_fixture['coverage_rate_1'] >= np.full(1, 0.9))


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_coverage_joint(test_dml_framework_coverage_fixture):
    assert all(test_dml_framework_coverage_fixture['coverage_rate_joint_1'] >= np.full(1, 0.9))