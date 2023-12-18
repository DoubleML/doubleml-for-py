import pytest
import numpy as np

from doubleml.double_ml_base_linear import DoubleMLBaseLinear
from doubleml.double_ml_framework import DoubleMLFramework


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
def n_thetas(request):
    return request.param


@pytest.fixture(scope='module')
def test_dml_framework_coverage_fixture(n_rep, n_thetas):
    np.random.seed(42)
    R = 1000
    coverage = np.zeros((R, n_thetas))
    coverage_all_cis = np.zeros((R, n_thetas, n_rep))
    coverage_joint = np.zeros((R, 1))
    coverage_all_cis_joint = np.zeros((R, 1, n_rep))

    coverage_add_obj = np.zeros((R, n_thetas))
    coverage_all_cis_add_obj = np.zeros((R, n_thetas, n_rep))
    coverage_joint_add_obj = np.zeros((R, 1))
    coverage_all_cis_joint_add_obj = np.zeros((R, 1, n_rep))
    for r in range(R):
        n_obs = 100
        psi_elements_1 = {
            'psi_a': np.ones(shape=(n_obs, n_thetas, n_rep)),
            'psi_b': np.random.normal(size=(n_obs, n_thetas, n_rep)),
        }
        psi_elements_2 = {
            'psi_a': psi_elements_1['psi_a'],
            'psi_b': psi_elements_1['psi_b'] + 1.0,
        }
        dml_obj_1 = DoubleMLBaseLinear(psi_elements_1).estimate_thetas()
        dml_obj_2 = DoubleMLBaseLinear(psi_elements_2).estimate_thetas()

        # combine objects and estimate parameters
        dml_framework_obj_1 = DoubleMLFramework(dml_obj_1)
        dml_framework_obj_2 = DoubleMLFramework(dml_obj_2)

        true_thetas = np.vstack((np.repeat(0.0, n_thetas), np.repeat(-1.0, n_thetas))).transpose()
        ci = dml_framework_obj_1.confint(joint=False, level=0.95)
        coverage[r, :] = (true_thetas[:, 0] >= ci['2.5 %'].to_numpy()) & (true_thetas[:, 0] <= ci['97.5 %'].to_numpy())
        coverage_all_cis[r, :, :] = (true_thetas[:, 0].reshape(-1, 1) >= dml_framework_obj_1._all_cis[:, 0, :]) & \
            (true_thetas[:, 0].reshape(-1, 1) <= dml_framework_obj_1._all_cis[:, 1, :])

        # joint confidence interval
        dml_framework_obj_1.bootstrap(method='normal')
        ci_joint = dml_framework_obj_1.confint(joint=True, level=0.95)
        coverage_joint[r, :] = np.all(
            (true_thetas[:, 0] >= ci_joint['2.5 %'].to_numpy()) &
            (true_thetas[:, 0] <= ci_joint['97.5 %'].to_numpy()))
        coverage_all_cis_joint[r, :, :] = np.all(
            (true_thetas[:, 0].reshape(-1, 1) >= dml_framework_obj_1._all_cis[:, 0, :]) &
            (true_thetas[:, 0].reshape(-1, 1) <= dml_framework_obj_1._all_cis[:, 1, :]),
            axis=0)

        # add objects
        dml_framework_obj_add_obj = dml_framework_obj_1 + dml_framework_obj_2
        true_thetas_add_obj = np.sum(true_thetas, axis=1)
        ci_add_obj = dml_framework_obj_add_obj.confint(joint=False, level=0.95)
        coverage_add_obj[r, :] = (
            (true_thetas_add_obj >= ci_add_obj['2.5 %'].to_numpy()) &
            (true_thetas_add_obj <= ci_add_obj['97.5 %'].to_numpy()))
        coverage_all_cis_add_obj[r, :, :] = (
            (true_thetas_add_obj.reshape(-1, 1) >= dml_framework_obj_add_obj._all_cis[:, 0, :]) &
            (true_thetas_add_obj.reshape(-1, 1) <= dml_framework_obj_add_obj._all_cis[:, 1, :]))

        dml_framework_obj_add_obj.bootstrap(method='normal')
        ci_joint_add_obj = dml_framework_obj_add_obj.confint(joint=True, level=0.95)
        coverage_joint_add_obj[r, :] = np.all(
            (true_thetas_add_obj >= ci_joint_add_obj['2.5 %'].to_numpy()) &
            (true_thetas_add_obj <= ci_joint_add_obj['97.5 %'].to_numpy()))
        coverage_all_cis_joint_add_obj[r, :, :] = np.all(
            (true_thetas_add_obj.reshape(-1, 1) >= dml_framework_obj_add_obj._all_cis[:, 0, :]) &
            (true_thetas_add_obj.reshape(-1, 1) <= dml_framework_obj_add_obj._all_cis[:, 1, :]),
            axis=0)

    result_dict = {
        'coverage_rate': np.mean(coverage, axis=0),
        'coverage_rate_all_cis': np.mean(coverage_all_cis, axis=0),
        'coverage_rate_joint': np.mean(coverage_joint, axis=0),
        'coverage_rate_all_cis_joint': np.mean(coverage_all_cis_joint, axis=0),
        'coverage_rate_add_obj': np.mean(coverage_add_obj, axis=0),
        'coverage_rate_all_cis_add_obj': np.mean(coverage_all_cis_add_obj, axis=0),
        'coverage_rate_joint_add_obj': np.mean(coverage_joint_add_obj, axis=0),
        'coverage_rate_all_cis_joint_add_obj': np.mean(coverage_all_cis_joint_add_obj, axis=0),
    }
    return result_dict


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_coverage(test_dml_framework_coverage_fixture):
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_add_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_add_obj'] < 1.0))


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_coverage_joint(test_dml_framework_coverage_fixture):
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_joint'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_joint_add_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_add_obj'] < 1.0))
