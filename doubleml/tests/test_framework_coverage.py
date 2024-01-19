import pytest
import numpy as np

from doubleml.double_ml_framework import DoubleMLFramework, concat
from ._utils import generate_dml_dict


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

    coverage_sub_obj = np.zeros((R, n_thetas))
    coverage_all_cis_sub_obj = np.zeros((R, n_thetas, n_rep))
    coverage_joint_sub_obj = np.zeros((R, 1))
    coverage_all_cis_joint_sub_obj = np.zeros((R, 1, n_rep))

    coverage_mul_obj = np.zeros((R, n_thetas))
    coverage_all_cis_mul_obj = np.zeros((R, n_thetas, n_rep))
    coverage_joint_mul_obj = np.zeros((R, 1))
    coverage_all_cis_joint_mul_obj = np.zeros((R, 1, n_rep))

    coverage_concat = np.zeros((R, 2*n_thetas))
    coverage_all_cis_concat = np.zeros((R, 2*n_thetas, n_rep))
    coverage_joint_concat = np.zeros((R, 1))
    coverage_all_cis_joint_concat = np.zeros((R, 1, n_rep))
    for r in range(R):
        n_obs = 200

        # generate score samples
        psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
        psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
        doubleml_dict = generate_dml_dict(psi_a, psi_b)
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep)) + 1.0
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)

        # combine objects and estimate parameters
        dml_framework_obj_1 = DoubleMLFramework(doubleml_dict)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)

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

        # substract objects
        dml_framework_obj_sub_obj = dml_framework_obj_1 - dml_framework_obj_2
        true_thetas_sub_obj = true_thetas[:, 0] - true_thetas[:, 1]
        ci_sub_obj = dml_framework_obj_sub_obj.confint(joint=False, level=0.95)
        coverage_sub_obj[r, :] = (
            (true_thetas_sub_obj >= ci_sub_obj['2.5 %'].to_numpy()) &
            (true_thetas_sub_obj <= ci_sub_obj['97.5 %'].to_numpy()))
        coverage_all_cis_sub_obj[r, :, :] = (
            (true_thetas_sub_obj.reshape(-1, 1) >= dml_framework_obj_sub_obj._all_cis[:, 0, :]) &
            (true_thetas_sub_obj.reshape(-1, 1) <= dml_framework_obj_sub_obj._all_cis[:, 1, :]))

        dml_framework_obj_sub_obj.bootstrap(method='normal')
        ci_joint_sub_obj = dml_framework_obj_sub_obj.confint(joint=True, level=0.95)
        coverage_joint_sub_obj[r, :] = np.all(
            (true_thetas_sub_obj >= ci_joint_sub_obj['2.5 %'].to_numpy()) &
            (true_thetas_sub_obj <= ci_joint_sub_obj['97.5 %'].to_numpy()))
        coverage_all_cis_joint_sub_obj[r, :, :] = np.all(
            (true_thetas_sub_obj.reshape(-1, 1) >= dml_framework_obj_sub_obj._all_cis[:, 0, :]) &
            (true_thetas_sub_obj.reshape(-1, 1) <= dml_framework_obj_sub_obj._all_cis[:, 1, :]),
            axis=0)

        # multiply objects
        dml_framework_obj_mul_obj = dml_framework_obj_2 * 2
        true_thetas_mul_obj = 2 * true_thetas[:, 1]
        ci_mul_obj = dml_framework_obj_mul_obj.confint(joint=False, level=0.95)
        coverage_mul_obj[r, :] = (
            (true_thetas_mul_obj >= ci_mul_obj['2.5 %'].to_numpy()) &
            (true_thetas_mul_obj <= ci_mul_obj['97.5 %'].to_numpy()))
        coverage_all_cis_mul_obj[r, :, :] = (
            (true_thetas_mul_obj.reshape(-1, 1) >= dml_framework_obj_mul_obj._all_cis[:, 0, :]) &
            (true_thetas_mul_obj.reshape(-1, 1) <= dml_framework_obj_mul_obj._all_cis[:, 1, :]))

        dml_framework_obj_mul_obj.bootstrap(method='normal')
        ci_joint_mul_obj = dml_framework_obj_mul_obj.confint(joint=True, level=0.95)
        coverage_joint_mul_obj[r, :] = np.all(
            (true_thetas_mul_obj >= ci_joint_mul_obj['2.5 %'].to_numpy()) &
            (true_thetas_mul_obj <= ci_joint_mul_obj['97.5 %'].to_numpy()))
        coverage_all_cis_joint_mul_obj[r, :, :] = np.all(
            (true_thetas_mul_obj.reshape(-1, 1) >= dml_framework_obj_mul_obj._all_cis[:, 0, :]) &
            (true_thetas_mul_obj.reshape(-1, 1) <= dml_framework_obj_mul_obj._all_cis[:, 1, :]),
            axis=0)

        # concat objects
        dml_framework_obj_concat = concat([dml_framework_obj_1, dml_framework_obj_2])
        true_thetas_concat = true_thetas.reshape(-1, order='F')
        ci_concat = dml_framework_obj_concat.confint(joint=False, level=0.95)
        coverage_concat[r, :] = (
            (true_thetas_concat >= ci_concat['2.5 %'].to_numpy()) &
            (true_thetas_concat <= ci_concat['97.5 %'].to_numpy()))
        coverage_all_cis_concat[r, :, :] = (
            (true_thetas_concat.reshape(-1, 1) >= dml_framework_obj_concat._all_cis[:, 0, :]) &
            (true_thetas_concat.reshape(-1, 1) <= dml_framework_obj_concat._all_cis[:, 1, :]))

        dml_framework_obj_concat.bootstrap(method='normal')
        ci_joint_concat = dml_framework_obj_concat.confint(joint=True, level=0.95)
        coverage_joint_concat[r, :] = np.all(
            (true_thetas_concat >= ci_joint_concat['2.5 %'].to_numpy()) &
            (true_thetas_concat <= ci_joint_concat['97.5 %'].to_numpy()))
        coverage_all_cis_joint_concat[r, :, :] = np.all(
            (true_thetas_concat.reshape(-1, 1) >= dml_framework_obj_concat._all_cis[:, 0, :]) &
            (true_thetas_concat.reshape(-1, 1) <= dml_framework_obj_concat._all_cis[:, 1, :]),
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
        'coverage_rate_sub_obj': np.mean(coverage_sub_obj, axis=0),
        'coverage_rate_all_cis_sub_obj': np.mean(coverage_all_cis_sub_obj, axis=0),
        'coverage_rate_joint_sub_obj': np.mean(coverage_joint_sub_obj, axis=0),
        'coverage_rate_all_cis_joint_sub_obj': np.mean(coverage_all_cis_joint_sub_obj, axis=0),
        'coverage_rate_mul_obj': np.mean(coverage_mul_obj, axis=0),
        'coverage_rate_all_cis_mul_obj': np.mean(coverage_all_cis_mul_obj, axis=0),
        'coverage_rate_joint_mul_obj': np.mean(coverage_joint_mul_obj, axis=0),
        'coverage_rate_all_cis_joint_mul_obj': np.mean(coverage_all_cis_joint_mul_obj, axis=0),
        'coverage_rate_concat': np.mean(coverage_concat, axis=0),
        'coverage_rate_all_cis_concat': np.mean(coverage_all_cis_concat, axis=0),
        'coverage_rate_joint_concat': np.mean(coverage_joint_concat, axis=0),
        'coverage_rate_all_cis_joint_concat': np.mean(coverage_all_cis_joint_concat, axis=0),
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_coverage(test_dml_framework_coverage_fixture):
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_add_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_add_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_sub_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_sub_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_sub_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_mul_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_mul_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_mul_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_concat'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_concat'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_concat'] < 1.0))


@pytest.mark.ci
def test_dml_framework_coverage_joint(test_dml_framework_coverage_fixture):
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint_add_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_add_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_add_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint_sub_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_sub_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_sub_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint_mul_obj'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_mul_obj'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_mul_obj'] < 1.0))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_joint_concat'] >= 0.9))
    assert np.all((test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_concat'] >= 0.9) &
                  (test_dml_framework_coverage_fixture['coverage_rate_all_cis_joint_concat'] < 1.0))
