import pytest
import numpy as np

from doubleml.utils._estimation import _var_est, _aggregate_coefs_and_ses


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
def n_coefs(request):
    return request.param


@pytest.fixture(scope='module')
def test_var_est_and_aggr_fixture(n_rep, n_coefs):
    np.random.seed(42)

    all_thetas = np.full((n_coefs, n_rep), np.nan)
    all_ses = np.full((n_coefs, n_rep), np.nan)
    expected_all_var = np.full((n_coefs, n_rep), np.nan)
    all_var_scaling_factors = np.full((n_coefs, n_rep), np.nan)

    for i_coef in range(n_coefs):
        n_obs = np.random.randint(100, 200)
        for i_rep in range(n_rep):

            psi = np.random.normal(size=(n_obs))
            psi_deriv = np.ones((n_obs))

            all_thetas[i_coef, i_rep] = np.mean(psi)

            var_estimate, var_scaling_factor = _var_est(
                psi=psi,
                psi_deriv=psi_deriv,
                smpls=None,
                is_cluster_data=False
            )

            all_ses[i_coef, i_rep] = np.sqrt(var_estimate)
            all_var_scaling_factors[i_coef, i_rep] = var_scaling_factor

    expected_theta = np.median(all_thetas, axis=1)
    for i_coef in range(n_coefs):
        for i_rep in range(n_rep):
            theta_deviation = np.square(all_thetas[i_coef, i_rep] - expected_theta[i_coef])
            expected_all_var[i_coef, i_rep] = np.square(all_ses[i_coef, i_rep]) + \
                np.divide(theta_deviation, all_var_scaling_factors[i_coef, i_rep])

    expected_se = np.sqrt(np.median(expected_all_var, axis=1))

    # without n_rep
    theta, se = _aggregate_coefs_and_ses(
        all_coefs=all_thetas,
        all_ses=all_ses,
        var_scaling_factors=all_var_scaling_factors[:, 0],
    )

    # with n_rep
    theta_2, se_2 = _aggregate_coefs_and_ses(
        all_coefs=all_thetas,
        all_ses=all_ses,
        var_scaling_factors=all_var_scaling_factors,
    )

    result_dict = {
        'theta': theta,
        'se': se,
        'theta_2': theta_2,
        'se_2': se_2,
        'expected_theta': expected_theta,
        'expected_se': expected_se,
        'all_var_scaling_factors': all_var_scaling_factors,
    }
    return result_dict


@pytest.mark.ci
def test_aggregate_theta(test_var_est_and_aggr_fixture):
    assert np.allclose(
        test_var_est_and_aggr_fixture['theta'],
        test_var_est_and_aggr_fixture['expected_theta']
    )
    assert np.allclose(
        test_var_est_and_aggr_fixture['theta_2'],
        test_var_est_and_aggr_fixture['expected_theta']
    )


@pytest.mark.ci
def test_aggregate_se(test_var_est_and_aggr_fixture):
    assert np.allclose(
        test_var_est_and_aggr_fixture['se'],
        test_var_est_and_aggr_fixture['expected_se']
    )
    assert np.allclose(
        test_var_est_and_aggr_fixture['se_2'],
        test_var_est_and_aggr_fixture['expected_se']
    )
