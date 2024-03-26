import pytest
import numpy as np

from doubleml.utils._estimation import _var_est, _aggregate_coefs_and_ses


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def test_var_est_and_aggr_fixture(n_rep):
    n_obs = 100
    psi = np.random.normal(size=(n_obs, n_rep))
    psi_deriv = np.ones((n_obs, n_rep))

    all_thetas = np.mean(psi, axis=0).reshape((-1, n_rep))
    all_ses = np.zeros(n_rep).reshape((-1, n_rep))
    all_var_scaling_factors = np.zeros(n_rep)

    for i_rep in range(n_rep):
        var_estimate, var_scaling_factor = _var_est(
            psi=psi[:, i_rep],
            psi_deriv=psi_deriv[:, i_rep],
            smpls=None,
            is_cluster_data=False
        )
        all_ses[0, i_rep] = np.sqrt(var_estimate)
        all_var_scaling_factors[i_rep] = var_scaling_factor

    theta, se = _aggregate_coefs_and_ses(
        all_coefs=all_thetas,
        all_ses=all_ses,
        var_scaling_factors=np.full(1, n_obs),
    )

    expected_theta = np.median(all_thetas)
    expected_se = np.sqrt(np.median(
        np.square(all_ses) + np.square(all_thetas - expected_theta) / n_obs
        )
    )

    result_dict = {
        'theta': theta,
        'se': se,
        'expected_theta': expected_theta,
        'expected_se': expected_se,
    }
    return result_dict


@pytest.mark.ci
def test_aggregate_theta(test_var_est_and_aggr_fixture):
    assert np.allclose(
        test_var_est_and_aggr_fixture['theta'],
        test_var_est_and_aggr_fixture['expected_theta']
    )


@pytest.mark.ci
def test_aggregate_se(test_var_est_and_aggr_fixture):
    assert np.allclose(
        test_var_est_and_aggr_fixture['se'],
        test_var_est_and_aggr_fixture['expected_se']
    )
