import numpy as np


def _var_est(psi, psi_deriv):
    var_scaling_factor = psi.shape[0]
    J = np.mean(psi_deriv)
    gamma_hat = np.mean(np.square(psi))

    scaling = np.divide(1.0, np.multiply(var_scaling_factor, np.square(J)))
    sigma2_hat = np.multiply(scaling, gamma_hat)

    return sigma2_hat, var_scaling_factor


def _aggregate_thetas_and_ses(
        all_thetas,
        all_ses,
        var_scaling_factor,
        aggregation_method='median'):

    if aggregation_method == 'median':
        aggregation_func = np.median
    else:
        assert aggregation_method == 'mean'
        aggregation_func = np.mean

    theta_hat = aggregation_func(all_thetas)
    theta_deviations = np.square(all_thetas - theta_hat)

    rescaled_variances = np.multiply(np.square(all_ses), var_scaling_factor)
    var_hat = aggregation_func(rescaled_variances + theta_deviations)
    se_hat = np.sqrt(np.divide(var_hat, var_scaling_factor))
    return theta_hat, se_hat


def _draw_weights(method, n_rep_boot, n_obs):
    if method == 'Bayes':
        weights = np.random.exponential(scale=1.0, size=(n_rep_boot, n_obs)) - 1.
    elif method == 'normal':
        weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
    elif method == 'wild':
        xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
    else:
        raise ValueError('invalid boot method')

    return weights
