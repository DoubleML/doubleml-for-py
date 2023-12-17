import numpy as np


def _initialize_arrays(n_thetas, n_rep, n_obs):
    thetas = np.full(shape=(n_thetas), fill_value=np.nan)
    ses = np.full(shape=(n_thetas), fill_value=np.nan)
    all_thetas = np.full(shape=(n_thetas, n_rep), fill_value=np.nan)
    all_ses = np.full(shape=(n_thetas, n_rep), fill_value=np.nan)
    var_scaling_factor = np.full(shape=(n_thetas), fill_value=np.nan)
    psi = np.full(shape=(n_obs, n_thetas, n_rep), fill_value=np.nan)
    psi_deriv = np.full(shape=(n_obs, n_thetas, n_rep), fill_value=np.nan)
    return thetas, ses, all_thetas, all_ses, var_scaling_factor, psi, psi_deriv


def _var_est(psi, psi_deriv):
    var_scaling_factor = psi.shape[0]
    J = np.mean(psi_deriv, axis=0)
    gamma_hat = np.mean(np.square(psi), axis=0)

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

    thetas_hat = aggregation_func(all_thetas, axis=1)
    thetas_deviations = np.square(all_thetas - thetas_hat)

    rescaled_variances = np.multiply(np.square(all_ses), var_scaling_factor)
    var_hat = aggregation_func(rescaled_variances + thetas_deviations)
    ses_hat = np.sqrt(np.divide(var_hat, var_scaling_factor))
    return thetas_hat, ses_hat


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
