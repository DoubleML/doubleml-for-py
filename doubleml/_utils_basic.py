import numpy as np


def _var_est(psi, psi_deriv):
    var_scaling_factor = psi.shape[0]
    J = np.mean(psi_deriv)
    gamma_hat = np.mean(np.square(psi))

    scaling = np.divide(1.0, np.multiply(var_scaling_factor, np.square(J)))
    sigma2_hat = np.multiply(scaling, gamma_hat)

    return sigma2_hat, var_scaling_factor


def _aggregate_thetas_and_ses(all_thetas, all_ses, var_scaling_factor):
    theta_hat = np.median(all_thetas)

    rescaled_variances = np.multiply(np.power(all_ses, 2), var_scaling_factor)
    theta_deviations = np.power(all_thetas - theta_hat, 2)
    median_var = np.median(rescaled_variances + theta_deviations)
    se_hat = np.sqrt(np.divide(median_var, var_scaling_factor))
    return theta_hat, se_hat
