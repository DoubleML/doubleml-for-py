import numpy as np
from scipy.stats import norm

from .._utils import _aggregate_coefs_and_ses


def doubleml_sensitivity_manual(sensitivity_elements, all_coefs, psi, psi_deriv, cf_y, cf_d, rho, level):

    # specify the parameters
    sigma2 = sensitivity_elements['sigma2']
    nu2 = sensitivity_elements['nu2']
    psi_sigma = sensitivity_elements['psi_sigma2']
    psi_nu = sensitivity_elements['psi_nu2']
    psi_scaled = np.divide(psi, np.mean(psi_deriv, axis=0))

    confounding_strength = np.multiply(np.abs(rho), np.sqrt(np.multiply(cf_y, np.divide(cf_d, 1.0-cf_d))))
    S = np.sqrt(np.multiply(sigma2, nu2))

    all_theta_lower = all_coefs - np.multiply(np.transpose(np.squeeze(S, axis=0)), confounding_strength)
    all_theta_upper = all_coefs + np.multiply(np.transpose(np.squeeze(S, axis=0)), confounding_strength)

    psi_S2 = np.multiply(sigma2, psi_nu) + np.multiply(nu2, psi_sigma)
    psi_bias = np.multiply(np.divide(confounding_strength, np.multiply(2.0, S)), psi_S2)
    psi_lower = psi_scaled - psi_bias
    psi_upper = psi_scaled + psi_bias

    var_scaling_factor = psi_scaled.shape[0]
    # transpose to obtain shape (n_coefs, n_reps); includes scaling with n^{-1/2}
    all_sigma_lower = np.transpose(np.sqrt(np.divide(np.mean(np.square(psi_lower), axis=0), var_scaling_factor)))
    all_sigma_upper = np.transpose(np.sqrt(np.divide(np.mean(np.square(psi_upper), axis=0), var_scaling_factor)))

    theta_lower, sigma_lower = _aggregate_coefs_and_ses(all_theta_lower, all_sigma_lower, var_scaling_factor)
    theta_upper, sigma_upper = _aggregate_coefs_and_ses(all_theta_upper, all_sigma_upper, var_scaling_factor)

    quant = norm.ppf(level)
    ci_lower = theta_lower - np.multiply(quant, sigma_lower)
    ci_upper = theta_upper + np.multiply(quant, sigma_upper)

    theta_dict = {'lower': theta_lower,
                  'upper': theta_upper}

    se_dict = {'lower': sigma_lower,
               'upper': sigma_upper}

    ci_dict = {'lower': ci_lower,
               'upper': ci_upper}

    res_dict = {'theta': theta_dict,
                'se': se_dict,
                'ci': ci_dict}

    return res_dict
