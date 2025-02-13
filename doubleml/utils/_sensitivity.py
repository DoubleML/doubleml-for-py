import numpy as np


def _compute_sensitivity_bias(sigma2, nu2, psi_sigma2, psi_nu2):
    max_bias = np.sqrt(np.multiply(sigma2, nu2))
    psi_max_bias = np.divide(np.add(np.multiply(sigma2, psi_nu2), np.multiply(nu2, psi_sigma2)), np.multiply(2.0, max_bias))
    return max_bias, psi_max_bias
