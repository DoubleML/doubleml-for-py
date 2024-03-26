import numpy as np
import pandas as pd
from scipy.stats import norm
import copy

from ..utils._estimation import _aggregate_coefs_and_ses


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

    var_scaling_factor = np.full(all_coefs.shape[0], psi_scaled.shape[0])
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


def doubleml_sensitivity_benchmark_manual(dml_obj, benchmarking_set):
    x_list_long = dml_obj._dml_data.x_cols
    x_list_short = [x for x in x_list_long if x not in benchmarking_set]

    dml_short = copy.deepcopy(dml_obj)
    dml_short._dml_data.x_cols = x_list_short
    dml_short.fit()

    var_y = np.var(dml_obj._dml_data.y)
    var_y_long = np.squeeze(dml_obj.sensitivity_elements['sigma2'], axis=0)
    nu2_long = np.squeeze(dml_obj.sensitivity_elements['nu2'], axis=0)
    var_y_short = np.squeeze(dml_short.sensitivity_elements['sigma2'], axis=0)
    nu2_short = np.squeeze(dml_short.sensitivity_elements['nu2'], axis=0)

    R2_y_long = 1.0 - var_y_long / var_y
    R2_y_short = 1.0 - var_y_short / var_y
    R2_riesz = nu2_short / nu2_long

    all_cf_y_benchmark = np.clip((R2_y_long - R2_y_short) / (1.0 - R2_y_long), 0, 1)
    all_cf_d_benchmark = np.clip((1.0 - R2_riesz) / R2_riesz, 0, 1)

    cf_y_benchmark = np.median(all_cf_y_benchmark, axis=0)
    cf_d_benchmark = np.median(all_cf_d_benchmark, axis=0)

    all_delta_theta = np.transpose(dml_short.all_coef - dml_obj.all_coef)
    delta_theta = np.median(all_delta_theta, axis=0)

    var_g = var_y_short - var_y_long
    var_riesz = nu2_long - nu2_short
    denom = np.sqrt(np.multiply(var_g, var_riesz), out=np.zeros_like(var_g), where=(var_g > 0) & (var_riesz > 0))
    all_rho_benchmark = np.sign(all_delta_theta) * \
        np.clip(np.divide(np.absolute(all_delta_theta), denom, out=np.ones_like(all_delta_theta), where=denom != 0),
                0, 1)
    rho_benchmark = np.median(all_rho_benchmark, axis=0)

    benchmark_dict = {
        'cf_y': cf_y_benchmark,
        'cf_d': cf_d_benchmark,
        'rho': rho_benchmark,
        'delta_theta': delta_theta,
    }
    return pd.DataFrame(benchmark_dict, index=dml_obj._dml_data.d_cols)
