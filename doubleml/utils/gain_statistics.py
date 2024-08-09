import numpy as np


def gain_statistics(dml_long, dml_short):
    """
    Compute gain statistics as benchmark values for sensitivity parameters ``cf_d`` and ``cf_y``.

    Parameters
    ----------

    dml_long :
        :class:`doubleml.DoubleML` model including all observed confounders

    dml_short :
        :class:`doubleml.DoubleML` model that excludes one or several benchmark confounders

    Returns
    --------
    benchmark_dict : dict
        Benchmarking dictionary (dict) with values for ``cf_d``, ``cf_y``, ``rho``, and ``delta_theta``.
    """

    # set input for readability
    sensitivity_elements_long = dml_long.framework.sensitivity_elements
    sensitivity_elements_short = dml_short.framework.sensitivity_elements

    if not isinstance(sensitivity_elements_long, dict):
        raise TypeError("dml_long does not contain the necessary sensitivity elements. "
                        "Expected dict for dml_long.framework.sensitivity_elements.")
    expected_keys = ['sigma2', 'nu2']
    if not all(key in sensitivity_elements_long.keys() for key in expected_keys):
        raise ValueError("dml_long does not contain the necessary sensitivity elements. "
                         "Required keys are: " + str(expected_keys))
    if not isinstance(sensitivity_elements_short, dict):
        raise TypeError("dml_short does not contain the necessary sensitivity elements. "
                        "Expected dict for dml_short.framework.sensitivity_elements.")
    if not all(key in sensitivity_elements_short.keys() for key in expected_keys):
        raise ValueError("dml_short does not contain the necessary sensitivity elements. "
                         "Required keys are: " + str(expected_keys))

    for key in expected_keys:
        if not isinstance(sensitivity_elements_long[key], np.ndarray):
            raise TypeError("dml_long does not contain the necessary sensitivity elements. "
                            f"Expected numpy.ndarray for key {key}.")
        if not isinstance(sensitivity_elements_short[key], np.ndarray):
            raise TypeError("dml_short does not contain the necessary sensitivity elements. "
                            f"Expected numpy.ndarray for key {key}.")
        if len(sensitivity_elements_long[key].shape) != 3 or sensitivity_elements_long[key].shape[0] != 1:
            raise ValueError("dml_long does not contain the necessary sensitivity elements. "
                             f"Expected 3 dimensions of shape (1, n_coef, n_rep) for key {key}.")
        if len(sensitivity_elements_short[key].shape) != 3 or sensitivity_elements_short[key].shape[0] != 1:
            raise ValueError("dml_short does not contain the necessary sensitivity elements. "
                             f"Expected 3 dimensions of shape (1, n_coef, n_rep) for key {key}.")
        if not np.array_equal(sensitivity_elements_long[key].shape, sensitivity_elements_short[key].shape):
            raise ValueError("dml_long and dml_short do not contain the same shape of sensitivity elements. "
                             "Shapes of " + key + " are: " + str(sensitivity_elements_long[key].shape) +
                             " and " + str(sensitivity_elements_short[key].shape))

    if not isinstance(dml_long.all_coef, np.ndarray):
        raise TypeError("dml_long.all_coef does not contain the necessary coefficients. Expected numpy.ndarray.")
    if not isinstance(dml_short.all_coef, np.ndarray):
        raise TypeError("dml_short.all_coef does not contain the necessary coefficients. Expected numpy.ndarray.")

    expected_shape = (sensitivity_elements_long['sigma2'].shape[1], sensitivity_elements_long['sigma2'].shape[2])
    if dml_long.all_coef.shape != expected_shape:
        raise ValueError("dml_long.all_coef does not contain the necessary coefficients. Expected shape: " +
                         str(expected_shape))
    if dml_short.all_coef.shape != expected_shape:
        raise ValueError("dml_short.all_coef does not contain the necessary coefficients. Expected shape: " +
                         str(expected_shape))

    # save elements for readability
    var_y = np.var(dml_long._dml_data.y)
    var_y_residuals_long = np.squeeze(sensitivity_elements_long['sigma2'], axis=0)
    nu2_long = np.squeeze(sensitivity_elements_long['nu2'], axis=0)
    var_y_residuals_short = np.squeeze(sensitivity_elements_short['sigma2'], axis=0)
    nu2_short = np.squeeze(sensitivity_elements_short['nu2'], axis=0)

    # compute nonparametric R2
    R2_y_long = 1.0 - np.divide(var_y_residuals_long, var_y)
    R2_y_short = 1.0 - np.divide(var_y_residuals_short, var_y)
    R2_riesz = np.divide(nu2_short, nu2_long)

    # Gain statistics
    all_cf_y_benchmark = np.clip(np.divide((R2_y_long - R2_y_short), (1.0 - R2_y_long)), 0, 1)
    all_cf_d_benchmark = np.clip(np.divide((1.0 - R2_riesz), R2_riesz), 0, 1)
    cf_y_benchmark = np.median(all_cf_y_benchmark, axis=1)
    cf_d_benchmark = np.median(all_cf_d_benchmark, axis=1)

    # change in estimates (slightly different to paper)
    all_delta_theta = dml_short.all_coef - dml_long.all_coef
    delta_theta = np.median(all_delta_theta, axis=1)

    # degree of adversity
    var_g = var_y_residuals_short - var_y_residuals_long
    var_riesz = nu2_long - nu2_short
    denom = np.sqrt(np.multiply(var_g, var_riesz), out=np.zeros_like(var_g), where=(var_g > 0) & (var_riesz > 0))
    rho_sign = np.sign(all_delta_theta)
    rho_values = np.clip(np.divide(np.absolute(all_delta_theta),
                                   denom,
                                   out=np.ones_like(all_delta_theta),
                                   where=denom != 0),
                         0.0, 1.0)
    all_rho_benchmark = np.multiply(rho_values, rho_sign)
    rho_benchmark = np.median(all_rho_benchmark, axis=1)
    benchmark_dict = {
        "cf_y": cf_y_benchmark,
        "cf_d": cf_d_benchmark,
        "rho": rho_benchmark,
        "delta_theta": delta_theta,
    }
    return benchmark_dict
