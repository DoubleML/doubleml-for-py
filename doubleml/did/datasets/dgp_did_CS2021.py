
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from ...data.panel_data import DoubleMLPanelData
from .dgp_did_SZ2020 import _generate_features, _select_features

# Based on https://doi.org/10.1016/j.jeconom.2020.12.001 (see Appendix SC)
# and https://d2cml-ai.github.io/csdid/examples/csdid_basic.html#Examples-with-simulated-data


def _f_ps_groups(w, xi, n_groups):
    # Create coefficient matrix: 4 features x n_groups
    coef_vec = np.array([-1.0, 0.5, -0.25, -0.1])

    # use i_group/n_groups as coeffect for columns
    coef_matrix = np.array([coef_vec * (i_group/n_groups) for i_group in range(1, n_groups + 1)]).T

    res = xi * (w @ coef_matrix)
    return res


def _f_reg_time(w, n_time_perios):
    coef_vec = np.array([27.4, 13.7, 13.7, 13.7])

    # use time period as coeffect for columns
    coef_matrix = np.array([coef_vec * i_time for i_time in range(1, n_time_perios + 1)]).T

    res = 210 + w @ coef_matrix
    return res


def make_did_CS2021(n_obs=1000, dgp_type=1, time_type="datetime", **kwargs):
    c = kwargs.get("c", 0.0)
    dim_x = kwargs.get("dim_x", 4)
    xi = kwargs.get("xi", 0.75)
    n_periods = kwargs.get("n_periods", 5)
    n_pre_treat_periods = kwargs.get("n_pre_treat_periods", 2)
    start_date = kwargs.get("start_date", "2025-01")

    expected_time_types = ("datetime", "float")
    if time_type not in expected_time_types:
        raise ValueError(f"time_type must be one of {expected_time_types}. Got {time_type}.")

    x, z = _generate_features(n_obs, c, dim_x=dim_x)
    features_ps, features_reg = _select_features(dgp_type, x, z)

    # generate possible time periods
    time_periods = np.array([np.datetime64(start_date) + np.timedelta64(i, 'M') for i in range(n_periods)])
    n_time_periods = len(time_periods)

    # set treatment values for time periods greater than n_pre_treat_periods
    treatment_values = time_periods[time_periods >= time_periods[n_pre_treat_periods]]
    treatment_values = np.append(treatment_values, np.datetime64('NaT'))
    n_treatment_groups = len(treatment_values)

    # treatment assignment and propensities (shape (n_obs,))
    is_experimental = (dgp_type == 5) or (dgp_type == 6)
    if is_experimental:
        # Set D to be experimental
        p = np.ones(n_treatment_groups) / n_treatment_groups
        d_index = np.random.choice(n_treatment_groups, size=n_obs, p=p)
    else:
        unnormalized_p = np.exp(_f_ps_groups(features_ps, xi, n_groups=n_treatment_groups))
        p = unnormalized_p / unnormalized_p.sum(1, keepdims=True)
        d_index = np.array([np.random.choice(n_treatment_groups, p=p_row) for p_row in p])

    # fixed effects (shape (n_obs, n_time_periods))
    time_effects = np.arange(n_time_periods)
    delta_t = np.tile(time_effects, (n_obs, 1))
    indiviual_effects = np.random.normal(loc=d_index, scale=1, size=(n_obs,))
    eta_i = np.tile(indiviual_effects, (n_time_periods, 1)).T

    # error terms (shape (n_obs, n_time_periods))
    epsilon_0 = np.random.normal(loc=0, scale=1, size=(n_obs, n_time_periods))
    epsilon_1 = np.random.normal(loc=0, scale=1, size=(n_obs, n_time_periods))

    # regression function (shape (n_obs, n_time_periods))
    f_reg = _f_reg_time(features_reg, n_time_periods)

    # treatment effecs (shape (n_obs, n_time_periods))
    exposure_pre_period = np.zeros((n_obs, n_pre_treat_periods))
    exposure_post_first_treatment = np.clip(np.arange(n_treatment_groups - 1) - d_index.reshape(-1, 1) + 1, a_min=0,
                                            a_max=None)
    exposure_time = np.column_stack((exposure_pre_period, exposure_post_first_treatment))
    delta_e = exposure_time

    # add heterogeneity in treatment effects
    heterogeneity_x = 0.1 * x[:, 0]
    heterogeneity = heterogeneity_x.reshape(-1, 1) * exposure_time
    delta_e += heterogeneity

    # potential outcomes (shape (n_obs, n_time_periods))
    y0 = f_reg + delta_t + eta_i + epsilon_0
    y1 = y0 + delta_e + (epsilon_1 - epsilon_0)

    # observed outcomes (shape (n_obs, n_time_periods))
    is_exposed = exposure_time > 0
    y = y1 * is_exposed + y0 * ~is_exposed

    # map treatment index to values
    d = np.array([treatment_values[i] for i in d_index])
    d_matrix = np.tile(d, (n_time_periods, 1)).T

    # create matrices to flatten the data
    id_matrix = np.tile(np.arange(n_obs), (n_time_periods, 1)).T
    time_matrix = np.tile(time_periods, (n_obs, 1))

    df = pd.DataFrame({
        "id": id_matrix.flatten(),
        "y": y.flatten(),
        "y0": y0.flatten(),
        "y1": y1.flatten(),
        "d": d_matrix.flatten(),
        "t": time_matrix.flatten(),
        **{f"Z{i + 1}": z[:, i].repeat(n_time_periods) for i in range(dim_x)}
    })

    return df
