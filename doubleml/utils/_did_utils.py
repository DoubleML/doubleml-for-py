import warnings

import pandas as pd

import numpy as np

expected_time_vec_types = (int, float, np.ndarray)
expected_time_types = (int, float)


def _check_preprocess_g_t(g_values, t_values, control_group):
    # TODO: Implement specific possiblities (date, float, etc.) and checks
    if not isinstance(g_values, expected_time_vec_types):
        raise TypeError(f"Invalid type for g_values: expected one of {expected_time_vec_types}.")
    if not isinstance(t_values, expected_time_vec_types):
        raise TypeError(f"Invalid type for t_values: expected one of {expected_time_vec_types}.")

    if isinstance(g_values, (float, int)):
        g_values = np.array([g_values])
    if isinstance(t_values, (float, int)):
        t_values = np.array([t_values])

    if not all(isinstance(g, expected_time_types) for g in g_values):
        raise TypeError(f"Invalid type for g_values: expected one of {expected_time_types}.")
    if not all(isinstance(t, expected_time_types) for t in t_values):
        raise TypeError(f"Invalid type for t_values: expected one of {expected_time_types}.")

    # check shape is vector
    if g_values.ndim != 1:
        raise ValueError(f"g_values must be a vector. Number of dimensions is {g_values.ndim}.")
    if t_values.ndim != 1:
        raise ValueError(f"t_values must be a vector. Number of dimensions is {t_values.ndim}.")

    # Don't evaluate always treated
    never_treated_value = 0
    if never_treated_value in g_values:
        warnings.warn(f"The never treated group {never_treated_value} is removed from g_values.")
        g_values = np.atleast_1d(g_values[g_values != never_treated_value])

    # specify time horizon
    t_last = np.max(t_values)
    t_first = np.min(t_values)

    if any(g_values <= t_first):
        warnings.warn(f"Values before/equal the first period {t_first} are removed from g_values.")
        values_to_keep = g_values > t_first
        g_values = np.atleast_1d(g_values[values_to_keep])

    if any(g_values > t_last):
        warnings.warn(f"Values after the last period {t_last} are removed from g_values.")
        values_to_keep = g_values <= t_last
        g_values = np.atleast_1d(g_values[values_to_keep])

    # Don't evaluate those individuals treated in last period
    if control_group == "not_yet_treated":
        if np.any(t_values > np.max(g_values)):
            # issue a warning
            g_values = g_values[g_values < t_last]
            warnings.warn("Individuals treated in the last period are excluded from the analysis.")

    g_values = g_values[g_values > t_first]
    # t_values = t_values[t_values < t_last]

    # For each combination of g and t values, we need to find a pretreatment and evaluation period
    # Here only varying base period (for universal t_fac would be 0)
    t_fac = 1
    # shift position of t values by t_fac
    t_values_eval = t_values[t_fac:]

    # All combinations of g and t_pre and t_eval for evaluation
    gt_combinations = []

    for g_val in g_values:
        for t_idx, t_eval in enumerate(t_values_eval):
            t_pre = t_values[t_idx]

            # if post_treatment, i.e., if g_val <= t_eval, take last pre-treatment period before g
            if g_val < t_eval:
                t_upd = np.where(t_values < g_val)[0][-1]
                t_pre = t_values[t_upd]

            gt_combinations.append((int(g_val), int(t_pre), int(t_eval)))

    return gt_combinations
