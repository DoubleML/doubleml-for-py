import warnings

from collections.abc import Iterable
import pandas as pd

import numpy as np

expected_time_types = (int, float)


def _convert_to_numpy_arrray(x, input_name, allow_nan=False):
    if isinstance(x, np.ndarray):
        if not x.ndim == 1:
            raise ValueError(f"{input_name} must be a vector. Number of dimensions is {x.ndim}.")
    elif isinstance(x, (int, float)):
        x = np.array([x])
    elif isinstance(x, Iterable):
        if not all(isinstance(i, expected_time_types) for i in x):
            raise TypeError(f"Invalid type for {input_name}: expected one of {expected_time_types}.")
        x = np.array(x)
    else:
        raise TypeError(f"Invalid type for {input_name}.")

    if not allow_nan and np.any(np.isnan(x)):
        raise ValueError(f"{input_name} contains missing values.")

    return x


def _check_g_t_values(g_values, t_values, control_group):
    # TODO: Implement specific possiblities (date, float, etc.) and checks

    g_values = _convert_to_numpy_arrray(g_values, "g_values", allow_nan=True)
    t_values = _convert_to_numpy_arrray(t_values, "t_values", allow_nan=False)

    g_values = np.sort(g_values)
    t_values = np.sort(t_values)

    # Don't evaluate always treated
    never_treated_value = 0
    never_treated_exist = False
    if never_treated_value in g_values:
        never_treated_exist = True
        warnings.warn(f"The never treated group {never_treated_value} is removed from g_values.")
        g_values = np.atleast_1d(g_values[g_values != never_treated_value])

    # specify time horizon
    t_last = np.max(t_values)
    t_first = np.min(t_values)

    valid_g_values = np.full_like(g_values, True, dtype=bool)

    if np.any(g_values <= t_first):
        warnings.warn(f"Values before/equal the first period {t_first} are removed from g_values.")
        valid_g_values &= g_values > t_first

    if np.any(g_values > t_last):
        warnings.warn(f"Values after the last period {t_last} are removed from g_values.")
        valid_g_values &= g_values <= t_last

    # Don't evaluate those individuals treated in last period
    if (control_group == "not_yet_treated") and (not never_treated_exist):
        if np.any(g_values == t_last):
            warnings.warn("Individuals treated in the last period are excluded from the analysis " +
                          "(no comparison group available).")
            valid_g_values &= g_values < t_last

    g_values = np.atleast_1d(g_values[valid_g_values])

    return g_values, t_values


def _check_preprocess_g_t(g_values, t_values, control_group):
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
