import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

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


def _get_never_treated_value(g_values):
    never_treated_value = 0
    if np.issubdtype(g_values.dtype, np.floating):
        never_treated_value = np.nan
    elif np.issubdtype(g_values.dtype, np.datetime64):
        never_treated_value = pd.NaT
    return never_treated_value


def _is_never_treated(x, never_treated_value):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if never_treated_value is np.nan:
        return np.isnan(x)
    elif never_treated_value is pd.NaT:
        return pd.isna(x)
    else:
        assert never_treated_value == 0
        return x == 0


def _check_control_group(control_group):
    valid_control_groups = ["never_treated", "not_yet_treated"]
    if control_group not in valid_control_groups:
        raise ValueError(f"The control group has to be one of {valid_control_groups}. " + f"{control_group} was passed.")

    return control_group


def _check_g_t_value_combination(g_value, t_value):
    pass


def _check_g_t_values(g_values, t_values, control_group):
    # TODO: Implement specific possiblities (date, float, etc.) and checks

    g_values = _convert_to_numpy_arrray(g_values, "g_values", allow_nan=True)
    t_values = _convert_to_numpy_arrray(t_values, "t_values", allow_nan=False)

    expected_dtypes = (np.floating, np.datetime64)
    if g_values.dtype not in expected_dtypes:
        raise ValueError(f"Invalid data type for g_values: expected one of {expected_dtypes}.")
    if t_values.dtype not in expected_dtypes:
        raise ValueError(f"Invalid data type for t_values: expected one of {expected_dtypes}.")

    if g_values.dtype != t_values.dtype:
        raise ValueError(f"g_values and t_values must have the same data type. Got {g_values.dtype} and {t_values.dtype}.")

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
            warnings.warn(
                "Individuals treated in the last period are excluded from the analysis " + "(no comparison group available)."
            )
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


def _set_id_positions(a, n_obs, id_positions, fill_value):
    if a is not None:
        new_a = np.full((n_obs, *a.shape[1:]), fill_value=fill_value)
        new_a[id_positions] = a
    else:
        new_a = None

    return new_a


def _get_id_positions(a, id_positions):
    if a is not None:
        new_a = a[id_positions]
    else:
        new_a = None

    return new_a
