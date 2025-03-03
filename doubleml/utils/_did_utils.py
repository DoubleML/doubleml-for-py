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

    if np.issubdtype(x.dtype, np.floating) and not allow_nan and np.any(np.isnan(x)):
        raise ValueError(f"{input_name} contains missing values.")

    if np.issubdtype(x.dtype, np.datetime64) and not allow_nan and np.any(np.isnat(x)):
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


def _check_gt_combination(gt_combination, g_values, t_values, never_treated_value):
    g_value, t_value_pre, t_value_eval = gt_combination
    if g_value not in g_values:
        raise ValueError(f"The value {g_value} is not in the set of treatment group values {g_values}.")
    if _is_never_treated(g_value, never_treated_value):
        raise ValueError(f"The never treated group is not allowed as treatment group (g_value={never_treated_value}).")
    if t_value_pre not in t_values:
        raise ValueError(f"The value {t_value_pre} is not in the set of evaluation period values {t_values}.")
    if t_value_eval not in t_values:
        raise ValueError(f"The value {t_value_eval} is not in the set of evaluation period values {t_values}.")

    if t_value_pre == t_value_eval:
        raise ValueError(f"The pre-treatment and evaluation period must be different. Got {t_value_pre} for both.")

    if t_value_pre > t_value_eval:
        raise ValueError(
            "The pre-treatment period must be before the evaluation period. "
            f"Got t_value_pre {t_value_pre} and t_value_eval {t_value_eval}."
        )

    if t_value_pre >= g_value:
        warnings.warn(
            "The treatment was assigned before the first pre-treatment period. "
            f"Got t_value_pre {t_value_pre} and g_value {g_value}."
        )


def _check_gt_values(g_values, t_values):

    g_values = _convert_to_numpy_arrray(g_values, "g_values", allow_nan=True)
    t_values = _convert_to_numpy_arrray(t_values, "t_values", allow_nan=False)

    expected_dtypes = (np.integer, np.floating, np.datetime64)
    if not any(np.issubdtype(g_values.dtype, dt) for dt in expected_dtypes):
        raise ValueError(f"Invalid data type for g_values: expected one of {expected_dtypes}.")
    if not any(np.issubdtype(t_values.dtype, dt) for dt in expected_dtypes):
        raise ValueError(f"Invalid data type for t_values: expected one of {expected_dtypes}.")

    if np.issubdtype(g_values.dtype, np.datetime64) != np.issubdtype(t_values.dtype, np.datetime64):
        raise ValueError(
            "g_values and t_values must have the same data type. "
            f"Got {g_values.dtype} for g_values and {t_values.dtype} for t_values."
        )


def _construct_gt_combinations(setting, g_values, t_values, never_treated_value):
    """Construct treatment-time combinations for difference-in-differences analysis.

    Parameters:
        setting (str): Strategy for constructing combinations ('standard' only)
        g_values (array): Treatment group values, must be sorted
        t_values (array): Time period values, must be sorted

    Returns:
        list: List of (g_val, t_pre, t_eval) tuples
    """
    valid_settings = ["standard", "all"]
    if setting not in valid_settings:
        raise ValueError(f"gt_combinations must be one of {valid_settings}. {setting} was passed.")

    treatment_groups = g_values[~_is_never_treated(g_values, never_treated_value)]
    if not np.all(np.diff(treatment_groups) > 0):
        raise ValueError("g_values must be sorted in ascending order (Excluding never treated group).")
    if not np.all(np.diff(t_values) > 0):
        raise ValueError("t_values must be sorted in ascending order.")

    gt_combinations = []
    if setting == "standard":
        for g_val in treatment_groups:
            for i_t_eval, t_eval in enumerate(t_values[1:]):
                t_previous = t_values[i_t_eval]
                t_before_g = t_values[t_values < g_val][-1]
                t_pre = min(t_previous, t_before_g)
                gt_combinations.append((g_val, t_pre, t_eval))

    if setting == "all":
        for g_val in treatment_groups:
            for t_eval in t_values[1:]:
                for t_pre in t_values[t_values < min(g_val, t_eval)]:
                    gt_combinations.append((g_val, t_pre, t_eval))

    return gt_combinations


def _construct_gt_index(gt_combinations, g_values, t_values):
    """Construct a 3D array mapping group-time combinations to their indices.

    Parameters:
        gt_combinations: List of tuples (g_val, t_pre, t_eval)
        g_values: Array of group values
        t_values: Array of time values

    Returns:
        3D numpy masked array where entry [i,j,k] contains the index of the combination
        in gt_combinations if it exists, masked otherwise
    """
    gt_index = np.ma.masked_array(
        data=np.full(shape=(len(g_values), len(t_values), len(t_values)), fill_value=-1, dtype=np.int64), mask=True
    )
    for i_gt_combination, (g_val, t_pre, t_eval) in enumerate(gt_combinations):
        i_g = np.where(g_values == g_val)[0][0]
        i_t_pre = np.where(t_values == t_pre)[0][0]
        i_t_eval = np.where(t_values == t_eval)[0][0]
        gt_index[i_g, i_t_pre, i_t_eval] = i_gt_combination
        gt_index.mask[i_g, i_t_pre, i_t_eval] = False

    return gt_index


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
