import warnings
from collections.abc import Iterable

import numpy as np


def _check_preprocess_g_t(g_values, t_values, control_group):
    # check if iterable is enough or if we should accept numpy arrays only
    is_iterable_g = isinstance(g_values, Iterable)
    is_iterable_t = isinstance(t_values, Iterable)
    if not is_iterable_g:
        g_values = np.array([g_values])
    if not is_iterable_t:
        t_values = np.array([t_values])

    # Don't evaluate always treated
    g_values = g_values[g_values > 0]
    t_last = np.max(t_values)

    # Don't evaluate those individuals treated in last period
    if control_group == "not_yet_treated":
        if np.any(t_values > np.max(g_values)):
            # issue a warning
            g_values < -g_values[g_values < t_last]
            warnings.warn("Individuals treated in the last period are excluded from the analysis.")

    t_first = np.min(t_values)

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
