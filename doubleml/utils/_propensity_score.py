import numpy as np


def _trimm(preds, trimming_rule, trimming_threshold):
    """Trim predictions based on the specified method and threshold."""
    if trimming_rule == "truncate":
        adjusted_predictions = np.clip(a=preds, a_min=trimming_threshold, a_max=1 - trimming_threshold)
    return adjusted_predictions


def _normalize_ipw(propensity, treatment):
    """Normalize inverse probability weights."""
    mean_treat1 = np.mean(np.divide(treatment, propensity))
    mean_treat0 = np.mean(np.divide(1.0 - treatment, 1.0 - propensity))

    normalized_weights = np.multiply(treatment, np.multiply(propensity, mean_treat1)) + np.multiply(
        1.0 - treatment, 1.0 - np.multiply(1.0 - propensity, mean_treat0)
    )

    return normalized_weights
