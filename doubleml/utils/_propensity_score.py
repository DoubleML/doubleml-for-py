import numpy as np


def _propensity_score_adjustment(propensity_score, treatment_indicator, normalize_ipw):
    """Adjust propensity score."""
    if normalize_ipw:
        adjusted_propensity_score = _normalize_ipw(propensity_score=propensity_score, treatment_indicator=treatment_indicator)
    else:
        adjusted_propensity_score = propensity_score

    return adjusted_propensity_score


def _trimm(preds, trimming_rule, trimming_threshold):
    """Trim predictions based on the specified method and threshold."""
    if trimming_rule == "truncate":
        adjusted_predictions = np.clip(a=preds, a_min=trimming_threshold, a_max=1 - trimming_threshold)
    else:
        adjusted_predictions = preds
    return adjusted_predictions


def _normalize_ipw(propensity_score, treatment_indicator):
    """Normalize inverse probability weights."""
    mean_treat1 = np.mean(np.divide(treatment_indicator, propensity_score))
    mean_treat0 = np.mean(np.divide(1.0 - treatment_indicator, 1.0 - propensity_score))

    normalized_weights = np.multiply(treatment_indicator, np.multiply(propensity_score, mean_treat1)) + np.multiply(
        1.0 - treatment_indicator, 1.0 - np.multiply(1.0 - propensity_score, mean_treat0)
    )

    return normalized_weights
