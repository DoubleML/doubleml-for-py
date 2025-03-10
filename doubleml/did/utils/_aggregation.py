import numpy as np


def _check_did_aggregation_dict(aggregation_dict, gt_index):
    if not isinstance(aggregation_dict, dict):
        raise ValueError("aggregation must be a dictionary")

    # Validate and extract custom parameters
    required_keys = {"weight_masks"}
    if not all(key in aggregation_dict for key in required_keys):
        raise ValueError(f"aggregation must contain all required keys: {required_keys}")

    # Check if weight_masks is a masked numpy array
    weight_masks = aggregation_dict["weight_masks"]
    if not isinstance(weight_masks, np.ma.MaskedArray):
        raise ValueError("weight_masks must be a numpy masked array")

    # check if weight_masks has 4 dim
    if weight_masks.ndim != 4:
        raise ValueError("weight_masks must have 4 dimensions")

    # Check if weight_masks has the same first three dimensions as gt_index
    if weight_masks.shape[:-1] != gt_index.shape:
        raise ValueError(
            f"weight_masks must have shape {gt_index.shape} + (n,) where n is the number of aggregations. "
            f"Got shape {weight_masks.shape}"
        )

    n_aggregations = weight_masks.shape[-1]
    # check if every weight_mask along last axis has the same mask as gt_index
    for i in range(n_aggregations):
        if not np.array_equal(weight_masks[..., i].mask, gt_index.mask):
            raise ValueError("weight_masks must have the same mask as gt_index")

    return aggregation_dict


def _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask):
    """
    Calculate weights for aggregating treatment effects by group.

    Parameters
    ----------
    gt_index : numpy.ma.MaskedArray
        Masked array containing group-time indices
    g_values : array-like
        Array of unique group values
    d_values : array-like
        Array of treatment values
    selected_gt_mask : numpy.ndarray
        Boolean mask indicating which group-time combinations to include

    Returns
    -------
    dict
        Dictionary containing:
        - weight_masks: numpy.ma.MaskedArray with weights for each group
        - agg_names: list of group names
        - agg_weights: numpy.ndarray of aggregation weights
    """
    selected_gt_indicies = np.where(selected_gt_mask)
    selected_unique_g_indices = np.unique(selected_gt_indicies[0])
    n_agg_effects = len(selected_unique_g_indices)

    if n_agg_effects == 0:
        raise ValueError("No valid groups found for aggregation.")

    agg_names = [None] * n_agg_effects
    agg_weights = [np.nan] * n_agg_effects

    # Create a weight mask (0 weights) for each of the groups
    weight_masks = np.ma.masked_array(
        data=np.zeros((*gt_index.shape, n_agg_effects)),
        mask=np.broadcast_to(gt_index.mask[..., np.newaxis], (*gt_index.shape, n_agg_effects)),
        dtype=np.float64,
    )

    # Write weight masks
    for idx_agg, g_idx in enumerate(selected_unique_g_indices):
        # Set group name & weights
        current_group = g_values[g_idx]
        agg_names[idx_agg] = str(current_group)
        agg_weights[idx_agg] = (d_values == current_group).mean()

        # Group weights_masks
        group_gt_indicies = [(i, j, k) for i, j, k in zip(*selected_gt_indicies) if i == g_idx]

        weight = 1 / len(group_gt_indicies)
        for i, j, k in group_gt_indicies:
            weight_masks.data[i, j, k, idx_agg] = weight

    # Normalize weights
    agg_weights = np.array(agg_weights) / sum(agg_weights)

    return {"weight_masks": weight_masks, "agg_names": agg_names, "agg_weights": agg_weights}


def _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask):
    """
    Calculate weights for aggregating treatment effects over time periods.

    Parameters
    ----------
    gt_index : numpy.ma.MaskedArray
        Masked array containing group-time indices
    g_values : array-like
        Array of unique group values
    t_values : array-like
        Array of unique time period values
    d_values : array-like
        Array of treatment values (g_values for each id)
    selected_gt_mask : numpy.ndarray
        Boolean mask indicating which group-time combinations to include

    Returns
    -------
    dict
        Dictionary containing:
        - weight_masks: numpy.ma.MaskedArray with weights for each group
        - agg_names: list of group names
        - agg_weights: numpy.ndarray of aggregation weights
    """
    selected_gt_indicies = np.where(selected_gt_mask)
    selected_unique_t_eval_indices = np.unique(selected_gt_indicies[2])
    n_agg_effects = len(selected_unique_t_eval_indices)

    if n_agg_effects == 0:
        raise ValueError("No time periods found for aggregation.")

    agg_names = [None] * n_agg_effects
    # equal weight due to balanced panel
    agg_weights = np.ones(n_agg_effects) / n_agg_effects

    # Create a weight mask (0 weights) for each of the groups
    weight_masks = np.ma.masked_array(
        data=np.zeros((*gt_index.shape, n_agg_effects)),
        mask=np.broadcast_to(gt_index.mask[..., np.newaxis], (*gt_index.shape, n_agg_effects)),
        dtype=np.float64,
    )

    group_weights = np.zeros(len(g_values))
    selected_unique_g_indices = np.unique(selected_gt_indicies[0])
    for g_idx in selected_unique_g_indices:
        group_weights[g_idx] = (d_values == g_values[g_idx]).mean()  # (requires balanced panel)

    # Write weight masks
    for idx_agg, t_eval_idx in enumerate(selected_unique_t_eval_indices):
        # Set time period name
        current_time_period = t_values[t_eval_idx]
        agg_names[idx_agg] = str(current_time_period)

        # time weights_masks
        time_gt_indicies = [(i, j, k) for i, j, k in zip(*selected_gt_indicies) if k == t_eval_idx]

        for i, j, k in time_gt_indicies:
            weight_masks.data[i, j, k, idx_agg] = group_weights[i]

        # normalize weights
        weight_masks.data[..., idx_agg] = weight_masks.data[..., idx_agg] / np.sum(weight_masks.data[..., idx_agg])

    return {"weight_masks": weight_masks, "agg_names": agg_names, "agg_weights": agg_weights}


def _compute_did_eventstudy_aggregation_weights(gt_index, g_values, t_values, d_values, time_values, selected_gt_mask):
    """
    Calculate weights for aggregating treatment effects over time periods.

    Parameters
    ----------
    gt_index : numpy.ma.MaskedArray
        Masked array containing group-time indices
    g_values : array-like
        Array of unique group values
    t_values : array-like
        Array of unique evaluation time values
    d_values : array-like
        Array of treatment values (g_values for each id)
    time_values : array-like
        Array of evaluation time values (t_values for each id)
    selected_gt_mask : numpy.ndarray
        Boolean mask indicating which group-time combinations to include

    Returns
    -------
    dict
        Dictionary containing:
        - weight_masks: numpy.ma.MaskedArray with weights for each group
        - agg_names: list of group names
        - agg_weights: numpy.ndarray of aggregation weights
    """
    selected_gt_indicies = np.where(selected_gt_mask)
    eventtime = time_values - d_values
    e_values = np.unique(eventtime)
    selected_unique_e_values = np.unique([t_values[k] - g_values[i] for i, _, k in zip(*selected_gt_indicies)])
    assert np.all(np.isin(selected_unique_e_values, e_values))
    n_agg_effects = len(selected_unique_e_values)

    if n_agg_effects == 0:
        raise ValueError("No time periods found for aggregation.")

    agg_names = [None] * n_agg_effects
    agg_weights = np.zeros(n_agg_effects)
    agg_weights[selected_unique_e_values >= 0] = 1 / np.sum(selected_unique_e_values >= 0)

    # Create a weight mask (0 weights) for each of the groups
    weight_masks = np.ma.masked_array(
        data=np.zeros((*gt_index.shape, n_agg_effects)),
        mask=np.broadcast_to(gt_index.mask[..., np.newaxis], (*gt_index.shape, n_agg_effects)),
        dtype=np.float64,
    )

    group_weights = np.zeros(len(g_values))
    selected_unique_g_indices = np.unique(selected_gt_indicies[0])
    for g_idx in selected_unique_g_indices:
        group_weights[g_idx] = (d_values == g_values[g_idx]).mean()  # (requires balanced panel)

    # Write weight masks
    for idx_agg, e_val in enumerate(selected_unique_e_values):
        # Set time period name
        agg_names[idx_agg] = str(e_val)

        # time weights_masks
        eventtime_gt_indicies = [(i, j, k) for i, j, k in zip(*selected_gt_indicies) if t_values[k] - g_values[i] == e_val]

        for i, j, k in eventtime_gt_indicies:
            weight_masks.data[i, j, k, idx_agg] = group_weights[i]

        # normalize weights
        weight_masks.data[..., idx_agg] = weight_masks.data[..., idx_agg] / np.sum(weight_masks.data[..., idx_agg])

    return {"weight_masks": weight_masks, "agg_names": agg_names, "agg_weights": agg_weights}
