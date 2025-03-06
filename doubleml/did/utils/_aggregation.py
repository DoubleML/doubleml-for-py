import numpy as np


def _check_aggregation_dict(aggregation_dict, gt_index):
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

    # check if agg_names not exist use default names
    if "agg_names" not in aggregation_dict.keys():
        aggregation_dict["agg_names"] = [f"Aggregation_{i}" for i in range(n_aggregations)]

    if "agg_weights" not in aggregation_dict.keys():
        aggregation_dict["agg_weights"] = np.ones(n_aggregations) / n_aggregations

    # check if agg_names is a list of strings
    if not all(isinstance(name, str) for name in aggregation_dict["agg_names"]):
        raise ValueError("agg_names must be a list of strings")
    # check if agg_weights is a numpy array
    if not isinstance(aggregation_dict["agg_weights"], np.ndarray):
        raise ValueError("agg_weights must be a numpy array")

    # check if length of agg_names equal to the number of aggregations
    if len(aggregation_dict["agg_names"]) != n_aggregations:
        raise ValueError("agg_names must have the same length as the number of aggregations")
    # check if length of agg_weights equal to the number of aggregations
    if len(aggregation_dict["agg_weights"]) != n_aggregations:
        raise ValueError("agg_weights must have the same length as the number of aggregations")

    return aggregation_dict


def _compute_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask):
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
