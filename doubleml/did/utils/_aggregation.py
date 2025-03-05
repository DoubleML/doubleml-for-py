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
