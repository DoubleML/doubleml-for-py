import numpy as np
import pytest

from doubleml.did.utils._aggregation import _compute_did_group_aggregation_weights


@pytest.mark.ci
def test_basic_functionality():
    # Setup basic test data
    gt_index = np.ma.MaskedArray(data=np.ones((3, 2, 1)), mask=np.zeros((3, 2, 1), dtype=bool))
    g_values = np.array([1, 2, 3])
    d_values = np.array([1, 2, 1, 2, 1, 2])
    selected_gt_mask = np.ones((3, 2, 1), dtype=bool)

    result = _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"weight_masks", "agg_names", "agg_weights"}
    assert isinstance(result["weight_masks"], np.ma.MaskedArray)
    assert result["weight_masks"].shape == (*gt_index.shape, 3)  # 3 groups


@pytest.mark.ci
def test_weight_computation():
    gt_index = np.ma.MaskedArray(data=np.ones((3, 4, 4)), mask=np.zeros((3, 4, 4), dtype=bool))
    g_values = np.array([1, 2, 3])
    d_values = np.array([1, 2, 1, 2, 1, 1, 1, 1, 3, 3])

    # select some group-time combinations
    selected_gt_mask = gt_index.mask.copy()
    selected_gt_mask[:2, :2, 0] = True

    result = _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask)

    # check if the number of aggregations is 2 (in this case, group 1 and group 2)
    assert len(result["agg_names"]) == 2

    # Check weights sum to 1 for each group
    assert np.allclose(np.sum(result["agg_weights"]), 1.0)

    # Check weight distribution within groups
    for i in range(result["weight_masks"].shape[-1]):
        group_weights = result["weight_masks"][..., i]
        if len(group_weights) > 0:
            assert np.allclose(np.sum(group_weights.compressed()), 1.0)

        # check if weights in the selected_gt_mask are 0.5
        assert np.allclose(group_weights[i, ...][selected_gt_mask[i, ...]], 0.5)

    # check if the aggregation weights are [0.75, 0.25]
    assert np.allclose(result["agg_weights"], np.array([0.75, 0.25]))


@pytest.mark.ci
def test_no_valid_groups():
    gt_index = np.ma.MaskedArray(data=np.ones((2, 2, 1)), mask=np.zeros((2, 2, 1), dtype=bool))
    g_values = np.array([1, 2])
    d_values = np.array([1, 2, 1, 2])
    selected_gt_mask = np.zeros((2, 2, 1), dtype=bool)  # No groups selected

    with pytest.raises(ValueError, match="No valid groups found for aggregation."):
        _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask)


@pytest.mark.ci
def test_single_group():
    gt_index = np.ma.MaskedArray(data=np.ones((1, 2, 1)), mask=np.zeros((1, 2, 1), dtype=bool))
    g_values = np.array([1])
    d_values = np.array([1, 1])
    selected_gt_mask = np.ones((1, 2, 1), dtype=bool)

    result = _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask)

    assert len(result["agg_names"]) == 1
    assert result["weight_masks"].shape[-1] == 1
    assert np.allclose(result["agg_weights"], [1.0])


@pytest.mark.ci
def test_masked_input():
    # Create data with shape (3,4,4)
    data = np.ones((3, 4, 4))
    mask = np.zeros((3, 4, 4), dtype=bool)

    # Mask some elements in different positions
    mask[0, 0, 0] = True
    mask[1, 2, 3] = True
    mask[2, 1, 1] = True

    gt_index = np.ma.MaskedArray(data=data, mask=mask)
    g_values = np.array([1, 2, 3])  # One value for each group
    d_values = np.array([1, 2, 3] * 16)  # Treatment values matching the data size
    selected_gt_mask = ~mask  # Select all masked elements

    result = _compute_did_group_aggregation_weights(gt_index, g_values, d_values, selected_gt_mask)

    # Check dimensions of output
    assert result["weight_masks"].shape == (3, 4, 4, 3)  # Last dimension is number of groups

    for group_idx in range(3):
        group_weights = result["weight_masks"][..., group_idx]
        assert np.array_equal(group_weights.mask, mask)

    # Check weight normalization
    for group_idx in range(3):
        weights = result["weight_masks"][..., group_idx].compressed()  # Get non-masked weights
        assert np.isclose(weights.sum(), 1.0)  # Weights should sum to 1 for each group

    # Check agg_names
    assert result["agg_names"] == ["1", "2", "3"]

    # Check agg_weights sum to 1
    assert np.isclose(sum(result["agg_weights"]), 1.0)
