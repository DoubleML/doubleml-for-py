import numpy as np
import pytest

from doubleml.did.utils._aggregation import _compute_did_time_aggregation_weights


@pytest.mark.ci
def test_basic_functionality_time():
    # Setup basic test data
    gt_index = np.ma.MaskedArray(data=np.ones((2, 3, 3)), mask=np.zeros((2, 3, 3), dtype=bool))
    g_values = np.array([2, 3])
    t_values = np.array([1, 2, 3])
    d_values = np.array([2, 2, 2, 3, 3, 3])
    selected_gt_mask = np.ones((2, 1, 3), dtype=bool)

    result = _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"weight_masks", "agg_names", "agg_weights"}
    assert isinstance(result["weight_masks"], np.ma.MaskedArray)
    assert result["weight_masks"].shape == (*gt_index.shape, 3)  # 3 time periods
    assert result["agg_names"] == ["1", "2", "3"]


@pytest.mark.ci
def test_weight_computation_time():
    gt_index = np.ma.MaskedArray(data=np.ones((2, 3, 3)), mask=np.zeros((2, 3, 3), dtype=bool))
    g_values = np.array([2, 3])
    t_values = np.array([1, 2, 3])
    d_values = np.array([2, 2, 2, 3, 3, 3])

    # Select specific group-time combinations
    selected_gt_mask = np.zeros((2, 3, 3), dtype=bool)
    selected_gt_mask[:, :2, :2] = True  # Select first two time periods for all groups

    result = _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask)

    # Check if number of aggregations is 2 (in this case, time periods 10 and 20)
    assert len(result["agg_names"]) == 2
    assert result["agg_names"] == ["1", "2"]

    # Check weights sum to 1 for each time period
    assert np.allclose(np.sum(result["agg_weights"]), 1.0)

    # Check weight distribution within time periods
    for i in range(result["weight_masks"].shape[-1]):
        time_weights = result["weight_masks"][..., i]
        non_masked_values = time_weights.compressed()
        if len(non_masked_values) > 0:
            assert np.allclose(np.sum(non_masked_values), 1.0)

        # Check if weights in the selected_gt_mask are 0.25
        non_zero = time_weights[selected_gt_mask] != 0
        assert np.allclose(time_weights[selected_gt_mask].data[non_zero], 0.25)


@pytest.mark.ci
def test_no_valid_time_periods():
    gt_index = np.ma.MaskedArray(data=np.ones((2, 2, 2)), mask=np.zeros((2, 2, 2), dtype=bool))
    g_values = np.array([1, 2])
    t_values = np.array([10, 20])
    d_values = np.array([1, 2, 1, 2])
    selected_gt_mask = np.zeros((2, 2, 2), dtype=bool)  # No time periods selected

    with pytest.raises(ValueError, match="No time periods found for aggregation."):
        _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask)


@pytest.mark.ci
def test_single_time_period():
    gt_index = np.ma.MaskedArray(data=np.ones((2, 3, 3)), mask=np.zeros((2, 3, 3), dtype=bool))
    g_values = np.array([2, 3])
    t_values = np.array([1, 2, 3])
    d_values = np.array([2, 2, 2, 3, 3, 3])
    selected_gt_mask = np.ones((2, 1, 1), dtype=bool)

    result = _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask)

    assert len(result["agg_names"]) == 1
    assert result["agg_names"] == ["1"]
    assert result["weight_masks"].shape[-1] == 1
    assert np.allclose(result["agg_weights"], [1.0])


@pytest.mark.ci
def test_masked_input_time():
    # Create data with shape (3,4,4)
    data = np.ones((2, 4, 4))
    mask = np.zeros((2, 4, 4), dtype=bool)

    # Mask some elements in different positions
    mask[0, 0, 0] = True
    mask[1, 2, 1] = True
    mask[1, 1, 2] = True

    gt_index = np.ma.MaskedArray(data=data, mask=mask)
    g_values = np.array([2, 3])  # One value for each group
    t_values = np.array([1, 2, 3, 4])  # One value for each time period
    d_values = np.array([1, 2, 3, 4] * 6)  # Treatment values
    selected_gt_mask = ~mask  # Select all non-masked elements

    result = _compute_did_time_aggregation_weights(gt_index, g_values, t_values, d_values, selected_gt_mask)

    # Check dimensions of output
    assert result["weight_masks"].shape == (2, 4, 4, 4)  # Last dimension is number of time periods

    # Check if masks are maintained
    for time_idx in range(4):
        time_weights = result["weight_masks"][..., time_idx]
        assert np.array_equal(time_weights.mask, mask)

    # Check weight normalization
    for time_idx in range(4):
        weights = result["weight_masks"][..., time_idx].compressed()  # Get non-masked weights
        if len(weights) > 0:
            assert np.isclose(weights.sum(), 1.0)  # Weights should sum to 1 for each time period

    # Check agg_names
    assert result["agg_names"] == ["1", "2", "3", "4"]

    # Check agg_weights sum to 1
    assert np.isclose(sum(result["agg_weights"]), 1.0)
