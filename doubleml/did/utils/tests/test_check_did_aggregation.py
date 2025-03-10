import numpy as np
import pytest

from doubleml.did.utils._aggregation import _check_did_aggregation_dict


@pytest.fixture
def sample_gt_index():
    """Create a sample gt_index for testing"""
    return np.ma.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], mask=np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    )


@pytest.fixture
def valid_weight_masks(sample_gt_index):
    """Create valid weight masks for testing"""
    return np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )


@pytest.mark.ci
def test_valid_aggregation_dict(sample_gt_index, valid_weight_masks):
    """Test a valid aggregation dictionary"""
    valid_dict = {"weight_masks": valid_weight_masks, "agg_names": ["g1", "g2"], "agg_weights": np.array([0.5, 0.5])}
    result = _check_did_aggregation_dict(valid_dict, sample_gt_index)
    assert isinstance(result, dict)
    assert "weight_masks" in result


@pytest.mark.ci
@pytest.mark.parametrize(
    "invalid_input,error_msg",
    [
        ("not_a_dict", "aggregation must be a dictionary"),
        ({}, "aggregation must contain all required keys: {'weight_masks'}"),
        ({"weight_masks": np.array([1, 2, 3])}, "weight_masks must be a numpy masked array"),
    ],
)
def test_invalid_input_types(sample_gt_index, invalid_input, error_msg):
    """Test various invalid input types"""
    with pytest.raises(ValueError, match=error_msg):
        _check_did_aggregation_dict(invalid_input, sample_gt_index)


@pytest.mark.ci
def test_invalid_dimensions(sample_gt_index):
    """Test weight_masks with wrong number of dimensions"""
    wrong_dims = np.ma.array(np.zeros((sample_gt_index.shape)), mask=sample_gt_index.mask)  # Only 3 dimensions
    invalid_dict = {"weight_masks": wrong_dims}
    with pytest.raises(ValueError, match="weight_masks must have 4 dimensions"):
        _check_did_aggregation_dict(invalid_dict, sample_gt_index)


@pytest.mark.ci
def test_invalid_shape(sample_gt_index):
    """Test weight_masks with wrong shape"""
    wrong_shape = np.ma.array(
        np.zeros((3, 3, 3, 2)), mask=np.zeros((3, 3, 3, 2), dtype=bool)  # Wrong shape for first 3 dimensions
    )
    invalid_dict = {"weight_masks": wrong_shape}
    with pytest.raises(ValueError, match=r"weight_masks must have shape .* \+ \(n,\)"):
        _check_did_aggregation_dict(invalid_dict, sample_gt_index)


@pytest.mark.ci
def test_invalid_mask_alignment(sample_gt_index):
    """Test weight_masks with misaligned mask"""
    wrong_mask = ~sample_gt_index.mask
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)), mask=np.broadcast_to(wrong_mask[..., np.newaxis], (*sample_gt_index.shape, 2))
    )
    invalid_dict = {"weight_masks": weight_masks}
    with pytest.raises(ValueError, match="weight_masks must have the same mask as gt_index"):
        _check_did_aggregation_dict(invalid_dict, sample_gt_index)


@pytest.mark.ci
def test_multiple_weight_masks(sample_gt_index, valid_weight_masks):
    """Test multiple weight masks with different masks"""
    # Create a weight_masks array with multiple aggregations
    weight_masks = np.ma.concatenate([valid_weight_masks, valid_weight_masks], axis=-1)
    # Modify mask of last aggregation
    weight_masks[..., -1].mask = ~weight_masks[..., -1].mask

    invalid_dict = {"weight_masks": weight_masks}
    with pytest.raises(ValueError, match="weight_masks must have the same mask as gt_index"):
        _check_did_aggregation_dict(invalid_dict, sample_gt_index)
