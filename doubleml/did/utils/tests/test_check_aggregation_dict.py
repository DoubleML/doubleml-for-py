import numpy as np
import pytest

from doubleml.did.utils._aggregation import _check_aggregation_dict


@pytest.fixture
def sample_gt_index():
    """Create a sample gt_index for testing"""
    return np.ma.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], mask=np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    )


def test_valid_aggregation_dict(sample_gt_index):
    """Test a valid aggregation dictionary"""
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )
    valid_dict = {"weight_masks": weight_masks, "agg_names": ["g1", "g2"], "agg_weights": np.array([0.5, 0.5])}
    result = _check_aggregation_dict(valid_dict, sample_gt_index)
    assert isinstance(result, dict)
    assert "weight_masks" in result
    assert "agg_names" in result
    assert "agg_weights" in result


def test_default_values(sample_gt_index):
    """Test default values are set correctly"""
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )
    minimal_dict = {"weight_masks": weight_masks}
    result = _check_aggregation_dict(minimal_dict, sample_gt_index)
    assert len(result["agg_names"]) == 2
    assert all(name == f"Aggregation_{i}" for i, name in enumerate(result["agg_names"]))
    assert np.array_equal(result["agg_weights"], np.array([0.5, 0.5]))


def test_invalid_mask_alignment(sample_gt_index):
    """Test weight_masks with misaligned mask raises ValueError"""
    wrong_mask = ~sample_gt_index.mask
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)), mask=np.broadcast_to(wrong_mask[..., np.newaxis], (*sample_gt_index.shape, 2))
    )
    invalid_dict = {"weight_masks": weight_masks, "agg_names": ["g1", "g2"], "agg_weights": np.array([0.5, 0.5])}
    with pytest.raises(ValueError, match="weight_masks must have the same mask as gt_index"):
        _check_aggregation_dict(invalid_dict, sample_gt_index)


def test_invalid_agg_names_type(sample_gt_index):
    """Test invalid agg_names type raises ValueError"""
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )
    invalid_dict = {
        "weight_masks": weight_masks,
        "agg_names": [1, 2],  # numbers instead of strings
        "agg_weights": np.array([0.5, 0.5]),
    }
    with pytest.raises(ValueError, match="agg_names must be a list of strings"):
        _check_aggregation_dict(invalid_dict, sample_gt_index)


def test_invalid_agg_weights_type(sample_gt_index):
    """Test invalid agg_weights type raises ValueError"""
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )
    invalid_dict = {
        "weight_masks": weight_masks,
        "agg_names": ["g1", "g2"],
        "agg_weights": [0.5, 0.5],  # list instead of numpy array
    }
    with pytest.raises(ValueError, match="agg_weights must be a numpy array"):
        _check_aggregation_dict(invalid_dict, sample_gt_index)


def test_mismatched_lengths(sample_gt_index):
    """Test mismatched lengths between weights and names raises ValueError"""
    weight_masks = np.ma.array(
        np.zeros((*sample_gt_index.shape, 2)),
        mask=np.broadcast_to(sample_gt_index.mask[..., np.newaxis], (*sample_gt_index.shape, 2)),
    )
    invalid_dict = {
        "weight_masks": weight_masks,
        "agg_names": ["g1", "g2", "g3"],  # 3 names but only 2 weights
        "agg_weights": np.array([0.5, 0.5]),
    }
    with pytest.raises(ValueError, match="agg_names must have the same length as the number of aggregations"):
        _check_aggregation_dict(invalid_dict, sample_gt_index)
