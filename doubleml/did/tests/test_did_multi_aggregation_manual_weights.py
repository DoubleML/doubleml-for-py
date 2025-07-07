import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021
from doubleml.did.utils._aggregation import (
    _compute_did_eventstudy_aggregation_weights,
    _compute_did_group_aggregation_weights,
    _compute_did_time_aggregation_weights,
)


@pytest.fixture(scope="module", params=["group", "time", "eventstudy"])
def aggregation_method(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def panel(request):
    return request.param


@pytest.fixture(scope="module", params=["observational", "experimental"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def dml_fitted_obj(panel, score):
    """Create a fitted DML object for testing."""
    n_obs = 200

    # Create data
    df = make_did_CS2021(n_obs=n_obs, dgp_type=1, time_type="float")
    dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

    # Create and fit model
    ml_g = LinearRegression()
    ml_m = LogisticRegression(solver="lbfgs", max_iter=250)

    dml_obj = dml.did.DoubleMLDIDMulti(
        obj_dml_data=dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        gt_combinations="standard",
        panel=panel,
        score=score,
        n_folds=3,
        n_rep=1,
    )
    dml_obj.fit()

    return dml_obj


def _extract_manual_weights(dml_obj, aggregation_method):
    """Extract manual weights from the aggregation method."""
    # Get the mask for non-masked values
    selected_gt_mask = ~dml_obj.gt_index.mask

    if aggregation_method == "group":
        # Exclude pre-treatment combinations for group aggregation
        selected_gt_mask = selected_gt_mask & dml_obj._post_treatment_mask
        aggregation_dict = _compute_did_group_aggregation_weights(
            gt_index=dml_obj.gt_index,
            g_values=dml_obj.g_values,
            d_values=dml_obj._dml_data.d,
            selected_gt_mask=selected_gt_mask,
        )
        aggregation_dict["method"] = "Group"
    elif aggregation_method == "time":
        # Exclude pre-treatment combinations for time aggregation
        selected_gt_mask = selected_gt_mask & dml_obj._post_treatment_mask
        aggregation_dict = _compute_did_time_aggregation_weights(
            gt_index=dml_obj.gt_index,
            g_values=dml_obj.g_values,
            t_values=dml_obj.t_values,
            d_values=dml_obj._dml_data.d,
            selected_gt_mask=selected_gt_mask,
        )
        aggregation_dict["method"] = "Time"
    else:
        assert aggregation_method == "eventstudy"
        aggregation_dict = _compute_did_eventstudy_aggregation_weights(
            gt_index=dml_obj.gt_index,
            g_values=dml_obj.g_values,
            t_values=dml_obj.t_values,
            d_values=dml_obj._dml_data.d,
            time_values=dml_obj._dml_data.t,
            selected_gt_mask=selected_gt_mask,
        )
        aggregation_dict["method"] = "Event Study"
    return aggregation_dict


@pytest.mark.ci
def test_string_vs_manual_weights_aggregation(dml_fitted_obj, aggregation_method):
    """Test that string aggregation methods produce identical results to manual weights."""

    # Get string-based aggregation result
    agg_string = dml_fitted_obj.aggregate(aggregation=aggregation_method)

    # Extract manual weights
    manual_weights_dict = _extract_manual_weights(dml_fitted_obj, aggregation_method)

    # Get manual aggregation result
    agg_manual = dml_fitted_obj.aggregate(aggregation=manual_weights_dict)

    # Compare aggregated frameworks - coefficients
    np.testing.assert_allclose(
        agg_string.aggregated_frameworks.thetas,
        agg_manual.aggregated_frameworks.thetas,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare aggregated frameworks - standard errors
    np.testing.assert_allclose(
        agg_string.aggregated_frameworks.ses,
        agg_manual.aggregated_frameworks.ses,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare overall aggregated framework - coefficients
    np.testing.assert_allclose(
        agg_string.overall_aggregated_framework.thetas,
        agg_manual.overall_aggregated_framework.thetas,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare overall aggregated framework - standard errors
    np.testing.assert_allclose(
        agg_string.overall_aggregated_framework.ses,
        agg_manual.overall_aggregated_framework.ses,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare aggregation weights
    np.testing.assert_allclose(
        agg_string.aggregation_weights,
        agg_manual.aggregation_weights,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare overall aggregation weights
    np.testing.assert_allclose(
        agg_string.overall_aggregation_weights,
        agg_manual.overall_aggregation_weights,
        rtol=1e-9,
        atol=1e-12,
    )

    # Compare aggregation names
    assert agg_string.aggregation_names == agg_manual.aggregation_names

    # Compare number of aggregations
    assert agg_string.n_aggregations == agg_manual.n_aggregations


@pytest.mark.ci
def test_manual_weights_properties(dml_fitted_obj, aggregation_method):
    """Test that manual weights have the expected properties."""

    manual_weights_dict = _extract_manual_weights(dml_fitted_obj, aggregation_method)

    # Check that required keys are present
    assert "weight_masks" in manual_weights_dict
    assert "agg_names" in manual_weights_dict
    assert "agg_weights" in manual_weights_dict

    weight_masks = manual_weights_dict["weight_masks"]
    agg_weights = manual_weights_dict["agg_weights"]

    # Check weight masks properties
    assert isinstance(weight_masks, np.ma.MaskedArray)
    assert weight_masks.ndim == 4
    assert weight_masks.shape[:-1] == dml_fitted_obj.gt_index.shape

    # Check that aggregation weights sum to 1
    assert math.isclose(np.sum(agg_weights), 1.0, rel_tol=1e-9, abs_tol=1e-12)

    # Check that individual weight masks sum to 1 (for non-masked elements)
    n_aggregations = weight_masks.shape[-1]
    for i in range(n_aggregations):
        weights = weight_masks[..., i].compressed()
        if len(weights) > 0:
            assert math.isclose(np.sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-12)

    # Check that weight masks have the same mask as gt_index
    for i in range(n_aggregations):
        np.testing.assert_array_equal(weight_masks[..., i].mask, dml_fitted_obj.gt_index.mask)
