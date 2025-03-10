import math

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021


@pytest.fixture(scope="module", params=["group", "time", "eventstudy"])
def aggregation(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250)],
    ],
)
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["observational", "experimental"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def in_sample_normalization(request):
    return request.param


@pytest.fixture(scope="module", params=[0.1])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module", params=["datetime", "float"])
def time_type(request):
    return request.param


@pytest.fixture(scope="module")
def dml_single_gt_aggregation(aggregation, time_type, learner, score, in_sample_normalization, trimming_threshold):
    n_obs = 500
    dpg = 1

    # collect data
    df = make_did_CS2021(n_obs=n_obs, dgp_type=dpg, time_type=time_type)
    dml_panel_data = dml.data.DoubleMLPanelData(
        df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
    )

    dml_args = {
        "n_folds": 3,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": True,
    }
    gt_combination = [(dml_panel_data.g_values[0], dml_panel_data.t_values[0], dml_panel_data.t_values[3])]
    dml_obj = dml.did.DoubleMLDIDMulti(
        dml_panel_data,
        ml_g=learner[0],
        ml_m=learner[1],
        gt_combinations=gt_combination,
        **dml_args,
    )
    dml_obj.fit()

    dml_obj_agg = dml_obj.aggregate(aggregation=aggregation)

    res_dict = {
        "dml_obj": dml_obj,
        "dml_obj_agg": dml_obj_agg,
    }

    return res_dict


@pytest.mark.ci
def test_dml_single_gt_thetas(dml_single_gt_aggregation):
    assert math.isclose(
        dml_single_gt_aggregation["dml_obj"].coef[0],
        dml_single_gt_aggregation["dml_obj_agg"].aggregated_frameworks.thetas[0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
    assert math.isclose(
        dml_single_gt_aggregation["dml_obj_agg"].aggregated_frameworks.thetas[0],
        dml_single_gt_aggregation["dml_obj_agg"].overall_aggregated_framework.thetas[0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


@pytest.mark.ci
def test_dml_single_gt_ses(dml_single_gt_aggregation):
    assert math.isclose(
        dml_single_gt_aggregation["dml_obj"].se[0],
        dml_single_gt_aggregation["dml_obj_agg"].aggregated_frameworks.ses[0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
    assert math.isclose(
        dml_single_gt_aggregation["dml_obj_agg"].aggregated_frameworks.ses[0],
        dml_single_gt_aggregation["dml_obj_agg"].overall_aggregated_framework.ses[0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
