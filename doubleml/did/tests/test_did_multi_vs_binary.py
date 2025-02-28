import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250)],
        [
            RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        ],
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
def dml_did_binary_vs_did_multi_fixture(time_type, learner, score, in_sample_normalization, trimming_threshold):
    n_obs = 500
    dpg = 1
    boot_methods = ["normal"]
    n_rep_boot = 50000

    # collect data
    df = make_did_CS2021(n_obs=n_obs, dgp_type=dpg, time_type=time_type)
    dml_panel_data = dml.data.DoubleMLPanelData(
        df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
    )

    dml_args = {
        "ml_g": clone(learner[0]),
        "ml_m": clone(learner[1]),
        "n_folds": 3,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": True,
    }
    gt_combination = [(dml_panel_data.g_values[0], dml_panel_data.t_values[0], dml_panel_data.t_values[1])]

    dml_did_binary_obj = dml.did.DoubleMLDIDBinary(
        dml_panel_data,
        g_value=gt_combination[0][0],
        t_value_pre=gt_combination[0][1],
        t_value_eval=gt_combination[0][2],
        **dml_args,
    )
    dml_did_binary_obj.fit()

    dml_did_multi_obj = dml.did.DoubleMLDIDMulti(
        dml_panel_data,
        gt_combinations=gt_combination,
        **dml_args,
    )
    dml_did_multi_obj.fit()

    res_dict = {
        "coef_multi": dml_did_multi_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "se_multi": dml_did_multi_obj.se,
        "se_binary": dml_did_binary_obj.se,
    }

    return res_dict


@pytest.mark.ci
def test_coefs(dml_did_binary_vs_did_multi_fixture):
    assert math.isclose(
        dml_did_binary_vs_did_multi_fixture["coef_binary"][0],
        dml_did_binary_vs_did_multi_fixture["coef_multi"][0],
        rel_tol=1e-9,
        abs_tol=1e-4
    )
