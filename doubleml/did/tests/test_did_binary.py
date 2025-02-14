import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_manual import boot_did, fit_did, fit_sensitivity_elements_did


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


@pytest.fixture(scope="module")
def dml_did_binary_fixture(generate_data_did_binary, learner, score, in_sample_normalization, trimming_threshold):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 499

    # collect data
    dml_panel_data = generate_data_did_binary

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    dml_did_binary_obj = dml.did.DoubleMLDIDBinary(
        dml_panel_data,
        ml_g,
        ml_m,
        g_value=1,
        t_value_pre=0,
        t_value_eval=1,
        n_folds=n_folds,
        score=score,
        in_sample_normalization=in_sample_normalization,
        trimming_threshold=trimming_threshold,
    )

    np.random.seed(3141)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    np.random.seed(3141)
    dml_did_obj = dml.DoubleMLDID(
        obj_dml_data,
        ml_g,
        ml_m,
        n_folds,
        score=score,
        in_sample_normalization=in_sample_normalization,
        draw_sample_splitting=False,
        trimming_threshold=trimming_threshold,
    )

    # synchronize the sample splitting
    dml_did_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_did_obj.fit()

    np.random.seed(3141)
    res_manual = fit_did(
        y,
        x,
        d,
        clone(learner[0]),
        clone(learner[1]),
        all_smpls,
        score,
        in_sample_normalization,
        trimming_threshold=trimming_threshold,
    )

    res_dict = {
        "coef": dml_did_obj.coef,
        "coef_manual": res_manual["theta"],
        "se": dml_did_obj.se,
        "se_manual": res_manual["se"],
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_did(
            y,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_psi_a"],
            res_manual["all_psi_b"],
            all_smpls,
            bootstrap,
            n_rep_boot,
        )

        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_did_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_did_obj.sensitivity_elements
    res_dict["sensitivity_elements_manual"] = fit_sensitivity_elements_did(
        y,
        d,
        all_coef=dml_did_obj.all_coef,
        predictions=dml_did_obj.predictions,
        score=score,
        in_sample_normalization=in_sample_normalization,
        n_rep=1,
    )

    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_did_obj.sensitivity_analysis(rho=0.0)
    res_dict["sensitivity_ses"] = dml_did_obj.sensitivity_params["se"]

    return res_dict


@pytest.mark.ci
def test_dml_did_binary_coef(dml_did_binary_fixture):
    assert math.isclose(dml_did_binary_fixture["coef"][0], dml_did_binary_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)
