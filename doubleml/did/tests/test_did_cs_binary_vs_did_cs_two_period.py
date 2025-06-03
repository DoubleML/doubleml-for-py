import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_cs_manual import fit_did_cs
from ._utils_did_manual import boot_did


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
def dml_did_cs_binary_vs_did_cs_fixture(generate_data_did_binary, learner, score, in_sample_normalization, trimming_threshold):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 499

    # collect data
    dml_panel_data = generate_data_did_binary
    df = dml_panel_data._data.sort_values(by=["id", "t"])

    n_obs = df.shape[0]
    all_smpls = draw_smpls(n_obs, n_folds)
    obj_dml_data = dml.DoubleMLData(df, y_col="y", d_cols="d", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    dml_args = {
        "ml_g": ml_g,
        "ml_m": ml_m,
        "n_folds": n_folds,
        "score": score,
        "in_sample_normalization": in_sample_normalization,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": False,
    }

    dml_did_binary_obj = dml.did.DoubleMLDIDCSBinary(
        dml_panel_data,
        g_value=1,
        t_value_pre=0,
        t_value_eval=1,
        **dml_args,
    )

    dml_did_obj = dml.DoubleMLDIDCS(
        obj_dml_data,
        **dml_args,
    )

    # synchronize the sample splitting
    dml_did_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_did_binary_obj.set_sample_splitting(all_smpls=all_smpls)

    dml_did_obj.fit()
    dml_did_binary_obj.fit()

    # manual fit
    y = df["y"].values
    d = df["d"].values
    x = df[["Z1", "Z2", "Z3", "Z4"]].values
    t = df["t"].values

    np.random.seed(3141)
    res_manual = fit_did_cs(
        y,
        x,
        d,
        t,
        clone(learner[0]),
        clone(learner[1]),
        all_smpls,
        score,
        in_sample_normalization,
        trimming_threshold=trimming_threshold,
    )

    res_dict = {
        "coef": dml_did_obj.coef,
        "coef_binary": dml_did_binary_obj.coef,
        "coef_manual": res_manual["theta"],
        "se": dml_did_obj.se,
        "se_binary": dml_did_binary_obj.se,
        "se_manual": res_manual["se"],
        "nuisance_loss": dml_did_obj.nuisance_loss,
        "nuisance_loss_binary": dml_did_binary_obj.nuisance_loss,
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
        np.random.seed(3141)
        dml_did_binary_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict["boot_t_stat" + bootstrap] = dml_did_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_binary"] = dml_did_binary_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_coefs(dml_did_cs_binary_vs_did_cs_fixture):
    assert math.isclose(
        dml_did_cs_binary_vs_did_cs_fixture["coef"][0],
        dml_did_cs_binary_vs_did_cs_fixture["coef_manual"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
    assert math.isclose(
        dml_did_cs_binary_vs_did_cs_fixture["coef_binary"][0],
        dml_did_cs_binary_vs_did_cs_fixture["coef"][0],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
