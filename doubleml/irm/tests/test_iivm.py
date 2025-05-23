import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_iivm_manual import boot_iivm, fit_iivm


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250)],
        [RandomForestRegressor(max_depth=2, n_estimators=10), RandomForestClassifier(max_depth=2, n_estimators=10)],
    ],
)
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["LATE"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module")
def dml_iivm_fixture(generate_data_iivm, learner, score, normalize_ipw, trimming_threshold):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 491

    # collect data
    data = generate_data_iivm
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()
    y = data["y"].values
    x = data.loc[:, x_cols].values
    d = data["d"].values
    z = data["z"].values

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])
    ml_r = clone(learner[1])

    np.random.seed(3141)
    n_obs = len(y)
    strata = d + 2 * z
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=strata)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, "y", ["d"], x_cols, "z")
    dml_iivm_obj = dml.DoubleMLIIVM(
        obj_dml_data,
        ml_g,
        ml_m,
        ml_r,
        n_folds,
        draw_sample_splitting=False,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )
    # synchronize the sample splitting
    dml_iivm_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_iivm_obj.fit()

    np.random.seed(3141)

    res_manual = fit_iivm(
        y,
        x,
        d,
        z,
        clone(learner[0]),
        clone(learner[1]),
        clone(learner[1]),
        all_smpls,
        score,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )

    res_dict = {
        "coef": dml_iivm_obj.coef.item(),
        "coef_manual": res_manual["theta"],
        "se": dml_iivm_obj.se.item(),
        "se_manual": res_manual["se"],
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_iivm(
            y,
            d,
            z,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_g_hat0"],
            res_manual["all_g_hat1"],
            res_manual["all_m_hat"],
            res_manual["all_r_hat0"],
            res_manual["all_r_hat1"],
            all_smpls,
            score,
            bootstrap,
            n_rep_boot,
            normalize_ipw=normalize_ipw,
        )

        np.random.seed(3141)
        dml_iivm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_iivm_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_iivm_coef(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture["coef"], dml_iivm_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_se(dml_iivm_fixture):
    assert math.isclose(dml_iivm_fixture["se"], dml_iivm_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_iivm_boot(dml_iivm_fixture):
    for bootstrap in dml_iivm_fixture["boot_methods"]:
        assert np.allclose(
            dml_iivm_fixture["boot_t_stat" + bootstrap],
            dml_iivm_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )
