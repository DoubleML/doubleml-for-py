import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_pliv_partial_x_manual import boot_pliv_partial_x, fit_pliv_partial_x


@pytest.fixture(scope="module", params=[Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["partialling out"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def dml_pliv_partial_x_fixture(generate_data_pliv_partialX, learner, score):
    boot_methods = ["Bayes", "normal", "wild"]
    n_folds = 2
    n_rep_boot = 503

    # collect data
    obj_dml_data = generate_data_pliv_partialX

    # Set machine learning methods for l, m & r
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV._partialX(obj_dml_data, ml_l, ml_m, ml_r, n_folds=n_folds)

    dml_pliv_obj.fit(store_predictions=True)

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    z = obj_dml_data.z
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_pliv_partial_x(y, x, d, z, clone(learner), clone(learner), clone(learner), all_smpls, score)

    res_dict = {
        "coef": dml_pliv_obj.coef.item(),
        "coef_manual": res_manual["theta"],
        "se": dml_pliv_obj.se.item(),
        "se_manual": res_manual["se"],
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_pliv_partial_x(
            y,
            d,
            z,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_l_hat"],
            res_manual["all_m_hat"],
            res_manual["all_r_hat"],
            all_smpls,
            score,
            bootstrap,
            n_rep_boot,
        )

        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict["boot_t_stat" + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


def test_dml_pliv_coef(dml_pliv_partial_x_fixture):
    assert math.isclose(
        dml_pliv_partial_x_fixture["coef"], dml_pliv_partial_x_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4
    )


def test_dml_pliv_se(dml_pliv_partial_x_fixture):
    assert math.isclose(dml_pliv_partial_x_fixture["se"], dml_pliv_partial_x_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_partial_x_fixture):
    for bootstrap in dml_pliv_partial_x_fixture["boot_methods"]:
        assert np.allclose(
            dml_pliv_partial_x_fixture["boot_t_stat" + bootstrap],
            dml_pliv_partial_x_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )
