import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso

import doubleml as dml

from ..datasets import make_plpr_CP2025


@pytest.fixture(scope="module", params=[Lasso()])
def learner_l(request):
    return request.param


@pytest.fixture(scope="module", params=[Lasso()])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=[Lasso()])
def learner_g(request):
    return request.param


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=["cre_general", "cre_normal", "fd_exact", "wg_approx"])
def approach(request):
    return request.param


def get_par_grid():
    par_grid = {"alpha": np.linspace(0.05, 0.95, 7)}
    return par_grid


@pytest.fixture(scope="module")
def dml_plpr_fixture(
    learner_l,
    learner_m,
    learner_g,
    score,
    approach,
    tune_on_folds=False,
):
    par_grid = {
        "ml_l": get_par_grid(),
        "ml_m": get_par_grid(),
        "ml_g": get_par_grid(),
    }
    n_folds_tune = 4
    n_folds = 5
    theta = 0.5

    ml_l = clone(learner_l)
    ml_m = clone(learner_m)
    ml_g = clone(learner_g)

    np.random.seed(3141)
    plpr_data = make_plpr_CP2025(theta=theta)
    obj_dml_data = dml.DoubleMLPanelData(
        plpr_data,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_sel_obj = dml.DoubleMLPLPR(
        obj_dml_data,
        ml_l,
        ml_m,
        ml_g,
        n_folds=n_folds,
        score=score,
        approach=approach,
    )

    # tune hyperparameters
    tune_res = dml_sel_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLPLPR)

    dml_sel_obj.fit()

    res_dict = {
        "coef": dml_sel_obj.coef[0],
        "se": dml_sel_obj.se[0],
        "true_coef": theta,
    }

    return res_dict


@pytest.mark.ci
def test_dml_plpr_coef(dml_plpr_fixture):
    # true_coef should lie within three standard deviations of the estimate
    coef = dml_plpr_fixture["coef"]
    se = dml_plpr_fixture["se"]
    true_coef = dml_plpr_fixture["true_coef"]
    assert abs(coef - true_coef) <= 3.0 * se
