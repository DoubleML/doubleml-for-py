import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression

import doubleml as dml

from ..datasets import make_plpr_CP2025


@pytest.fixture(scope="module", params=[LinearRegression(), Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=["cre_general", "cre_normal", "fd_exact", "wg_approx"])
def approach(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plpr_fixture(
    learner,
    score,
    approach,
):
    n_folds = 5
    theta = 0.5

    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)

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
    dml_plpr_obj = dml.DoubleMLPLPR(
        obj_dml_data,
        ml_l,
        ml_m,
        ml_g,
        n_folds=n_folds,
        score=score,
        approach=approach,
    )

    dml_plpr_obj.fit()

    res_dict = {
        "coef": dml_plpr_obj.coef[0],
        "se": dml_plpr_obj.se[0],
        "true_coef": theta,
    }

    return res_dict


@pytest.mark.ci
def test_dml_selection_coef(dml_plpr_fixture):
    # true_coef should lie within three standard deviations of the estimate
    coef = dml_plpr_fixture["coef"]
    se = dml_plpr_fixture["se"]
    true_coef = dml_plpr_fixture["true_coef"]
    assert abs(coef - true_coef) <= 3.0 * se
