import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml

from ..datasets import make_lplr_LZZ2020


@pytest.fixture(scope="module", params=[RandomForestClassifier(random_state=42)])
def learner_M(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42)])
def learner_t(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42)])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42)])
def learner_a(request):
    return request.param


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def score(request):
    return request.param


def get_par_grid():
    return {"n_estimators": [5, 10, 20]}


@pytest.fixture(scope="module")
def dml_lplr_fixture(
    learner_M,
    learner_t,
    learner_m,
    learner_a,
    score,
    tune_on_folds=False,
):
    par_grid = {
        "ml_M": get_par_grid(),
        "ml_t": get_par_grid(),
        "ml_m": get_par_grid(),
        "ml_a": get_par_grid(),
    }
    n_folds_tune = 4
    n_folds = 5
    alpha = 0.5

    ml_M = clone(learner_M)
    ml_t = clone(learner_t)
    ml_m = clone(learner_m)
    ml_a = clone(learner_a)

    obj_dml_data = make_lplr_LZZ2020(alpha=alpha)
    dml_sel_obj = dml.DoubleMLLPLR(
        obj_dml_data,
        ml_M,
        ml_t,
        ml_m,
        ml_a=ml_a,
        n_folds=n_folds,
        score=score,
    )

    # tune hyperparameters
    tune_res = dml_sel_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLLPLR)

    dml_sel_obj.fit()

    res_dict = {
        "coef": dml_sel_obj.coef[0],
        "se": dml_sel_obj.se[0],
        "true_coef": alpha,
    }

    return res_dict


@pytest.mark.ci
def test_dml_selection_coef(dml_lplr_fixture):
    # true_coef should lie within three standard deviations of the estimate
    coef = dml_lplr_fixture["coef"]
    se = dml_lplr_fixture["se"]
    true_coef = dml_lplr_fixture["true_coef"]
    assert abs(coef - true_coef) <= 3.0 * np.sqrt(se)
