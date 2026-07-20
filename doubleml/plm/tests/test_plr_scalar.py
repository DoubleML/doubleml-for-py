import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR


@pytest.fixture(scope="module", params=[LinearRegression(), Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_scalar_fixture(learner, score):
    n_folds = 5
    theta = 0.5

    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)

    np.random.seed(3141)
    obj_dml_data = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=theta)

    dml_obj = PLR(obj_dml_data, score=score)
    if score == "IV-type":
        dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)
    else:
        dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m)
    dml_obj.draw_sample_splitting(n_folds=n_folds)
    dml_obj.fit()

    res_dict = {
        "coef": dml_obj.coef[0],
        "se": dml_obj.se[0],
        "true_coef": theta,
    }

    return res_dict


@pytest.mark.ci
def test_dml_plr_scalar_coef(dml_plr_scalar_fixture):
    coef = dml_plr_scalar_fixture["coef"]
    se = dml_plr_scalar_fixture["se"]
    true_coef = dml_plr_scalar_fixture["true_coef"]
    assert abs(coef - true_coef) <= 3.0 * se


@pytest.fixture(scope="module")
def dml_plr_scalar_rep_fixture():
    """Test with multiple repetitions."""
    n_folds = 3
    n_rep = 3
    theta = 0.5

    np.random.seed(3141)
    obj_dml_data = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=theta)

    dml_obj = PLR(obj_dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_obj.fit()

    return {
        "dml_obj": dml_obj,
        "true_coef": theta,
        "n_rep": n_rep,
    }


@pytest.mark.ci
def test_dml_plr_scalar_rep_coef(dml_plr_scalar_rep_fixture):
    dml_obj = dml_plr_scalar_rep_fixture["dml_obj"]
    true_coef = dml_plr_scalar_rep_fixture["true_coef"]
    assert abs(dml_obj.coef[0] - true_coef) <= 3.0 * dml_obj.se[0]


@pytest.mark.ci
def test_dml_plr_scalar_rep_shapes(dml_plr_scalar_rep_fixture):
    dml_obj = dml_plr_scalar_rep_fixture["dml_obj"]
    n_rep = dml_plr_scalar_rep_fixture["n_rep"]
    assert dml_obj.all_thetas.shape == (1, n_rep)
    assert dml_obj.all_ses.shape == (1, n_rep)
