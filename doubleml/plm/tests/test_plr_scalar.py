import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression

from doubleml import DoubleMLData
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


@pytest.mark.ci
@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
def test_dml_plr_scalar_small_scale_treatment(score):
    """Rescaling the treatment rescales coef and se, also when E[psi_a] is tiny in absolute terms."""
    scale = 1e-7  # E[psi_a] is about -1e-14 for the rescaled treatment
    np.random.seed(3141)
    df = make_plr_CCDDHNR2018(n_obs=500, dim_x=5, alpha=0.5, return_type="DataFrame")
    x_cols = [c for c in df.columns if c.startswith("X")]

    models = []
    for d_scale in [1.0, scale]:
        dml_data = DoubleMLData(df.assign(d=df["d"] * d_scale), y_col="y", d_cols="d", x_cols=x_cols)
        dml_obj = PLR(dml_data, score=score)
        if score == "IV-type":
            dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression(), ml_g=LinearRegression())
        else:
            dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
        if models:
            dml_obj.set_sample_splitting(models[0].smpls)
        else:
            dml_obj.draw_sample_splitting(n_folds=5)
        dml_obj.fit()
        models.append(dml_obj)

    np.testing.assert_allclose(models[1].coef * scale, models[0].coef, rtol=1e-9)
    np.testing.assert_allclose(models[1].se * scale, models[0].se, rtol=1e-9)
