import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR


@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def plr_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_l_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_plr_scalar_fixture(plr_score, n_rep, set_ml_m_ext, set_ml_l_ext, set_ml_g_ext):
    n_folds = 3
    ext_predictions = {}

    x, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, return_type="np.array")

    np.random.seed(3141)
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    # Fit reference model
    dml_plr = PLR(dml_data, score=plr_score)
    if plr_score == "IV-type":
        dml_plr.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression(), ml_g=LinearRegression())
    else:
        dml_plr.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    np.random.seed(3141)
    dml_plr.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_plr.fit()

    # Build external predictions dict
    if set_ml_m_ext:
        ext_predictions["ml_m"] = dml_plr.predictions["ml_m"]

    if set_ml_l_ext:
        ext_predictions["ml_l"] = dml_plr.predictions["ml_l"]

    if plr_score == "IV-type" and set_ml_g_ext:
        ext_predictions["ml_g"] = dml_plr.predictions["ml_g"]

    # Fit model with external predictions — only set learners that are needed
    dml_plr_ext = PLR(dml_data, score=plr_score)
    learner_kwargs = {}
    if not set_ml_l_ext:
        learner_kwargs["ml_l"] = LinearRegression()
    if not set_ml_m_ext:
        learner_kwargs["ml_m"] = LinearRegression()
    if plr_score == "IV-type" and not set_ml_g_ext:
        learner_kwargs["ml_g"] = LinearRegression()
    if learner_kwargs:
        dml_plr_ext.set_learners(**learner_kwargs)

    np.random.seed(3141)
    dml_plr_ext.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_plr_ext.fit(external_predictions=ext_predictions if ext_predictions else None)

    res_dict = {
        "coef_normal": dml_plr.coef[0],
        "coef_ext": dml_plr_ext.coef[0],
        "se_normal": dml_plr.se[0],
        "se_ext": dml_plr_ext.se[0],
    }

    return res_dict


@pytest.mark.ci
def test_doubleml_plr_scalar_coef(doubleml_plr_scalar_fixture):
    assert math.isclose(
        doubleml_plr_scalar_fixture["coef_normal"],
        doubleml_plr_scalar_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


@pytest.mark.ci
def test_doubleml_plr_scalar_se(doubleml_plr_scalar_fixture):
    assert math.isclose(
        doubleml_plr_scalar_fixture["se_normal"],
        doubleml_plr_scalar_fixture["se_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
