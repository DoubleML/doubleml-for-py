import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def irm_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g0_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g1_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_irm_scalar_fixture(irm_score, n_rep, set_ml_g0_ext, set_ml_g1_ext, set_ml_m_ext):
    n_folds = 3
    ext_predictions = {}

    np.random.seed(42)
    data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type="DoubleMLData")

    ml_g = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

    # Fit reference model
    dml_irm = IRM(data, score=irm_score)
    dml_irm.set_learners(ml_g=ml_g, ml_m=ml_m)
    np.random.seed(3141)
    dml_irm.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_irm.fit()

    # Build external predictions dict
    if set_ml_g0_ext:
        ext_predictions["ml_g0"] = dml_irm.predictions["ml_g0"]

    if set_ml_g1_ext:
        ext_predictions["ml_g1"] = dml_irm.predictions["ml_g1"]

    if set_ml_m_ext:
        ext_predictions["ml_m"] = dml_irm.predictions["ml_m"]

    # Fit model with external predictions — only set learners that are needed
    dml_irm_ext = IRM(data, score=irm_score)
    learner_kwargs = {}
    if not (set_ml_g0_ext and set_ml_g1_ext):
        learner_kwargs["ml_g"] = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    if not set_ml_m_ext:
        learner_kwargs["ml_m"] = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    if learner_kwargs:
        dml_irm_ext.set_learners(**learner_kwargs)

    np.random.seed(3141)
    dml_irm_ext.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_irm_ext.fit(external_predictions=ext_predictions if ext_predictions else None)

    res_dict = {
        "coef_normal": dml_irm.coef[0],
        "coef_ext": dml_irm_ext.coef[0],
        "se_normal": dml_irm.se[0],
        "se_ext": dml_irm_ext.se[0],
    }

    return res_dict


@pytest.mark.ci
def test_doubleml_irm_scalar_coef(doubleml_irm_scalar_fixture):
    assert math.isclose(
        doubleml_irm_scalar_fixture["coef_normal"],
        doubleml_irm_scalar_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


@pytest.mark.ci
def test_doubleml_irm_scalar_se(doubleml_irm_scalar_fixture):
    assert math.isclose(
        doubleml_irm_scalar_fixture["se_normal"],
        doubleml_irm_scalar_fixture["se_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
