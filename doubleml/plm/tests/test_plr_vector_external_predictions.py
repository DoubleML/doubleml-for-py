"""External predictions equivalence for PLRVector across multiple treatments."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.plm.plr_vector import PLRVector


def _make_bivariate_data(n_obs: int = 300, dim_x: int = 5) -> dml.DoubleMLData:
    np.random.seed(42)
    x = np.random.normal(size=(n_obs, dim_x))
    d0 = np.random.normal(size=n_obs)
    d1 = np.random.normal(size=n_obs)
    y = 0.5 * d0 + 0.9 * d1 + x[:, 0] + np.random.normal(size=n_obs)
    df = pd.DataFrame(
        np.column_stack([x, y, d0, d1]),
        columns=[f"X{i + 1}" for i in range(dim_x)] + ["y", "d1", "d2"],
    )
    return dml.DoubleMLData(df, y_col="y", d_cols=["d1", "d2"], x_cols=[f"X{i + 1}" for i in range(dim_x)])


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def external_predictions_fixture(score, n_rep):
    """Fit a reference PLRVector and a second one consuming its predictions externally."""
    n_folds = 3
    obj_dml_data = _make_bivariate_data()
    learner_kwargs: dict[str, object] = {"ml_l": LinearRegression(), "ml_m": LinearRegression()}
    if score == "IV-type":
        learner_kwargs["ml_g"] = LinearRegression()

    np.random.seed(3141)
    dml_ref = PLRVector(obj_dml_data, score=score)
    dml_ref.set_learners(**learner_kwargs)
    dml_ref.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_ref.fit()

    # Build external predictions per treatment, replicating every required learner.
    learner_names = ["ml_l", "ml_m"] + (["ml_g"] if score == "IV-type" else [])
    external_predictions = {
        d_col: {name: dml_ref.modellist[i]._predictions[name] for name in learner_names}
        for i, d_col in enumerate(obj_dml_data.d_cols)
    }

    # Fit a fresh PLRVector consuming the external predictions on identical splits.
    dml_ext = PLRVector(obj_dml_data, score=score)
    dml_ext.set_learners(**learner_kwargs)
    dml_ext.set_sample_splitting(dml_ref.smpls)
    dml_ext.fit(external_predictions=external_predictions)

    return {"ref": dml_ref, "ext": dml_ext}


@pytest.mark.ci
def test_coef_matches_external(external_predictions_fixture):
    """Per-treatment coefficients match the reference fit when fed via external_predictions."""
    ref = external_predictions_fixture["ref"]
    ext = external_predictions_fixture["ext"]
    for i in range(ref.coef.shape[0]):
        assert math.isclose(ref.coef[i], ext.coef[i], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_se_matches_external(external_predictions_fixture):
    """Per-treatment standard errors match the reference fit when fed via external_predictions."""
    ref = external_predictions_fixture["ref"]
    ext = external_predictions_fixture["ext"]
    for i in range(ref.se.shape[0]):
        assert math.isclose(ref.se[i], ext.se[i], rel_tol=1e-9, abs_tol=1e-4)
