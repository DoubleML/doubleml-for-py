"""Core multi-treatment estimation accuracy for PLRVector."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso

import doubleml as dml
from doubleml.plm.plr_vector import PLRVector


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def fitted_plr_vector_bivariate(generate_data_bivariate, score):
    """PLRVector fitted on bivariate data with theta = [0.5, 0.9]."""
    data = generate_data_bivariate
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()
    d_cols = data.columns[data.columns.str.startswith("d")].tolist()
    obj_dml_data = dml.DoubleMLData(data, y_col="y", d_cols=d_cols, x_cols=x_cols)

    learner = Lasso(alpha=0.1)
    np.random.seed(3141)
    dml_obj = PLRVector(obj_dml_data, score=score)
    dml_obj.set_learners(ml_l=clone(learner), ml_m=clone(learner), ml_g=clone(learner) if score == "IV-type" else None)
    dml_obj.draw_sample_splitting(n_folds=5, n_rep=1)
    dml_obj.fit()
    return dml_obj, np.array([0.5, 0.9])


@pytest.mark.ci
def test_coef_within_3_sigma(fitted_plr_vector_bivariate):
    """All treatment coefficients fall within 3 SE of the true thetas."""
    dml_obj, true_theta = fitted_plr_vector_bivariate
    assert np.all(np.abs(dml_obj.coef - true_theta) <= 3.0 * dml_obj.se)


@pytest.mark.ci
def test_se_positive(fitted_plr_vector_bivariate):
    """Standard errors are strictly positive for every treatment."""
    dml_obj, _ = fitted_plr_vector_bivariate
    assert np.all(dml_obj.se > 0)


@pytest.mark.ci
def test_coef_shape_matches_d_cols(fitted_plr_vector_bivariate):
    """Coefficient vector has one entry per treatment column."""
    dml_obj, _ = fitted_plr_vector_bivariate
    assert dml_obj.coef.shape == (len(dml_obj._dml_data.d_cols),)
