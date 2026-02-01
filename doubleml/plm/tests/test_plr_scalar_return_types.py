import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 200
N_FOLDS = 3
N_REP = 2
N_REP_BOOT = 314

np.random.seed(3141)
obj_dml_data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=10, alpha=0.5)


@pytest.fixture(scope="module")
def fitted_dml_obj():
    np.random.seed(3141)
    dml_obj = PLR(obj_dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_coef_type_and_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.coef, np.ndarray)
    assert fitted_dml_obj.coef.shape == (1,)


@pytest.mark.ci
def test_se_type_and_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.se, np.ndarray)
    assert fitted_dml_obj.se.shape == (1,)


@pytest.mark.ci
def test_all_thetas_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_thetas, np.ndarray)
    assert fitted_dml_obj.all_thetas.shape == (1, N_REP)


@pytest.mark.ci
def test_all_coef_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_coef, np.ndarray)
    assert fitted_dml_obj.all_coef.shape == (1, N_REP)


@pytest.mark.ci
def test_all_ses_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.all_ses, np.ndarray)
    assert fitted_dml_obj.all_ses.shape == (1, N_REP)


@pytest.mark.ci
def test_summary_type(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.summary, pd.DataFrame)
    assert fitted_dml_obj.summary.shape[0] == 1


@pytest.mark.ci
def test_confint_type_and_shape(fitted_dml_obj):
    ci = fitted_dml_obj.confint()
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (1, 2)


@pytest.mark.ci
def test_confint_joint(fitted_dml_obj):
    ci_joint = fitted_dml_obj.confint(joint=True)
    assert isinstance(ci_joint, pd.DataFrame)
    assert ci_joint.shape == (1, 2)


@pytest.mark.ci
def test_psi_shape(fitted_dml_obj):
    assert isinstance(fitted_dml_obj.psi, np.ndarray)
    assert fitted_dml_obj.psi.shape == (N_OBS, 1, N_REP)


@pytest.mark.ci
def test_predictions_type(fitted_dml_obj):
    preds = fitted_dml_obj.predictions
    assert isinstance(preds, dict)
    assert "ml_l" in preds
    assert "ml_m" in preds
    assert preds["ml_l"].shape == (N_OBS, N_REP)
    assert preds["ml_m"].shape == (N_OBS, N_REP)


@pytest.mark.ci
def test_smpls_type(fitted_dml_obj):
    smpls = fitted_dml_obj.smpls
    assert isinstance(smpls, list)
    assert len(smpls) == N_REP
    assert len(smpls[0]) == N_FOLDS


@pytest.mark.ci
def test_n_properties(fitted_dml_obj):
    assert fitted_dml_obj.n_obs == N_OBS
    assert fitted_dml_obj.n_folds == N_FOLDS
    assert fitted_dml_obj.n_rep == N_REP
    assert fitted_dml_obj.score == "partialling out"


@pytest.mark.ci
def test_learner_names(fitted_dml_obj):
    assert fitted_dml_obj.learner_names == ["ml_l", "ml_m"]
    assert "ml_l" in fitted_dml_obj.learners
    assert "ml_m" in fitted_dml_obj.learners


@pytest.mark.ci
def test_str_repr(fitted_dml_obj):
    assert isinstance(str(fitted_dml_obj), str)
    assert isinstance(repr(fitted_dml_obj), str)


@pytest.mark.ci
def test_before_fit_raises():
    np.random.seed(3141)
    dml_obj = PLR(obj_dml_data)
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef
    with pytest.raises(ValueError, match="Predictions not available. Call fit"):
        _ = dml_obj.predictions
