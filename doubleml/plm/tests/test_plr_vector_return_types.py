"""Validate PLRVector return types and reset behavior."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.plm.plr_vector import PLRVector

N_OBS = 200
N_FOLDS = 3
N_REP = 2
N_REP_BOOT = 251


def _make_data(n_obs: int = N_OBS, dim_x: int = 5) -> dml.DoubleMLData:
    """Build a small bivariate-treatment DoubleMLData for return-type tests."""
    np.random.seed(7)
    x = np.random.normal(size=(n_obs, dim_x))
    d0 = np.random.normal(size=n_obs)
    d1 = np.random.normal(size=n_obs)
    y = 0.5 * d0 + 0.9 * d1 + x[:, 0] + np.random.normal(size=n_obs)
    df = pd.DataFrame(
        np.column_stack([x, y, d0, d1]),
        columns=[f"X{i + 1}" for i in range(dim_x)] + ["y", "d1", "d2"],
    )
    return dml.DoubleMLData(df, y_col="y", d_cols=["d1", "d2"], x_cols=[f"X{i + 1}" for i in range(dim_x)])


N_TREAT = 2  # tied to _make_data


@pytest.fixture(scope="module")
def fitted_plr_vector():
    """Fit a PLRVector once and share across tests."""
    np.random.seed(3141)
    obj_dml_data = _make_data()
    dml_obj = PLRVector(obj_dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_coef_type_and_shape(fitted_plr_vector):
    """coef is a 1D array with one entry per treatment."""
    assert isinstance(fitted_plr_vector.coef, np.ndarray)
    assert fitted_plr_vector.coef.shape == (N_TREAT,)


@pytest.mark.ci
def test_se_type_and_shape(fitted_plr_vector):
    """se is a 1D array with one entry per treatment."""
    assert isinstance(fitted_plr_vector.se, np.ndarray)
    assert fitted_plr_vector.se.shape == (N_TREAT,)


@pytest.mark.ci
def test_all_thetas_shape(fitted_plr_vector):
    """all_thetas is (n_treat, n_rep)."""
    assert fitted_plr_vector.all_thetas.shape == (N_TREAT, N_REP)


@pytest.mark.ci
def test_all_ses_shape(fitted_plr_vector):
    """all_ses is (n_treat, n_rep)."""
    assert fitted_plr_vector.all_ses.shape == (N_TREAT, N_REP)


@pytest.mark.ci
def test_summary_index_matches_d_cols(fitted_plr_vector):
    """summary is a DataFrame indexed by d_cols in declaration order."""
    summary = fitted_plr_vector.summary
    assert isinstance(summary, pd.DataFrame)
    assert summary.shape[0] == N_TREAT
    assert summary.index.tolist() == ["d1", "d2"]


@pytest.mark.ci
def test_confint_shape(fitted_plr_vector):
    """confint returns (n_treat, 2) DataFrame."""
    ci = fitted_plr_vector.confint()
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (N_TREAT, 2)


@pytest.mark.ci
def test_confint_joint_shape(fitted_plr_vector):
    """confint(joint=True) returns (n_treat, 2) DataFrame after bootstrap."""
    ci = fitted_plr_vector.confint(joint=True)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (N_TREAT, 2)


@pytest.mark.ci
def test_psi_shape(fitted_plr_vector):
    """psi has shape (n_obs, n_treat, n_rep)."""
    assert fitted_plr_vector.psi.shape == (N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_modellist_length_and_type(fitted_plr_vector):
    """modellist exposes one PLR scalar sub-model per treatment."""
    from doubleml.plm.plr_scalar import PLR

    models = fitted_plr_vector.modellist
    assert isinstance(models, list)
    assert len(models) == N_TREAT
    assert all(isinstance(m, PLR) for m in models)


@pytest.mark.ci
def test_smpls_shared_across_submodels(fitted_plr_vector):
    """Sample splits are propagated identically into each sub-model."""
    parent_smpls = fitted_plr_vector.smpls
    for model in fitted_plr_vector.modellist:
        for i_rep in range(N_REP):
            for j_fold in range(N_FOLDS):
                np.testing.assert_array_equal(model.smpls[i_rep][j_fold][0], parent_smpls[i_rep][j_fold][0])
                np.testing.assert_array_equal(model.smpls[i_rep][j_fold][1], parent_smpls[i_rep][j_fold][1])


@pytest.mark.ci
def test_n_properties(fitted_plr_vector):
    """n_obs, n_folds, n_rep, score reflect configuration."""
    assert fitted_plr_vector.n_obs == N_OBS
    assert fitted_plr_vector.n_folds == N_FOLDS
    assert fitted_plr_vector.n_rep == N_REP
    assert fitted_plr_vector.score == "partialling out"


@pytest.mark.ci
def test_required_learners(fitted_plr_vector):
    """required_learners is score-dependent and matches scalar PLR."""
    assert fitted_plr_vector.required_learners == ["ml_l", "ml_m"]


@pytest.mark.ci
def test_get_params_returns_per_submodel_list(fitted_plr_vector):
    """get_params returns one parameter dict per sub-model, in d_cols order."""
    params = fitted_plr_vector.get_params("ml_l")
    assert isinstance(params, list)
    assert len(params) == N_TREAT
    for p in params:
        assert isinstance(p, dict)
        assert "fit_intercept" in p


@pytest.mark.ci
def test_set_params_updates_all_submodels(fitted_plr_vector):
    """set_params propagates to every sub-model and returns self."""
    result = fitted_plr_vector.set_params("ml_l", fit_intercept=False)
    assert result is fitted_plr_vector
    params = fitted_plr_vector.get_params("ml_l")
    assert all(p["fit_intercept"] is False for p in params)
    fitted_plr_vector.set_params("ml_l", fit_intercept=True)


@pytest.mark.ci
def test_sensitivity_elements_shape(fitted_plr_vector):
    """sensitivity_elements exposes framework-level keys with multi-treatment shapes."""
    elems = fitted_plr_vector.sensitivity_elements
    assert isinstance(elems, dict)
    for key in ["sigma2", "nu2", "max_bias"]:
        assert elems[key].shape == (1, N_TREAT, N_REP)
    assert elems["psi_max_bias"].shape == (N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_treatment_names_set_on_framework(fitted_plr_vector):
    """treatment_names on the framework match d_cols."""
    assert fitted_plr_vector.framework.treatment_names == ["d1", "d2"]


@pytest.mark.ci
def test_before_fit_raises():
    """Properties relying on framework raise before fit()."""
    np.random.seed(3141)
    dml_obj = PLRVector(_make_data())
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef


@pytest.mark.ci
def test_reset_after_draw_sample_splitting():
    """draw_sample_splitting clears framework and fitted properties on vector and sub-models."""
    np.random.seed(3141)
    dml_obj = PLRVector(_make_data())
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    _ = dml_obj.framework
    _ = dml_obj.coef

    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.framework
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.coef
    for model in dml_obj.modellist:
        with pytest.raises(ValueError, match="framework is not yet initialized"):
            _ = model.framework


@pytest.mark.ci
def test_reset_after_set_learners():
    """set_learners after fit clears the vector framework so stale results aren't returned."""
    np.random.seed(3141)
    dml_obj = PLRVector(_make_data())
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj.fit(n_folds=N_FOLDS, n_rep=N_REP)
    _ = dml_obj.framework

    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    with pytest.raises(ValueError, match="framework is not yet initialized"):
        _ = dml_obj.framework
    for model in dml_obj.modellist:
        with pytest.raises(ValueError, match="framework is not yet initialized"):
            _ = model.framework
