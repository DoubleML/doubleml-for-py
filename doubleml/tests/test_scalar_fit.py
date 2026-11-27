"""Test fit() argument handling on DoubleMLScalar (vehicle: PLR scalar)."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021, make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 200
N_FOLDS = 3


def _build_unfitted_plr() -> PLR:
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=10, alpha=0.5)
    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    return dml_obj


def _build_unfitted_plr_multiway_cluster() -> PLR:
    np.random.seed(3141)
    df = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, dim_X=5, return_type="DataFrame")
    x_cols = [c for c in df.columns if c.startswith("X")]
    dml_data = DoubleMLData(df, y_col="Y", d_cols="D", x_cols=x_cols, cluster_cols=["cluster_var_i", "cluster_var_j"])
    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    return dml_obj


@pytest.mark.ci
def test_fit_redraws_on_n_rep_mismatch():
    """fit(n_rep=...) re-draws splits and warns when n_rep differs from existing splits."""
    dml_obj = _build_unfitted_plr()
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=1)
    msg = r"Re-drawing sample splitting"
    with pytest.warns(UserWarning, match=msg):
        dml_obj.fit(n_rep=3)
    assert dml_obj.n_rep == 3
    assert dml_obj.n_folds == N_FOLDS  # n_folds preserved
    assert dml_obj.all_thetas.shape == (1, 3)


@pytest.mark.ci
def test_fit_redraws_on_n_folds_mismatch():
    """fit(n_folds=...) re-draws splits and warns when n_folds differs from existing splits."""
    dml_obj = _build_unfitted_plr()
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=2)
    msg = r"Re-drawing sample splitting"
    with pytest.warns(UserWarning, match=msg):
        dml_obj.fit(n_folds=N_FOLDS + 2)
    assert dml_obj.n_folds == N_FOLDS + 2
    assert dml_obj.n_rep == 2  # n_rep preserved


@pytest.mark.ci
def test_fit_no_warning_when_consistent():
    """fit(n_rep, n_folds) matching existing splits emits no UserWarning and keeps splits."""
    dml_obj = _build_unfitted_plr()
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=2)
    original_smpls = dml_obj.smpls
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dml_obj.fit(n_folds=N_FOLDS, n_rep=2)
    # smpls were not redrawn
    assert dml_obj.smpls is original_smpls


@pytest.mark.ci
def test_fit_no_warning_when_args_omitted():
    """fit() with no args emits no UserWarning even when splits differ from defaults."""
    dml_obj = _build_unfitted_plr()
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=2)
    original_smpls = dml_obj.smpls
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dml_obj.fit()
    assert dml_obj.n_rep == 2
    assert dml_obj.n_folds == N_FOLDS
    assert dml_obj.smpls is original_smpls


@pytest.mark.ci
def test_fit_draws_default_splits_when_none_set():
    """fit() without prior draw_sample_splitting() falls back to default n_folds=5, n_rep=1."""
    dml_obj = _build_unfitted_plr()
    dml_obj.fit()
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1


@pytest.mark.ci
def test_fit_draws_explicit_splits_when_none_set():
    """fit(n_folds, n_rep) without prior draw_sample_splitting() honors the args without warning."""
    dml_obj = _build_unfitted_plr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dml_obj.fit(n_folds=4, n_rep=2)
    assert dml_obj.n_folds == 4
    assert dml_obj.n_rep == 2


@pytest.mark.ci
@pytest.mark.parametrize("use_set_sample_splitting", [False, True])
def test_fit_multiway_cluster_no_warning_when_consistent(use_set_sample_splitting):
    """With two-way clustering, fit(n_folds=k) with the k of the drawn or set splits keeps them."""
    dml_obj = _build_unfitted_plr_multiway_cluster()
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=1)
    if use_set_sample_splitting:
        dml_obj.set_sample_splitting(dml_obj.smpls, dml_obj.smpls_cluster)
    original_smpls = dml_obj.smpls
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dml_obj.fit(n_folds=2, n_rep=1)
    assert dml_obj.smpls is original_smpls
    assert dml_obj.n_folds == 4  # 2 folds per cluster variable


@pytest.mark.ci
@pytest.mark.parametrize("fit_kwargs, k_new, n_rep_new", [({"n_rep": 2}, 2, 2), ({"n_folds": 3}, 3, 1)])
def test_fit_multiway_cluster_redraw(fit_kwargs, k_new, n_rep_new):
    """With two-way clustering, fit() re-draws with n_folds interpreted per cluster variable."""
    dml_obj = _build_unfitted_plr_multiway_cluster()
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=1)
    msg = (
        r"Re-drawing sample splitting \(was n_folds=2 per cluster variable, n_rep=1; "
        rf"now n_folds={k_new} per cluster variable, n_rep={n_rep_new}\)"
    )
    with pytest.warns(UserWarning, match=msg):
        dml_obj.fit(**fit_kwargs)
    assert dml_obj.n_rep == n_rep_new
    assert dml_obj.n_folds == k_new**2
    assert all(len(smpl) == k_new**2 for smpl in dml_obj.smpls)
