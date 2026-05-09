"""Test fit() argument handling on DoubleMLScalar (vehicle: PLR scalar)."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 200
N_FOLDS = 3


def _build_unfitted_plr() -> PLR:
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=10, alpha=0.5)
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
