import pytest

from doubleml import DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018

from sklearn.linear_model import Lasso

dml_data = make_plr_CCDDHNR2018()
ml_g = Lasso()
ml_m = Lasso()
dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m)


def test_doubleml_exception_resampling():
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=1.5)
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_rep=1.5)
    with pytest.raises(ValueError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=0)
    with pytest.raises(ValueError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_rep=0)
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=1)
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, draw_sample_splitting='true')


def test_doubleml_exception_dml_procedure():
    with pytest.raises(ValueError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, dml_procedure='1')
    with pytest.raises(ValueError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, dml_procedure='dml')


def test_doubleml_warning_crossfitting_onefold():
    with pytest.warns(UserWarning,
                      match='apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.'):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=True, n_folds=1)


def test_doubleml_exception_no_cross_fit():
    with pytest.raises(AssertionError):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=False)


def test_doubleml_exception_get_params():
    with pytest.raises(ValueError):
        dml_plr.get_params('ml_r')


def test_doubleml_exception_smpls():
    dml_plr_no_smpls = DoubleMLPLR(dml_data, ml_g, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError):
        _ = dml_plr_no_smpls.smpls


def test_doubleml_exception_fit():
    with pytest.raises(TypeError):
        dml_plr.fit(n_jobs_cv='5')
    with pytest.raises(TypeError):
        dml_plr.fit(keep_scores=1)


def test_doubleml_exception_bootstrap():
    dml_plr_boot = DoubleMLPLR(dml_data, ml_g, ml_m)
    with pytest.raises(ValueError):
        dml_plr_boot.bootstrap()  # 'apply fit() before bootstrap()'

    dml_plr_boot.fit()
    with pytest.raises(ValueError):
        dml_plr_boot.bootstrap(method='Gaussian')
    with pytest.raises(TypeError):
        dml_plr_boot.bootstrap(n_rep_boot='500')
    with pytest.raises(ValueError):
        dml_plr_boot.bootstrap(n_rep_boot=0)
