import pytest
import pandas as pd

from doubleml import DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018

from sklearn.linear_model import Lasso

dml_data = make_plr_CCDDHNR2018()
ml_g = Lasso()
ml_m = Lasso()
dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m)


def test_doubleml_exception_resampling():
    msg = "The number of folds must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=1.5)
    msg = ('The number of repetitions for the sample splitting must be of int type. '
           "1.5 of type <class 'float'> was passed.")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_rep=1.5)
    msg = 'The number of folds must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=0)
    msg = 'The number of repetitions for the sample splitting must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, n_rep=0)
    msg = 'apply_cross_fitting must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=1)
    msg = 'draw_sample_splitting must be True or False. Got true.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, draw_sample_splitting='true')


def test_doubleml_exception_dml_procedure():
    msg = 'dml_procedure must be "dml1" or "dml2". Got 1.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, dml_procedure='1')
    msg = 'dml_procedure must be "dml1" or "dml2". Got dml.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, dml_procedure='dml')


def test_doubleml_warning_crossfitting_onefold():
    msg = 'apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=True, n_folds=1)


def test_doubleml_exception_no_cross_fit():
    msg = 'Estimation without cross-fitting not supported for n_folds > 2.'
    with pytest.raises(AssertionError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_g, ml_m, apply_cross_fitting=False)


def test_doubleml_exception_get_params():
    msg = 'Invalid nuisance learner ml_r. Valid nuisance learner ml_g or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.get_params('ml_r')


def test_doubleml_exception_smpls():
    msg = ('Sample splitting not specified. '
           r'Either draw samples via .draw_sample splitting\(\) or set external samples via .set_sample_splitting\(\).')
    dml_plr_no_smpls = DoubleMLPLR(dml_data, ml_g, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_plr_no_smpls.smpls


def test_doubleml_exception_fit():
    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.fit(n_jobs_cv='5')
    msg = 'keep_scores must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.fit(keep_scores=1)


def test_doubleml_exception_bootstrap():
    dml_plr_boot = DoubleMLPLR(dml_data, ml_g, ml_m)
    msg = r'Apply fit\(\) before bootstrap\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap()

    dml_plr_boot.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Gaussian.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap(method='Gaussian')
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_boot.bootstrap(n_rep_boot='500')
    msg = 'The number of bootstrap replications must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap(n_rep_boot=0)


def test_doubleml_exception_confint():
    dml_plr_confint = DoubleMLPLR(dml_data, ml_g, ml_m)

    msg = 'joint must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr_confint.confint(joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_confint.confint(level='5%')
    msg = 'The confidence level must be in (0,1). 0.0 was passed.'
    with pytest.raises(TypeError, match=msg):
        dml_plr_confint.confint(level=0.)

    msg = r'Apply fit\(\) before confint\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint()
    msg = r'Apply fit\(\) & bootstrap\(\) before confint\(joint=True\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint(joint=True)
    dml_plr_confint.fit()  # error message should still appear till bootstrap was applied as well
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint(joint=True)
    dml_plr_confint.bootstrap()
    df_ci = dml_plr_confint.confint(joint=True)
    assert isinstance(df_ci, pd.DataFrame)


def test_doubleml_exception_p_adjust():
    dml_plr_p_adjust = DoubleMLPLR(dml_data, ml_g, ml_m)

    msg = r'Apply fit\(\) before p_adjust\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_p_adjust.p_adjust()
    dml_plr_p_adjust.fit()
    msg = r'Apply fit\(\) & bootstrap\(\) before p_adjust'
    with pytest.raises(ValueError, match=msg):
        dml_plr_p_adjust.p_adjust(method='romano-wolf')
    dml_plr_p_adjust.bootstrap()
    p_val = dml_plr_p_adjust.p_adjust(method='romano-wolf')
    assert isinstance(p_val, pd.DataFrame)

    msg = "The p_adjust method must be of str type. 0.05 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_p_adjust.p_adjust(method=0.05)


def test_doubleml_exception_tune():

    msg = r'Invalid param_grids \[0.05, 0.5\]. param_grids must be a dictionary with keys ml_g and ml_m'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune([0.05, 0.5])
    msg = (r"Invalid param_grids {'ml_g': {'alpha': \[0.05, 0.5\]}}. "
           "param_grids must be a dictionary with keys ml_g and ml_m.")
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune({'ml_g': {'alpha': [0.05, 0.5]}})

    param_grids = {'ml_g': {'alpha': [0.05, 0.5]}, 'ml_m': {'alpha': [0.05, 0.5]}}
    msg = ('Invalid scoring_methods neg_mean_absolute_error. '
           'scoring_methods must be a dictionary. '
           'Valid keys are ml_g and ml_m.')
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, scoring_methods='neg_mean_absolute_error')

    msg = 'tune_on_folds must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, tune_on_folds=1)

    msg = 'The number of folds used for tuning must be at least two. 1 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, n_folds_tune=1)
    msg = "The number of folds used for tuning must be of int type. 1.0 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_folds_tune=1.)

    msg = 'search_mode must be "grid_search" or "randomized_search". Got gridsearch.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, search_mode='gridsearch')

    msg = 'The number of parameter settings sampled for the randomized search must be at least two. 1 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, n_iter_randomized_search=1)
    msg = ("The number of parameter settings sampled for the randomized search must be of int type. "
           "1.0 of type <class 'float'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_iter_randomized_search=1.)

    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_jobs_cv='5')

    msg = 'set_as_params must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, set_as_params=1)

    msg = 'return_tune_res must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, return_tune_res=1)


