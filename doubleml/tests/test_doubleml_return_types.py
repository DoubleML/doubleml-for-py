import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data

from sklearn.linear_model import Lasso, LogisticRegression

dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=100)
dml_data_iivm = make_iivm_data(n_obs=100)

dml_plr = DoubleMLPLR(dml_data_plr, Lasso(), Lasso())
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression())

plr_dml1 = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(), dml_procedure='dml1', n_rep=2, n_folds=3)
plr_dml1.fit()
plr_dml1.bootstrap(n_rep_boot=314)


@pytest.mark.ci
def test_plr_return_types():
    # ToDo: A second test case with multiple treatment variables would be helpful
    assert isinstance(dml_plr.__str__(), str)
    assert isinstance(dml_plr.summary, pd.DataFrame)
    assert isinstance(dml_plr.draw_sample_splitting(), DoubleMLPLR)
    assert isinstance(dml_plr.set_sample_splitting(dml_plr.smpls), DoubleMLPLR)
    assert isinstance(dml_plr.fit(), DoubleMLPLR)
    assert isinstance(dml_plr.__str__(), str)  # called again after fit, now with numbers
    assert isinstance(dml_plr.summary, pd.DataFrame)  # called again after fit, now with numbers
    assert isinstance(dml_plr.bootstrap(), DoubleMLPLR)
    assert isinstance(dml_plr.confint(), pd.DataFrame)
    assert isinstance(dml_plr.p_adjust(), pd.DataFrame)
    assert isinstance(dml_plr.get_params('ml_m'), dict)
    assert isinstance(dml_plr._dml_data.__str__(), str)

    # for the following checks we need additional inputs
    # assert isinstance(dml_plr.set_ml_nuisance_params(), DoubleMLPLR)
    # assert isinstance(dml_plr.tune(), DoubleMLPLR)


@pytest.mark.ci
def test_plr_property_types_and_shapes():
    # not checked: apply_cross_fitting, dml_procedure, learner, learner_names, params, params_names, score
    # already checked: summary

    # check that the setting is still in line with the hard-coded values
    n_treat = 1
    n_rep = 2
    n_folds = 3
    n_obs = 100
    n_rep_boot = 314
    assert plr_dml1._dml_data.n_treat == n_treat
    assert plr_dml1.n_rep == n_rep
    assert plr_dml1.n_folds == n_folds
    assert plr_dml1._dml_data.n_obs == n_obs
    assert plr_dml1.n_rep_boot == n_rep_boot

    assert isinstance(plr_dml1.all_coef, np.ndarray)
    assert plr_dml1.all_coef.shape == (n_treat, n_rep)

    assert isinstance(plr_dml1.all_dml1_coef, np.ndarray)
    assert plr_dml1.all_dml1_coef.shape == (n_treat, n_rep, n_folds)

    assert isinstance(plr_dml1.all_se, np.ndarray)
    assert plr_dml1.all_se.shape == (n_treat, n_rep)

    assert isinstance(plr_dml1.boot_coef, np.ndarray)
    assert plr_dml1.boot_coef.shape == (n_treat, (n_rep_boot * n_rep))

    assert isinstance(plr_dml1.boot_t_stat, np.ndarray)
    assert plr_dml1.boot_t_stat.shape == (n_treat, (n_rep_boot * n_rep))

    assert isinstance(plr_dml1.coef, np.ndarray)
    assert plr_dml1.coef.shape == (n_treat, )

    assert isinstance(plr_dml1.psi, np.ndarray)
    assert plr_dml1.psi.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(plr_dml1.psi_a, np.ndarray)
    assert plr_dml1.psi_a.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(plr_dml1.psi_b, np.ndarray)
    assert plr_dml1.psi_b.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(plr_dml1.pval, np.ndarray)
    assert plr_dml1.pval.shape == (n_treat, )

    assert isinstance(plr_dml1.se, np.ndarray)
    assert plr_dml1.se.shape == (n_treat, )

    assert isinstance(plr_dml1.t_stat, np.ndarray)
    assert plr_dml1.t_stat.shape == (n_treat, )

    assert isinstance(plr_dml1.smpls, list)
    assert len(plr_dml1.smpls) == n_rep
    all_tuple = all([all([isinstance(tpl, tuple) for tpl in smpl]) for smpl in plr_dml1.smpls])
    assert all_tuple
    all_pairs = all([all([len(tpl) == 2 for tpl in smpl]) for smpl in plr_dml1.smpls])
    assert all_pairs
    n_folds_each_smpl = np.array([len(smpl) for smpl in plr_dml1.smpls])
    assert np.all(n_folds_each_smpl == n_folds_each_smpl[0])
    assert n_folds_each_smpl[0] == n_folds
