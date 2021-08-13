import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV, DoubleMLClusterData
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data,\
    make_pliv_multiway_cluster_CKMS2021

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)
dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=100)
dml_data_iivm = make_iivm_data(n_obs=100)
dml_cluster_data_pliv = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)

dml_plr = DoubleMLPLR(dml_data_plr, Lasso(), Lasso())
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression())
dml_pliv_cluster = DoubleMLPLIV(dml_cluster_data_pliv, Lasso(), Lasso(), Lasso())


@pytest.mark.ci
@pytest.mark.parametrize('dml_obj, cls',
                         [(dml_plr, DoubleMLPLR),
                          (dml_pliv, DoubleMLPLIV),
                          (dml_irm, DoubleMLIRM),
                          (dml_iivm, DoubleMLIIVM),
                          (dml_pliv_cluster, DoubleMLPLIV)])
def test_return_types(dml_obj, cls):
    # ToDo: A second test case with multiple treatment variables would be helpful
    assert isinstance(dml_obj.__str__(), str)
    assert isinstance(dml_obj.summary, pd.DataFrame)
    assert isinstance(dml_obj.draw_sample_splitting(), cls)
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.set_sample_splitting(dml_obj.smpls), cls)
    else:
        assert isinstance(dml_obj._dml_data, DoubleMLClusterData)
    assert isinstance(dml_obj.fit(), cls)
    assert isinstance(dml_obj.__str__(), str)  # called again after fit, now with numbers
    assert isinstance(dml_obj.summary, pd.DataFrame)  # called again after fit, now with numbers
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.bootstrap(), cls)
    else:
        assert isinstance(dml_obj._dml_data, DoubleMLClusterData)
    assert isinstance(dml_obj.confint(), pd.DataFrame)
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.p_adjust(), pd.DataFrame)
    else:
        isinstance(dml_obj.p_adjust('bonferroni'), pd.DataFrame)
    assert isinstance(dml_obj.get_params('ml_m'), dict)
    assert isinstance(dml_obj._dml_data.__str__(), str)

    # for the following checks we need additional inputs
    # assert isinstance(dml_obj.set_ml_nuisance_params(), cls)
    # assert isinstance(dml_obj.tune(), cls)


n_treat = 1
n_rep = 2
n_folds = 3
n_obs = 100
n_rep_boot = 314

plr_dml1 = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(),
                       dml_procedure='dml1', n_rep=n_rep, n_folds=n_folds)
plr_dml1.fit()
plr_dml1.bootstrap(n_rep_boot=n_rep_boot)

pliv_dml1 = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso(),
                         dml_procedure='dml1', n_rep=n_rep, n_folds=n_folds)
pliv_dml1.fit()
pliv_dml1.bootstrap(n_rep_boot=n_rep_boot)

irm_dml1 = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(),
                       dml_procedure='dml1', n_rep=n_rep, n_folds=n_folds)
irm_dml1.fit()
irm_dml1.bootstrap(n_rep_boot=n_rep_boot)

iivm_dml1 = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         dml_procedure='dml1', n_rep=n_rep, n_folds=n_folds)
iivm_dml1.fit()
iivm_dml1.bootstrap(n_rep_boot=n_rep_boot)


@pytest.mark.ci
@pytest.mark.parametrize('dml_obj',
                         [plr_dml1, pliv_dml1,  irm_dml1,  iivm_dml1])
def test_property_types_and_shapes(dml_obj):
    # not checked: apply_cross_fitting, dml_procedure, learner, learner_names, params, params_names, score
    # already checked: summary

    # check that the setting is still in line with the hard-coded values
    assert dml_obj._dml_data.n_treat == n_treat
    assert dml_obj.n_rep == n_rep
    assert dml_obj.n_folds == n_folds
    assert dml_obj._dml_data.n_obs == n_obs
    assert dml_obj.n_rep_boot == n_rep_boot

    assert isinstance(dml_obj.all_coef, np.ndarray)
    assert dml_obj.all_coef.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.all_dml1_coef, np.ndarray)
    assert dml_obj.all_dml1_coef.shape == (n_treat, n_rep, n_folds)

    assert isinstance(dml_obj.all_se, np.ndarray)
    assert dml_obj.all_se.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.boot_coef, np.ndarray)
    assert dml_obj.boot_coef.shape == (n_treat, (n_rep_boot * n_rep))

    assert isinstance(dml_obj.boot_t_stat, np.ndarray)
    assert dml_obj.boot_t_stat.shape == (n_treat, (n_rep_boot * n_rep))

    assert isinstance(dml_obj.coef, np.ndarray)
    assert dml_obj.coef.shape == (n_treat, )

    assert isinstance(dml_obj.psi, np.ndarray)
    assert dml_obj.psi.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(dml_obj.psi_a, np.ndarray)
    assert dml_obj.psi_a.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(dml_obj.psi_b, np.ndarray)
    assert dml_obj.psi_b.shape == (n_obs, n_rep, n_treat, )

    assert isinstance(dml_obj.pval, np.ndarray)
    assert dml_obj.pval.shape == (n_treat, )

    assert isinstance(dml_obj.se, np.ndarray)
    assert dml_obj.se.shape == (n_treat, )

    assert isinstance(dml_obj.t_stat, np.ndarray)
    assert dml_obj.t_stat.shape == (n_treat, )

    assert isinstance(dml_obj._dml_data.binary_treats, pd.Series)
    assert len(dml_obj._dml_data.binary_treats) == n_treat

    assert isinstance(dml_obj.smpls, list)
    assert len(dml_obj.smpls) == n_rep
    all_tuple = all([all([isinstance(tpl, tuple) for tpl in smpl]) for smpl in dml_obj.smpls])
    assert all_tuple
    all_pairs = all([all([len(tpl) == 2 for tpl in smpl]) for smpl in dml_obj.smpls])
    assert all_pairs
    n_folds_each_smpl = np.array([len(smpl) for smpl in dml_obj.smpls])
    assert np.all(n_folds_each_smpl == n_folds_each_smpl[0])
    assert n_folds_each_smpl[0] == n_folds
