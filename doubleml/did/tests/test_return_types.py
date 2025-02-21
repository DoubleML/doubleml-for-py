import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLFramework
from doubleml.double_ml_score_mixins import NonLinearScoreMixin
from doubleml.data import DoubleMLClusterData, DoubleMLData
from doubleml.did import DoubleMLDID, DoubleMLDIDCS
from doubleml.did.datasets import make_did_SZ2020

# Test constants
N_OBS = 200
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314


# create all datasets
np.random.seed(3141)
datasets = {}

datasets["did"] = make_did_SZ2020(n_obs=N_OBS)
datasets["did_cs"] = make_did_SZ2020(n_obs=N_OBS, cross_sectional_data=True)

# Binary outcome
(x, y, d, t) = make_did_SZ2020(n_obs=N_OBS, cross_sectional_data=True, return_type="array")
binary_outcome = np.random.binomial(n=1, p=0.5, size=N_OBS)
datasets["did_binary_outcome"] = DoubleMLData.from_arrays(x, binary_outcome, d)
datasets["did_cs_binary_outcome"] = DoubleMLData.from_arrays(x, binary_outcome, d, t=t)

dml_objs = [
    (DoubleMLDID(datasets["did"], Lasso(), LogisticRegression()), DoubleMLDID),
    (DoubleMLDID(datasets["did_binary_outcome"], LogisticRegression(), LogisticRegression()), DoubleMLDID),
    (DoubleMLDIDCS(datasets["did_cs"], Lasso(), LogisticRegression()), DoubleMLDIDCS),
    (DoubleMLDIDCS(datasets["did_cs_binary_outcome"], LogisticRegression(), LogisticRegression()), DoubleMLDIDCS),
]


def test_basic_return_types(dml_obj, cls):
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
        isinstance(dml_obj.p_adjust("bonferroni"), pd.DataFrame)
    assert isinstance(dml_obj._dml_data.__str__(), str)


def test_basic_property_types_and_shapes(dml_obj, n_obs, n_treat, n_rep, n_folds, n_rep_boot):
    # not checked: learner, learner_names, params, params_names, score
    # already checked: summary

    # check that the setting is still in line with the hard-coded values
    assert dml_obj._dml_data.n_treat == n_treat
    assert dml_obj.n_rep == n_rep
    assert dml_obj.n_folds == n_folds
    assert dml_obj._dml_data.n_obs == n_obs
    assert dml_obj.n_rep_boot == n_rep_boot

    assert isinstance(dml_obj.all_coef, np.ndarray)
    assert dml_obj.all_coef.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.all_se, np.ndarray)
    assert dml_obj.all_se.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.boot_t_stat, np.ndarray)
    assert dml_obj.boot_t_stat.shape == (n_rep_boot, n_treat, n_rep)

    assert isinstance(dml_obj.coef, np.ndarray)
    assert dml_obj.coef.shape == (n_treat,)

    assert isinstance(dml_obj.psi, np.ndarray)
    assert dml_obj.psi.shape == (
        n_obs,
        n_rep,
        n_treat,
    )

    is_nonlinear = isinstance(dml_obj, NonLinearScoreMixin)
    if is_nonlinear:
        for score_element in dml_obj._score_element_names:
            assert isinstance(dml_obj.psi_elements[score_element], np.ndarray)
            assert dml_obj.psi_elements[score_element].shape == (
                n_obs,
                n_rep,
                n_treat,
            )
    else:
        assert isinstance(dml_obj.psi_elements["psi_a"], np.ndarray)
        assert dml_obj.psi_elements["psi_a"].shape == (
            n_obs,
            n_rep,
            n_treat,
        )

        assert isinstance(dml_obj.psi_elements["psi_b"], np.ndarray)
        assert dml_obj.psi_elements["psi_b"].shape == (
            n_obs,
            n_rep,
            n_treat,
        )

    assert isinstance(dml_obj.framework, DoubleMLFramework)
    assert isinstance(dml_obj.pval, np.ndarray)
    assert dml_obj.pval.shape == (n_treat,)

    assert isinstance(dml_obj.se, np.ndarray)
    assert dml_obj.se.shape == (n_treat,)

    assert isinstance(dml_obj.t_stat, np.ndarray)
    assert dml_obj.t_stat.shape == (n_treat,)

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

    return


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_return_types(dml_obj, cls):
    test_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)


# Fit all objs
for dml_obj, _ in dml_objs:
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_property_types_and_shapes(dml_obj, cls):
    cls = cls
    test_basic_property_types_and_shapes(dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)