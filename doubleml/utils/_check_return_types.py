import numpy as np
import pandas as pd
import plotly

from doubleml import DoubleMLFramework
from doubleml.double_ml_score_mixins import NonLinearScoreMixin


def check_basic_return_types(dml_obj, cls):
    # ToDo: A second test case with multiple treatment variables would be helpful
    assert isinstance(dml_obj.__str__(), str)
    assert isinstance(dml_obj.summary, pd.DataFrame)
    assert isinstance(dml_obj.draw_sample_splitting(), cls)
    if not dml_obj._is_cluster_data and not hasattr(dml_obj, "n_folds_inner"):  # set_sample_splitting is not available
        assert isinstance(dml_obj.set_sample_splitting(dml_obj.smpls), cls)
    elif dml_obj._is_cluster_data:
        assert dml_obj._dml_data.is_cluster_data
    assert isinstance(dml_obj.fit(), cls)
    assert isinstance(dml_obj.__str__(), str)  # called again after fit, now with numbers
    assert isinstance(dml_obj.summary, pd.DataFrame)  # called again after fit, now with numbers
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.bootstrap(), cls)
    else:
        assert dml_obj._dml_data.is_cluster_data
    assert isinstance(dml_obj.confint(), pd.DataFrame)
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.p_adjust(), pd.DataFrame)
    else:
        isinstance(dml_obj.p_adjust("bonferroni"), pd.DataFrame)
    assert isinstance(dml_obj._dml_data.__str__(), str)


def check_basic_property_types_and_shapes(dml_obj, n_obs, n_treat, n_rep, n_folds, n_rep_boot, score_dim=None):
    # not checked: learner, learner_names, params, params_names, score
    # already checked: summary

    # use default combination
    if score_dim is None:
        score_dim = (n_obs, n_rep, n_treat)

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
    assert dml_obj.psi.shape == score_dim

    assert isinstance(dml_obj.psi_deriv, np.ndarray)
    assert dml_obj.psi_deriv.shape == score_dim
    is_nonlinear = isinstance(dml_obj, NonLinearScoreMixin)
    if is_nonlinear:
        for score_element in dml_obj._score_element_names:
            assert isinstance(dml_obj.psi_elements[score_element], np.ndarray)
            assert dml_obj.psi_elements[score_element].shape == score_dim
    else:
        assert isinstance(dml_obj.psi_elements["psi_a"], np.ndarray)
        assert dml_obj.psi_elements["psi_a"].shape == score_dim

        assert isinstance(dml_obj.psi_elements["psi_b"], np.ndarray)
        assert dml_obj.psi_elements["psi_b"].shape == score_dim

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


def check_basic_predictions_and_targets(dml_obj, n_obs, n_treat, n_rep):

    expected_keys = dml_obj.params_names
    for key in expected_keys:
        assert isinstance(dml_obj.predictions[key], np.ndarray)
        assert dml_obj.predictions[key].shape == (n_obs, n_rep, n_treat)

        assert isinstance(dml_obj.nuisance_targets[key], np.ndarray)
        assert dml_obj.nuisance_targets[key].shape == (n_obs, n_rep, n_treat)

        assert isinstance(dml_obj.nuisance_loss[key], np.ndarray)
        assert dml_obj.nuisance_loss[key].shape == (n_rep, n_treat)

    learner_eval = dml_obj.evaluate_learners()
    assert isinstance(learner_eval, dict)
    for key in expected_keys:
        assert key in learner_eval
        assert isinstance(learner_eval[key], np.ndarray)
        assert learner_eval[key].shape == (n_rep, n_treat)
    return


def check_sensitivity_return_types(dml_obj, n_obs, n_rep, n_treat, benchmarking_set):
    assert isinstance(dml_obj.sensitivity_elements, dict)
    for key in ["sigma2", "nu2"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (1, n_rep, n_treat)
    for key in ["psi_sigma2", "psi_nu2", "riesz_rep"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (n_obs, n_rep, n_treat)

    assert isinstance(dml_obj.sensitivity_summary, str)
    dml_obj.sensitivity_analysis()
    assert isinstance(dml_obj.sensitivity_summary, str)
    assert isinstance(dml_obj.sensitivity_plot(), plotly.graph_objs._figure.Figure)
    benchmarks = {"cf_y": [0.1, 0.2], "cf_d": [0.15, 0.2], "name": ["test1", "test2"]}
    assert isinstance(dml_obj.sensitivity_plot(value="ci", benchmarks=benchmarks), plotly.graph_objs._figure.Figure)

    assert isinstance(dml_obj.framework._calc_sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95), dict)
    assert isinstance(
        dml_obj.framework._calc_robustness_value(null_hypothesis=0.0, level=0.95, rho=1.0, idx_treatment=0), tuple
    )
    benchmark = dml_obj.sensitivity_benchmark(benchmarking_set=benchmarking_set)
    assert isinstance(benchmark, pd.DataFrame)

    return
