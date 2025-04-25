import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDAggregation, DoubleMLDIDMulti
from doubleml.did.datasets import make_did_CS2021
from doubleml.double_ml_framework import DoubleMLFramework

# Test constants
N_OBS = 200
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
    "gt_combinations": "standard",
}


# create all datasets
np.random.seed(3141)
datasets = {}

# panel data
df_panel = make_did_CS2021(n_obs=N_OBS, dgp_type=1, n_pre_treat_periods=2, n_periods=5, time_type="float")
df_panel["y_binary"] = np.random.binomial(n=1, p=0.5, size=df_panel.shape[0])
datasets["did_panel"] = DoubleMLPanelData(
    df_panel, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)
datasets["did_panel_binary_outcome"] = DoubleMLPanelData(
    df_panel, y_col="y_binary", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)


dml_objs = [
    (DoubleMLDIDMulti(datasets["did_panel"], ml_g=Lasso(), ml_m=LogisticRegression(), **dml_args), DoubleMLDIDMulti),
    (
        DoubleMLDIDMulti(
            datasets["did_panel_binary_outcome"], ml_g=LogisticRegression(), ml_m=LogisticRegression(), **dml_args
        ),
        DoubleMLDIDMulti,
    ),
]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_panel_return_types(dml_obj, cls):
    assert isinstance(dml_obj.__str__(), str)
    assert isinstance(dml_obj.summary, pd.DataFrame)
    # assert isinstance(dml_obj.draw_sample_splitting(), cls)  # not implemented
    assert isinstance(dml_obj.fit(), cls)
    assert isinstance(dml_obj.__str__(), str)  # called again after fit, now with numbers
    assert isinstance(dml_obj.summary, pd.DataFrame)  # called again after fit, now with numbers
    assert isinstance(dml_obj.bootstrap(), cls)

    assert isinstance(dml_obj.confint(), pd.DataFrame)
    assert isinstance(dml_obj.p_adjust(), pd.DataFrame)

    assert isinstance(dml_obj._dml_data.__str__(), str)

    # further return type tests


@pytest.fixture(params=dml_objs)
def fitted_dml_obj(request):
    dml_obj, _ = request.param
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_panel_property_types_and_shapes(fitted_dml_obj):
    n_treat = len(fitted_dml_obj.gt_combinations)
    dml_obj = fitted_dml_obj

    # check_basic_property_types_and_shapes
    # check that the setting is still in line with the hard-coded values
    assert dml_obj._dml_data.n_treat == 1
    assert dml_obj.n_gt_atts == n_treat
    assert dml_obj.n_rep == N_REP
    assert dml_obj.n_folds == N_FOLDS
    assert dml_obj._dml_data.n_obs == N_OBS
    assert dml_obj.n_rep_boot == N_REP_BOOT

    assert isinstance(dml_obj.all_coef, np.ndarray)
    assert dml_obj.all_coef.shape == (n_treat, N_REP)

    assert isinstance(dml_obj.all_se, np.ndarray)
    assert dml_obj.all_se.shape == (n_treat, N_REP)

    assert isinstance(dml_obj.boot_t_stat, np.ndarray)
    assert dml_obj.boot_t_stat.shape == (N_REP_BOOT, n_treat, N_REP)

    assert isinstance(dml_obj.coef, np.ndarray)
    assert dml_obj.coef.shape == (n_treat,)

    assert isinstance(dml_obj.se, np.ndarray)
    assert dml_obj.se.shape == (n_treat,)

    assert isinstance(dml_obj.t_stat, np.ndarray)
    assert dml_obj.t_stat.shape == (n_treat,)

    assert isinstance(dml_obj.framework.scaled_psi, np.ndarray)
    assert dml_obj.framework.scaled_psi.shape == (
        N_OBS,
        n_treat,
        N_REP,
    )

    assert isinstance(dml_obj.framework, DoubleMLFramework)
    assert isinstance(dml_obj.pval, np.ndarray)
    assert dml_obj.pval.shape == (n_treat,)

    assert isinstance(dml_obj._dml_data.binary_treats, pd.Series)
    assert len(dml_obj._dml_data.binary_treats) == 1

    # check_basic_predictions_and_targets
    expected_keys = ["ml_g0", "ml_g1", "ml_m"]
    for key in expected_keys:
        assert isinstance(dml_obj.nuisance_loss[key], np.ndarray)
        assert dml_obj.nuisance_loss[key].shape == (N_REP, n_treat)


@pytest.mark.ci
def test_panel_sensitivity_return_types(fitted_dml_obj):
    n_treat = len(fitted_dml_obj.gt_combinations)
    benchmarking_set = [fitted_dml_obj._dml_data.x_cols[0]]
    dml_obj = fitted_dml_obj

    assert isinstance(dml_obj.sensitivity_elements, dict)
    for key in ["sigma2", "nu2", "max_bias"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (1, n_treat, N_REP)
    for key in ["psi_max_bias"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (N_OBS, n_treat, N_REP)

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


@pytest.mark.ci
def test_panel_plot_effects(fitted_dml_obj):
    fig, axes = fitted_dml_obj.plot_effects()
    assert isinstance(fig, Figure)

    # list of axes objects
    assert isinstance(axes, list)
    for ax in axes:
        assert isinstance(ax, Axes)

    plt.close(fig)


@pytest.fixture(scope="module", params=["eventstudy", "group", "time"])
def aggregation(request):
    return request.param


@pytest.mark.ci
def test_panel_agg_return_types(fitted_dml_obj, aggregation):
    agg_obj = fitted_dml_obj.aggregate(aggregation=aggregation)
    agg_obj.aggregated_frameworks.bootstrap(n_rep_boot=10)

    assert isinstance(agg_obj, DoubleMLDIDAggregation)
    assert isinstance(agg_obj.__str__(), str)

    # test plotting
    fig, ax = agg_obj.plot_effects()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)
