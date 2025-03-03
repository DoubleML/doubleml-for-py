import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml.data import DoubleMLData, DoubleMLPanelData
from doubleml.did import DoubleMLDID, DoubleMLDIDBinary, DoubleMLDIDCS
from doubleml.did.datasets import make_did_CS2021, make_did_SZ2020
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
    check_sensitivity_return_types,
)

# Test constants
N_OBS = 200
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}


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
    (DoubleMLDID(datasets["did"], Lasso(), LogisticRegression(), **dml_args), DoubleMLDID),
    (DoubleMLDID(datasets["did_binary_outcome"], LogisticRegression(), LogisticRegression(), **dml_args), DoubleMLDID),
    (DoubleMLDIDCS(datasets["did_cs"], Lasso(), LogisticRegression(), **dml_args), DoubleMLDIDCS),
    (DoubleMLDIDCS(datasets["did_cs_binary_outcome"], LogisticRegression(), LogisticRegression(), **dml_args), DoubleMLDIDCS),
]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_return_types(dml_obj, cls):
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)


@pytest.fixture(params=dml_objs)
def fitted_dml_obj(request):
    dml_obj, _ = request.param
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_sensitivity_return_types(fitted_dml_obj):
    if fitted_dml_obj._sensitivity_implemented:
        benchmarking_set = [fitted_dml_obj._dml_data.x_cols[0]]
        check_sensitivity_return_types(fitted_dml_obj, N_OBS, N_REP, N_TREAT, benchmarking_set=benchmarking_set)


# panel data
df_panel = make_did_CS2021(n_obs=N_OBS, dgp_type=1, n_pre_treat_periods=0, n_periods=3, time_type="float")
df_panel["y_binary"] = np.random.binomial(n=1, p=0.5, size=df_panel.shape[0])
datasets["did_panel"] = DoubleMLPanelData(
    df_panel, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)
datasets["did_panel_binary_outcome"] = DoubleMLPanelData(
    df_panel, y_col="y_binary", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
)

dml_panel_binary_args = dml_args | {
    "g_value": 1,
    "t_value_pre": 0,
    "t_value_eval": 1,
}

dml_objs_panel = [
    (
        DoubleMLDIDBinary(datasets["did_panel"], ml_g=Lasso(), ml_m=LogisticRegression(), **dml_panel_binary_args),
        DoubleMLDIDBinary,
    ),
    (
        DoubleMLDIDBinary(
            datasets["did_panel_binary_outcome"], ml_g=LogisticRegression(), ml_m=LogisticRegression(), **dml_panel_binary_args
        ),
        DoubleMLDIDBinary,
    ),
]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj_panel, cls", dml_objs)
def test_panel_return_types(dml_obj_panel, cls):
    check_basic_return_types(dml_obj_panel, cls)

    # further return type tests
    assert isinstance(dml_obj_panel.get_params("ml_m"), dict)


@pytest.fixture(params=dml_objs_panel)
def fitted_panel_dml_obj(request):
    dml_obj, _ = request.param
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_panel_property_types_and_shapes(fitted_panel_dml_obj):
    check_basic_property_types_and_shapes(fitted_panel_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_panel_dml_obj, N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_panel_sensitivity_return_types(fitted_panel_dml_obj):
    if fitted_panel_dml_obj._sensitivity_implemented:
        benchmarking_set = [fitted_panel_dml_obj._dml_data.x_cols[0]]
        check_sensitivity_return_types(fitted_panel_dml_obj, N_OBS, N_REP, N_TREAT, benchmarking_set=benchmarking_set)
