import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLLPLR, DoubleMLPanelData, DoubleMLPLPR
from doubleml.plm.datasets import make_lplr_LZZ2020, make_plpr_CP2025
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
# PLPR specific
NUM_ID = 100
NUM_T = 10
DIM_X = 30

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}


# create all datasets
np.random.seed(3141)
datasets = {}


datasets["lplr"] = make_lplr_LZZ2020(n_obs=N_OBS)
datasets["lplr_binary"] = make_lplr_LZZ2020(n_obs=N_OBS, treatment="binary")
plpr_data = make_plpr_CP2025(num_id=NUM_ID, num_t=NUM_T, dim_x=DIM_X)
datasets["plpr"] = DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)

dml_lplr_obj = DoubleMLLPLR(datasets["lplr"], LogisticRegression(), LinearRegression(), LinearRegression(), **dml_args)
dml_lplr_obj_binary = DoubleMLLPLR(
    datasets["lplr_binary"], LogisticRegression(), LinearRegression(), LogisticRegression(), **dml_args
)
dml_plpr_obj_fd_exact = DoubleMLPLPR(datasets["plpr"], LinearRegression(), LinearRegression(), approach="fd_exact", **dml_args)
dml_plpr_obj_wg_appox = DoubleMLPLPR(
    datasets["plpr"], LinearRegression(), LinearRegression(), approach="wg_approx", **dml_args
)

dml_objs = [
    (dml_lplr_obj, DoubleMLLPLR),
    (dml_lplr_obj_binary, DoubleMLLPLR),
    (dml_plpr_obj_fd_exact, DoubleMLPLPR),
    (dml_plpr_obj_wg_appox, DoubleMLPLPR),
]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_return_types(dml_obj, cls):
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)

    # test plpr cluster
    if cls == DoubleMLPLPR:
        assert dml_obj._is_cluster_data
        assert dml_obj._dml_data.is_cluster_data
        assert dml_obj._dml_data.cluster_cols[0] == dml_obj._dml_data.id_col


@pytest.fixture(params=dml_objs)
def fitted_dml_obj(request):
    dml_obj, _ = request.param
    dml_obj.fit()
    if not dml_obj._is_cluster_data:
        dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    if not fitted_dml_obj.__class__ == DoubleMLPLPR:
        check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
        check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)
    else:
        if fitted_dml_obj.approach == "fd_exact":
            check_basic_property_types_and_shapes(fitted_dml_obj, NUM_ID * (NUM_T - 1), N_TREAT, N_REP, N_FOLDS, None)
            check_basic_predictions_and_targets(fitted_dml_obj, NUM_ID * (NUM_T - 1), N_TREAT, N_REP)
        elif fitted_dml_obj.approach == "wg_approx":
            check_basic_property_types_and_shapes(fitted_dml_obj, NUM_ID * NUM_T, N_TREAT, N_REP, N_FOLDS, None)
            check_basic_predictions_and_targets(fitted_dml_obj, NUM_ID * NUM_T, N_TREAT, N_REP)


@pytest.mark.ci
def test_sensitivity_return_types(fitted_dml_obj):
    if fitted_dml_obj._sensitivity_implemented:
        benchmarking_set = [fitted_dml_obj._dml_data.x_cols[0]]
        check_sensitivity_return_types(fitted_dml_obj, N_OBS, N_REP, N_TREAT, benchmarking_set=benchmarking_set)
