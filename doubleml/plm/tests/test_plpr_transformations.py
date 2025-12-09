import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import doubleml as dml

from ..datasets import make_plpr_CP2025

dim_x = 30

learner = LinearRegression()
ml_l = clone(learner)
ml_m = clone(learner)

np.random.seed(3141)
plpr_data = make_plpr_CP2025(dim_x=dim_x)
obj_dml_data = dml.DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)
dml_plpr_cre_general = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="cre_general")
dml_plpr_cre_normal = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="cre_normal")
dml_plpr_fd_exact = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="fd_exact")
dml_plpr_wg_approx = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="wg_approx")

df = pd.DataFrame(
    {
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "y": [1, 6, 2, 8],
        "d": [1, 2, 3, 4],
        "x1": [1, 1, 0, 1],
        "x2": [1, 2, 0, 2],
    }
)
df_dml_data = dml.DoubleMLPanelData(
    df,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)
cre_array = np.array(
    [
        [1, 1, 1, 1, 1, 1.5],
        [6, 2, 1, 2, 1, 1.5],
        [2, 3, 0, 0, 0.5, 1],
        [8, 4, 1, 2, 0.5, 1],
    ]
)
fd_array = np.array(
    [
        [5, 1, 1, 2, 1, 1],
        [6, 1, 1, 2, 0, 0],
    ]
)
wg_array = np.array(
    [
        [1.75, 2, 0.75, 0.75],
        [6.75, 3, 0.75, 1.75],
        [1.25, 2, 0.25, 0.25],
        [7.25, 3, 1.25, 2.25],
    ]
)
df_unbalanced = pd.DataFrame(
    {
        "id": [1, 1, 2, 2, 3],
        "time": [1, 2, 1, 2, 2],
        "y": [1, 6, 2, 8, 1],
        "d": [1, 2, 3, 4, 1],
        "x1": [1, 1, 0, 1, 1],
        "x2": [1, 2, 0, 2, 1],
    }
)
df_unbalanced_dml_data = dml.DoubleMLPanelData(
    df_unbalanced,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)


@pytest.fixture(scope="module", params=["cre_general", "cre_normal"])
def cre_approach(request):
    return request.param


@pytest.mark.ci
def test_plpr_approach_x_dim():
    assert len(dml_plpr_cre_general._dml_data.x_cols) == dim_x * 2
    assert len(dml_plpr_cre_normal._dml_data.x_cols) == dim_x * 2
    assert len(dml_plpr_fd_exact._dml_data.x_cols) == dim_x * 2
    assert len(dml_plpr_wg_approx._dml_data.x_cols) == dim_x


@pytest.mark.ci
def test_plpr_approach_d_mean():
    assert dml_plpr_cre_general.d_mean is not None
    assert dml_plpr_cre_normal.d_mean is not None
    assert dml_plpr_fd_exact.d_mean is None
    assert dml_plpr_wg_approx.d_mean is None


@pytest.mark.ci
def test_plpr_d_mean_calc(cre_approach):
    dml_obj = dml.DoubleMLPLPR(df_dml_data, ml_l, ml_m, approach=cre_approach, n_folds=2)
    assert np.array_equal(dml_obj.d_mean, np.array([1.5, 1.5, 3.5, 3.5]).reshape(-1, 1))


@pytest.mark.ci
def test_plpr_cre_transformation(cre_approach):
    dml_obj = dml.DoubleMLPLPR(df_dml_data, ml_l, ml_m, approach=cre_approach, n_folds=2)
    assert dml_obj.transform_cols["y_col"] == "y"
    assert dml_obj.transform_cols["d_cols"] == ["d"]
    assert dml_obj.transform_cols["x_cols"] == ["x1", "x2", "x1_mean", "x2_mean"]
    assert np.array_equal(dml_obj.data_transform.y, cre_array[:, 0])
    assert np.array_equal(dml_obj.data_transform.d, cre_array[:, 1])
    assert np.array_equal(dml_obj.data_transform.x, cre_array[:, 2:])


@pytest.mark.ci
def test_plpr_fd_exact_transformation():
    dml_obj = dml.DoubleMLPLPR(df_dml_data, ml_l, ml_m, approach="fd_exact", n_folds=2)
    assert dml_obj.transform_cols["y_col"] == "y_diff"
    assert dml_obj.transform_cols["d_cols"] == ["d_diff"]
    assert dml_obj.transform_cols["x_cols"] == ["x1", "x2", "x1_lag", "x2_lag"]
    assert np.array_equal(dml_obj.data_transform.y, fd_array[:, 0])
    assert np.array_equal(dml_obj.data_transform.d, fd_array[:, 1])
    assert np.array_equal(dml_obj.data_transform.x, fd_array[:, 2:])


@pytest.mark.ci
def test_plpr_wg_approx_transformation():
    dml_obj = dml.DoubleMLPLPR(df_dml_data, ml_l, ml_m, approach="wg_approx", n_folds=2)
    assert dml_obj.transform_cols["y_col"] == "y_demean"
    assert dml_obj.transform_cols["d_cols"] == ["d_demean"]
    assert dml_obj.transform_cols["x_cols"] == ["x1_demean", "x2_demean"]
    assert np.array_equal(dml_obj.data_transform.y, wg_array[:, 0])
    assert np.array_equal(dml_obj.data_transform.d, wg_array[:, 1])
    assert np.array_equal(dml_obj.data_transform.x, wg_array[:, 2:])


@pytest.mark.ci
def test_plpr_fd_exact_unbalanced():
    msg_warn = r"The panel data contains 1 missing \(id, time\) combinations. "
    with pytest.warns(UserWarning, match=msg_warn):
        _ = dml.DoubleMLPLPR(df_unbalanced_dml_data, ml_l, ml_m, approach="fd_exact", n_folds=2)
