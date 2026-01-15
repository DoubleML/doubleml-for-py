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

df_int = pd.DataFrame(
    {
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "y": [1, 6, 2, 8],
        "d": [1, 2, 3, 4],
        "x1": [1, 1, 0, 1],
        "x2": [1, 2, 0, 2],
    }
)
df_float = df_int.copy()
df_float["time"] = df_float["time"].astype(float)

df_datetime = df_int.copy()
df_datetime["time"] = pd.to_datetime([f"2020-{int(t):02d}-01" for t in df_datetime["time"]])

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


@pytest.fixture(scope="module", params=["cre_general", "cre_normal", "fd_exact", "wg_approx"])
def approach(request):
    return request.param


@pytest.fixture(scope="module", params=["cre_general", "cre_normal"])
def cre_approach(request):
    return request.param


@pytest.fixture(scope="module", params=["int", "float", "datetime"])
def time_type(request):
    return request.param


@pytest.fixture(scope="module", params=[df_int, df_float, df_datetime])
def data_time_type(request):
    return request.param


@pytest.mark.ci
def test_plpr_approach_x_dim(approach, time_type):
    np.random.seed(3141)
    plpr_data = make_plpr_CP2025(dim_x=dim_x, time_type=time_type)
    obj_dml_data = dml.DoubleMLPanelData(
        plpr_data,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_plpr = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach=approach)
    if approach == "wg_approx":
        assert len(dml_plpr._dml_data.x_cols) == dim_x
    else:
        assert len(dml_plpr._dml_data.x_cols) == dim_x * 2


@pytest.mark.ci
def test_plpr_approach_d_mean(approach, time_type):
    np.random.seed(3141)
    plpr_data = make_plpr_CP2025(dim_x=dim_x, time_type=time_type)
    obj_dml_data = dml.DoubleMLPanelData(
        plpr_data,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_plpr = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach=approach)
    if approach in ["cre_general", "cre_normal"]:
        assert dml_plpr.d_mean is not None
    else:
        assert dml_plpr.d_mean is None


@pytest.mark.ci
def test_plpr_d_mean_calc(cre_approach, data_time_type):
    obj_dml_data = dml.DoubleMLPanelData(
        data_time_type,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_obj = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach=cre_approach, n_folds=2)
    assert np.array_equal(dml_obj.d_mean, np.array([1.5, 1.5, 3.5, 3.5]).reshape(-1, 1))


@pytest.mark.ci
def test_plpr_fd_exact_unbalanced(time_type):
    msg_warn = r"The panel data contains 1 missing \(id, time\) combinations. "

    np.random.seed(3141)
    data = make_plpr_CP2025(num_id=3, num_t=3, dim_x=4, time_type=time_type)
    data_unbalanced = data.drop(data.index[[1]])
    data_unbalanced = data_unbalanced.sample(frac=1)

    obj_dml_data = dml.DoubleMLPanelData(
        data_unbalanced,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    with pytest.warns(UserWarning, match=msg_warn):
        obj_plpr = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="fd_exact", n_folds=2)
    # 4 rows after fd transformation as id 3 has no possible first difference
    assert obj_plpr.data_transform.data.shape[0] == 4


@pytest.mark.ci
def test_plpr_one_id(approach, time_type):
    msg_warn = r"The data contains 2 id\(s\) with only one row. These row\(s\) have been dropped."

    np.random.seed(3141)
    data = make_plpr_CP2025(num_id=4, num_t=2, dim_x=4, time_type=time_type)
    data_one_id = data.drop(data.index[[1, 4]]).reset_index(drop=True)
    data_one_id = data_one_id.sample(frac=1)

    obj_dml_data = dml.DoubleMLPanelData(
        data_one_id,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    with pytest.warns(UserWarning, match=msg_warn):
        obj_plpr = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach=approach, n_folds=2)
    # 2 rows after fd transformation, 4 rows else
    if approach == "fd_exact":
        assert obj_plpr.data_transform.data.shape[0] == 2
    else:
        assert obj_plpr.data_transform.data.shape[0] == 4


@pytest.mark.ci
def test_plpr_fd_exact_one_id_unbalanced(time_type):
    msg_warn_one_id = r"The data contains 1 id(s) with only one row. These row(s) have been dropped."
    msg_warn_unbalanced = r"The panel data contains 1 missing (id, time) combinations. "

    np.random.seed(3141)
    data = make_plpr_CP2025(num_id=4, num_t=3, dim_x=4, time_type=time_type)
    data_one_id_unbalanced = data.drop(data.index[[1, 2, 4]])
    data_one_id_unbalanced = data_one_id_unbalanced.sample(frac=1)

    obj_dml_data = dml.DoubleMLPanelData(
        data_one_id_unbalanced,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    # capture warnings
    with pytest.warns(UserWarning) as record:
        obj_plpr = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="fd_exact", n_folds=2)
    # assert two warnings were raised and content
    assert len(record) == 2
    assert msg_warn_one_id in str(record[0].message)
    assert msg_warn_unbalanced in str(record[1].message)
    # 4 rows after fd transformation, id 1 with only 1 row and id 2 has no possible first difference dropped
    assert obj_plpr.data_transform.data.shape[0] == 4


@pytest.mark.ci
def test_plpr_time_cre_transformation(cre_approach, data_time_type):
    obj_dml_data = dml.DoubleMLPanelData(
        data_time_type,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_cre = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach=cre_approach, n_folds=2)
    assert dml_cre.transform_cols["y_col"] == "y"
    assert dml_cre.transform_cols["d_cols"] == ["d"]
    assert dml_cre.transform_cols["x_cols"] == ["x1", "x2", "x1_mean", "x2_mean"]
    assert np.array_equal(dml_cre.data_transform.y, cre_array[:, 0])
    assert np.array_equal(dml_cre.data_transform.d, cre_array[:, 1])
    assert np.array_equal(dml_cre.data_transform.x, cre_array[:, 2:])


@pytest.mark.ci
def test_plpr_time_fd_wg_transformation(data_time_type):
    obj_dml_data = dml.DoubleMLPanelData(
        data_time_type,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )
    dml_fd = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="fd_exact", n_folds=2)
    assert dml_fd.transform_cols["y_col"] == "y_diff"
    assert dml_fd.transform_cols["d_cols"] == ["d_diff"]
    assert dml_fd.transform_cols["x_cols"] == ["x1", "x2", "x1_lag", "x2_lag"]
    assert np.array_equal(dml_fd.data_transform.y, fd_array[:, 0])
    assert np.array_equal(dml_fd.data_transform.d, fd_array[:, 1])
    assert np.array_equal(dml_fd.data_transform.x, fd_array[:, 2:])

    dml_wg = dml.DoubleMLPLPR(obj_dml_data, ml_l, ml_m, approach="wg_approx", n_folds=2)
    assert dml_wg.transform_cols["y_col"] == "y_demean"
    assert dml_wg.transform_cols["d_cols"] == ["d_demean"]
    assert dml_wg.transform_cols["x_cols"] == ["x1_demean", "x2_demean"]
    assert np.array_equal(dml_wg.data_transform.y, wg_array[:, 0])
    assert np.array_equal(dml_wg.data_transform.d, wg_array[:, 1])
    assert np.array_equal(dml_wg.data_transform.x, wg_array[:, 2:])
