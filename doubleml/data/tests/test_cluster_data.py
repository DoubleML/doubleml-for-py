import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLData, DoubleMLDIDData, DoubleMLSSMData
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021, make_plr_CCDDHNR2018


@pytest.mark.ci
def test_obj_vs_from_arrays():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)
    dml_data_from_array = DoubleMLData.from_arrays(
        x=dml_data.data[dml_data.x_cols],
        y=dml_data.data[dml_data.y_col],
        d=dml_data.data[dml_data.d_cols],
        cluster_vars=dml_data.data[dml_data.cluster_cols],
        z=dml_data.data[dml_data.z_cols],
    )
    df = dml_data.data.copy()
    df.rename(
        columns={"cluster_var_i": "cluster_var1", "cluster_var_j": "cluster_var2", "Y": "y", "D": "d", "Z": "z"}, inplace=True
    )
    assert dml_data_from_array.data[list(df.columns)].equals(df)

    # with a single cluster variable
    dml_data_from_array = DoubleMLData.from_arrays(
        x=dml_data.data[dml_data.x_cols],
        y=dml_data.data[dml_data.y_col],
        d=dml_data.data[dml_data.d_cols],
        cluster_vars=dml_data.data[dml_data.cluster_cols[1]],
        z=dml_data.data[dml_data.z_cols],
    )
    df = dml_data.data.copy().drop(columns="cluster_var_i")
    df.rename(columns={"cluster_var_j": "cluster_var", "Y": "y", "D": "d", "Z": "z"}, inplace=True)
    assert dml_data_from_array.data[list(df.columns)].equals(df)


@pytest.mark.ci
def test_x_cols_setter_defaults_w_cluster():
    df = pd.DataFrame(np.tile(np.arange(6), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "xx3", "cluster1"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1")
    assert dml_data.x_cols == ["xx1", "xx2", "xx3"]
    dml_data.x_cols = ["xx1", "xx3"]
    assert dml_data.x_cols == ["xx1", "xx3"]
    dml_data.x_cols = None
    assert dml_data.x_cols == ["xx1", "xx2", "xx3"]

    # with instrument
    df = pd.DataFrame(np.tile(np.arange(6), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "z", "cluster1"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1", z_cols="z")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # without instrument and with time
    df = pd.DataFrame(np.tile(np.arange(6), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "tt", "cluster1"])
    dml_data = DoubleMLDIDData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1", t_col="tt")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument and with time
    df = pd.DataFrame(np.tile(np.arange(7), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "zz", "tt", "cluster1"])
    dml_data = DoubleMLDIDData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1", z_cols="zz", t_col="tt")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # without instrument and with selection
    df = pd.DataFrame(np.tile(np.arange(6), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "ss", "cluster1"])
    dml_data = DoubleMLSSMData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument and with selection
    df = pd.DataFrame(np.tile(np.arange(7), (6, 1)), columns=["yy", "dd", "xx1", "xx2", "zz", "ss", "cluster1"])
    dml_data = DoubleMLSSMData(df, y_col="yy", d_cols="dd", cluster_cols="cluster1", z_cols="zz", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]


@pytest.mark.ci
def test_cluster_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(7)] + ["y", "d1", "d2"]
    dml_data = DoubleMLData(
        df, "y", ["d1", "d2"], cluster_cols=[f"X{i + 1}" for i in [5, 6]], x_cols=[f"X{i + 1}" for i in np.arange(5)]
    )

    cluster_vars = df[["X6", "X7"]].values
    assert np.array_equal(dml_data.cluster_vars, cluster_vars)
    assert dml_data.n_cluster_vars == 2

    # check that after changing cluster_cols, the cluster_vars array gets updated
    cluster_vars = df[["X7", "X6"]].values
    dml_data.cluster_cols = ["X7", "X6"]
    assert np.array_equal(dml_data.cluster_vars, cluster_vars)

    msg = r"Invalid cluster variable\(s\) cluster_cols. At least one cluster variable is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.cluster_cols = ["X6", "X13"]
    with pytest.raises(ValueError, match=msg):
        dml_data.cluster_cols = "X13"

    msg = (
        r"The cluster variable\(s\) cluster_cols must be of str or list type \(or None\)\. "
        "5 of type <class 'int'> was passed."
    )
    with pytest.raises(TypeError, match=msg):
        dml_data.cluster_cols = 5

    # check single cluster variable
    cluster_vars = df[["X7"]].values
    dml_data.cluster_cols = "X7"
    assert np.array_equal(dml_data.cluster_vars, cluster_vars)
    assert dml_data.n_cluster_vars == 1


@pytest.mark.ci
def test_disjoint_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(6), (4, 1)), columns=["yy", "dd1", "xx1", "xx2", "zz", "tt"])

    # cluster data
    msg = (
        r"At least one variable/column is set as outcome variable \(``y_col``\) "
        r"and cluster variable\(s\) \(``cluster_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], cluster_cols="yy")
    msg = (
        r"At least one variable/column is set as treatment variable \(``d_cols``\) "
        r"and cluster variable\(s\) \(``cluster_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], cluster_cols="dd1")
    msg = (
        r"At least one variable/column is set as covariate \(``x_cols``\) " r"and cluster variable\(s\) \(``cluster_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], cluster_cols="xx2")

    msg = (
        r"At least one variable/column is set as instrumental variable \(``z_cols``\) "
        r"and cluster variable\(s\) \(``cluster_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1"], z_cols=["xx2"], cluster_cols="xx2")

    msg = (
        r"At least one variable/column is set as cluster variable\(s\) \(``cluster_cols``\) "
        r"and time variable \(``t_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDIDData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1"], t_col="xx2", cluster_cols="xx2")

    msg = (
        r"At least one variable/column is set as cluster variable\(s\) \(``cluster_cols``\) "
        r"and selection variable \(``s_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSMData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1"], s_col="xx2", cluster_cols="xx2")


@pytest.mark.ci
def test_duplicates():
    np.random.seed(3141)
    dml_cluster_data = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)

    msg = r"Invalid cluster variable\(s\) cluster_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_cluster_data.data, y_col="Y", d_cols=["D"], cluster_cols=["X3", "X2", "X3"])
    with pytest.raises(ValueError, match=msg):
        dml_cluster_data.cluster_cols = ["X3", "X2", "X3"]

    msg = "Invalid pd.DataFrame: Contains duplicate column names."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(
            pd.DataFrame(np.zeros((100, 5)), columns=["y", "d", "X3", "X2", "y"]), y_col="y", d_cols=["d"], cluster_cols=["X2"]
        )


@pytest.mark.ci
def test_dml_datatype():
    data_array = np.zeros((100, 10))
    with pytest.raises(TypeError):
        _ = DoubleMLData(data_array, y_col="y", d_cols=["d"], cluster_cols=["X3", "X2"])


@pytest.mark.ci
def test_cluster_data_str():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)

    # Convert the object to string
    dml_str = str(dml_data)

    # Check that all important sections are present in the string
    assert "================== DoubleMLData Object ==================" in dml_str
    assert "------------------ Data summary      ------------------" in dml_str
    assert "------------------ DataFrame info    ------------------" in dml_str

    # Check that specific data attributes are correctly included
    assert "Outcome variable: Y" in dml_str
    assert "Treatment variable(s): ['D']" in dml_str
    assert "Cluster variable(s): ['cluster_var_i', 'cluster_var_j']" in dml_str
    assert "Covariates: " in dml_str
    assert "Instrument variable(s): ['Z']" in dml_str
    assert "No. Observations:" in dml_str

    # Test with additional optional attributes
    df = dml_data.data.copy()
    df["time_var"] = 1
    df["score_var"] = 0.5

    dml_data_with_optional = DoubleMLDIDData(
        data=df,
        y_col="Y",
        d_cols="D",
        cluster_cols=["cluster_var_i", "cluster_var_j"],
        z_cols="Z",
        t_col="time_var",
    )

    dml_str_optional = str(dml_data_with_optional)
    assert "Time variable: time_var" in dml_str_optional
