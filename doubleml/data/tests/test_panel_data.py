import numpy as np
import pandas as pd
import pytest

from doubleml.data import DoubleMLPanelData
from doubleml.did.datasets import make_did_SZ2020


@pytest.mark.ci
def test_dml_datatype():
    data_array = np.zeros((100, 10))
    with pytest.raises(TypeError):
        _ = DoubleMLPanelData(data_array, y_col="y", d_cols=["d"], t_col="t", id_col="id")


@pytest.mark.ci
def test_t_col_setter():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")._data
    df["t_new"] = 1.0
    dml_data = DoubleMLPanelData(
        data=df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    # check that after changing t_col, the t array gets updated
    t_comp = dml_data.data["t_new"].values
    dml_data.t_col = "t_new"
    assert np.array_equal(dml_data.t, t_comp)
    assert dml_data._t_values == np.unique(t_comp)
    assert dml_data.n_t_periods == 1

    msg = "Invalid time variable t_col. a13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = "a13"

    msg = r"The time variable t_col must be of str type \(or None\). " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.t_col = 5

    msg = "Invalid time variable t_col. Time variable required for panel data."
    with pytest.raises(TypeError, match=msg):
        dml_data.t_col = None


@pytest.mark.ci
def test_id_col_setter():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")._data
    df["id_new"] = 1.0
    dml_data = DoubleMLPanelData(
        data=df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    # check that after changing id_col, the id array etc. gets updated
    id_comp = dml_data.data["id_new"].values
    dml_data.id_col = "id_new"
    assert np.array_equal(dml_data.id_var, id_comp)
    assert dml_data._id_var_unique == np.unique(id_comp)
    assert dml_data.n_obs == 1

    msg = "Invalid id variable id_col. a13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.id_col = "a13"

    msg = "The id variable id_col must be of str type. " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.id_col = 5

    msg = "The id variable id_col must be of str type. None of type <class 'NoneType'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.id_col = None


@pytest.mark.ci
def test_d_col_setter():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")._data
    df["d_new"] = 1.0
    dml_data = DoubleMLPanelData(
        data=df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    # check that after changing d_col, the id array etc. gets updated
    d_comp = dml_data.data["d_new"].values
    dml_data.d_cols = "d_new"
    assert dml_data.d_cols == ["d_new"]
    assert np.array_equal(dml_data.d, d_comp)
    assert dml_data.g_col == "d_new"
    assert dml_data._g_values == np.unique(d_comp)
    assert dml_data.n_groups == 1

    msg = r"Invalid treatment variable\(s\) d_cols. At least one treatment variable is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = "a13"

    msg = r"The treatment variable\(s\) d_cols must be of str or list type. 5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.d_cols = 5

    msg = r"The treatment variable\(s\) d_cols must be of str or list type. None of type <class 'NoneType'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.d_cols = None


@pytest.mark.ci
def test_disjoint_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(7), (4, 1)), columns=["yy", "dd1", "xx1", "xx2", "zz", "tt", "id"])

    msg = r"At least one variable/column is set as outcome variable \(``y_col``\) " r"and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPanelData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", id_col="yy")

    msg = (
        r"At least one variable/column is set as treatment variable \(``d_cols``\) " r"and identifier variable \(``id_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPanelData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", id_col="dd1")

    msg = r"At least one variable/column is set as covariate \(``x_cols``\) " r"and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPanelData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", id_col="xx1")

    msg = r"At least one variable/column is set as time variable \(``t_col``\) " r"and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPanelData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", id_col="tt")

    msg = (
        r"At least one variable/column is set as instrumental variable \(``z_cols``\) "
        r"and identifier variable \(``id_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPanelData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", z_cols=["zz"], id_col="zz")


@pytest.mark.ci
def test_panel_data_str():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")._data
    dml_data = DoubleMLPanelData(
        data=df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    # Convert the object to string
    dml_str = str(dml_data)

    # Check that all important sections are present in the string
    assert "================== DoubleMLPanelData Object ==================" in dml_str
    assert "------------------ Data summary      ------------------" in dml_str
    assert "------------------ DataFrame info    ------------------" in dml_str

    # Check that specific data attributes are correctly included
    assert "Outcome variable: y" in dml_str
    assert "Treatment variable(s): ['d']" in dml_str
    assert "Covariates: ['Z1', 'Z2', 'Z3', 'Z4']" in dml_str
    assert "Instrument variable(s): None" in dml_str
    assert "Time variable: t" in dml_str
    assert "Id variable: id" in dml_str
    assert "No. Observations:" in dml_str


@pytest.mark.ci
def test_panel_data_properties():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, return_type="DoubleMLPanelData")._data
    dml_data = DoubleMLPanelData(
        data=df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    assert np.array_equal(dml_data.id_var, df["id"].values)
    assert np.array_equal(dml_data.id_var_unique, np.unique(df["id"].values))
    assert dml_data.n_obs == len(np.unique(df["id"].values))
    assert dml_data.g_col == "d"
    assert np.array_equal(dml_data.g_values, np.sort(np.unique(df["d"].values)))
    assert dml_data.n_groups == len(np.unique(df["d"].values))
    assert np.array_equal(dml_data.t_values, np.sort(np.unique(df["t"].values)))
    assert dml_data.n_t_periods == len(np.unique(df["t"].values))
