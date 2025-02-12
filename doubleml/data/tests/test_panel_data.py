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
        data=df,
        y_col="y",
        d_cols="d",
        t_col="t",
        id_col="id",
        x_cols=[f"Z{i + 1}" for i in np.arange(4)]
    )

    # check that after changing t_col, the t array gets updated
    t_comp = dml_data.data["t_new"].values
    dml_data.t_col = "t_new"
    assert np.array_equal(dml_data.t, t_comp)
    assert dml_data._t_values == np.unique(t_comp)
    assert dml_data.n_t_periods == 1

    msg = r"Invalid time variable t_col. a13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = "a13"

    msg = r"The time variable t_col must be of str type \(or None\). " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.t_col = 5

    msg = "Invalid time variable t_col. Time variable required for panel data."
    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = None
