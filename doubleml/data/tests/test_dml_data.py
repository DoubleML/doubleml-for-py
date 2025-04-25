import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLData, DoubleMLDIDCS, DoubleMLPLR, DoubleMLSSM
from doubleml.data.base_data import DoubleMLBaseData
from doubleml.datasets import (
    _make_pliv_data,
    make_pliv_CHS2015,
    make_plr_CCDDHNR2018,
    make_ssm_data,
)
from doubleml.did.datasets import make_did_SZ2020


class DummyDataClass(DoubleMLBaseData):
    def __init__(self, data):
        DoubleMLBaseData.__init__(self, data)

    @property
    def n_coefs(self):
        return 1


@pytest.mark.ci
def test_doubleml_basedata():
    dummy_dml_data = DummyDataClass(pd.DataFrame(np.zeros((100, 10))))
    assert dummy_dml_data.d_cols[0] == "theta"
    assert dummy_dml_data.n_treat == 1
    assert dummy_dml_data.n_coefs == 1


@pytest.fixture(scope="module")
def dml_data_fixture(generate_data1):
    data = generate_data1
    np.random.seed(3141)
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()

    obj_from_np = DoubleMLData.from_arrays(data.loc[:, x_cols].values, data["y"].values, data["d"].values)

    obj_from_pd = DoubleMLData(data, "y", ["d"], x_cols)

    return {"obj_from_np": obj_from_np, "obj_from_pd": obj_from_pd}


@pytest.mark.ci
def test_dml_data_x(dml_data_fixture):
    assert np.allclose(dml_data_fixture["obj_from_np"].x, dml_data_fixture["obj_from_pd"].x, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_y(dml_data_fixture):
    assert np.allclose(dml_data_fixture["obj_from_np"].y, dml_data_fixture["obj_from_pd"].y, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_d(dml_data_fixture):
    assert np.allclose(dml_data_fixture["obj_from_np"].d, dml_data_fixture["obj_from_pd"].d, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_obj_vs_from_arrays():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    dml_data_from_array = DoubleMLData.from_arrays(
        dml_data.data[dml_data.x_cols], dml_data.data[dml_data.y_col], dml_data.data[dml_data.d_cols]
    )
    assert dml_data_from_array.data.equals(dml_data.data)

    dml_data = _make_pliv_data(n_obs=100)
    dml_data_from_array = DoubleMLData.from_arrays(
        dml_data.data[dml_data.x_cols],
        dml_data.data[dml_data.y_col],
        dml_data.data[dml_data.d_cols],
        dml_data.data[dml_data.z_cols],
    )
    assert dml_data_from_array.data.equals(dml_data.data)

    dml_data = make_pliv_CHS2015(n_obs=100, dim_z=5)
    dml_data_from_array = DoubleMLData.from_arrays(
        dml_data.data[dml_data.x_cols],
        dml_data.data[dml_data.y_col],
        dml_data.data[dml_data.d_cols],
        dml_data.data[dml_data.z_cols],
    )
    assert np.array_equal(dml_data_from_array.data, dml_data.data)  # z_cols name differ

    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(7)] + ["y", "d1", "d2"]
    dml_data = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(7)])
    dml_data_from_array = DoubleMLData.from_arrays(
        dml_data.data[dml_data.x_cols], dml_data.data[dml_data.y_col], dml_data.data[dml_data.d_cols]
    )
    assert np.array_equal(dml_data_from_array.data, dml_data.data)

    dml_data = make_did_SZ2020(n_obs=100, cross_sectional_data=False)
    dml_data_from_array = DoubleMLData.from_arrays(
        x=dml_data.data[dml_data.x_cols], y=dml_data.data[dml_data.y_col], d=dml_data.data[dml_data.d_cols]
    )
    assert np.array_equal(dml_data_from_array.data, dml_data.data)

    dml_data = make_did_SZ2020(n_obs=100, cross_sectional_data=True)
    dml_data_from_array = DoubleMLData.from_arrays(
        x=dml_data.data[dml_data.x_cols],
        y=dml_data.data[dml_data.y_col],
        d=dml_data.data[dml_data.d_cols],
        t=dml_data.data[dml_data.t_col],
    )
    assert np.array_equal(dml_data_from_array.data, dml_data.data)

    # check with instrument and time variable
    dml_data = make_did_SZ2020(n_obs=100, cross_sectional_data=True)
    dml_data.data["z"] = dml_data.data["t"]
    dml_data_from_array = DoubleMLData.from_arrays(
        x=dml_data.data[dml_data.x_cols],
        y=dml_data.data[dml_data.y_col],
        d=dml_data.data[dml_data.d_cols],
        z=dml_data.data["z"],
        t=dml_data.data[dml_data.t_col],
    )
    assert np.array_equal(dml_data_from_array.data, dml_data.data)


@pytest.mark.ci
def test_add_vars_in_df():
    # additional variables in the df shouldn't affect results
    np.random.seed(3141)
    df = make_plr_CCDDHNR2018(n_obs=100, return_type="DataFrame")
    dml_data_full_df = DoubleMLData(df, "y", "d", ["X1", "X2", "X3"])
    df_subset = df.loc[:, ["X1", "X2", "X3", "y", "d"]]
    dml_data_subset = DoubleMLData(df_subset, "y", "d", ["X1", "X2", "X3"])
    dml_plr_full_df = DoubleMLPLR(dml_data_full_df, Lasso(), Lasso())
    dml_plr_subset = DoubleMLPLR(dml_data_subset, Lasso(), Lasso(), draw_sample_splitting=False)
    dml_plr_subset.set_sample_splitting(dml_plr_full_df.smpls)
    dml_plr_full_df.fit()
    dml_plr_subset.fit()
    assert np.allclose(dml_plr_full_df.coef, dml_plr_subset.coef, rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_full_df.se, dml_plr_subset.se, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_no_instr_no_time_no_selection():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    assert dml_data.z is None
    assert dml_data.n_instr == 0
    assert dml_data.t is None

    x, y, d = make_plr_CCDDHNR2018(n_obs=100, return_type="array")
    dml_data = DoubleMLData.from_arrays(x, y, d)
    assert dml_data.z is None
    assert dml_data.n_instr == 0
    assert dml_data.t is None
    assert dml_data.s is None


@pytest.mark.ci
def test_dml_summary_with_time():
    dml_data_did_cs = make_did_SZ2020(n_obs=200, cross_sectional_data=True)
    dml_did_cs = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression())
    assert isinstance(dml_did_cs.__str__(), str)
    assert isinstance(dml_did_cs.summary, pd.DataFrame)

    dml_did_cs.fit()
    assert isinstance(dml_did_cs.__str__(), str)
    assert isinstance(dml_did_cs.summary, pd.DataFrame)


@pytest.mark.ci
def test_dml_summary_with_selection():
    dml_data_ssm = make_ssm_data(n_obs=200)
    dml_ssm = DoubleMLSSM(dml_data_ssm, Lasso(), LogisticRegression(), LogisticRegression())
    assert isinstance(dml_ssm.__str__(), str)
    assert isinstance(dml_ssm.summary, pd.DataFrame)

    dml_ssm.fit()
    assert isinstance(dml_ssm.__str__(), str)
    assert isinstance(dml_ssm.summary, pd.DataFrame)


@pytest.mark.ci
def test_x_cols_setter_defaults():
    df = pd.DataFrame(np.tile(np.arange(4), (4, 1)), columns=["yy", "dd", "xx1", "xx2"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument
    df = pd.DataFrame(np.tile(np.arange(5), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "zz"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", z_cols="zz")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # without instrument with time
    df = pd.DataFrame(np.tile(np.arange(5), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "tt"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", t_col="tt")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument with time
    df = pd.DataFrame(np.tile(np.arange(6), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "zz", "tt"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", z_cols="zz", t_col="tt")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # without instrument with selection
    df = pd.DataFrame(np.tile(np.arange(5), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "ss"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument with selection
    df = pd.DataFrame(np.tile(np.arange(6), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "zz", "ss"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", z_cols="zz", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with selection and time
    df = pd.DataFrame(np.tile(np.arange(6), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "tt", "ss"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", t_col="tt", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]

    # with instrument, selection and time
    df = pd.DataFrame(np.tile(np.arange(7), (4, 1)), columns=["yy", "dd", "xx1", "xx2", "zz", "tt", "ss"])
    dml_data = DoubleMLData(df, y_col="yy", d_cols="dd", z_cols="zz", t_col="tt", s_col="ss")
    assert dml_data.x_cols == ["xx1", "xx2"]


@pytest.mark.ci
def test_x_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    orig_x_cols = dml_data.x_cols

    # check that after changing the x_cols, the x array gets updated
    x_comp = dml_data.data[["X1", "X11", "X13"]].values
    dml_data.x_cols = ["X1", "X11", "X13"]
    assert np.array_equal(dml_data.x, x_comp)

    msg = "Invalid covariates x_cols. At least one covariate is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.x_cols = ["X1", "X11", "A13"]

    msg = r"The covariates x_cols must be of str or list type \(or None\). " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.x_cols = 5

    # check single covariate
    x_comp = dml_data.data[["X13"]].values
    dml_data.x_cols = "X13"
    assert np.array_equal(dml_data.x, x_comp)

    # check setting None brings us back to orig_x_cols
    x_comp = dml_data.data[orig_x_cols].values
    dml_data.x_cols = None
    assert np.array_equal(dml_data.x, x_comp)


@pytest.mark.ci
def test_d_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(7)] + ["y", "d1", "d2"]
    dml_data = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(7)])

    # check that after changing d_cols, the d array gets updated
    d_comp = dml_data.data["d2"].values
    dml_data.d_cols = ["d2", "d1"]
    assert np.array_equal(dml_data.d, d_comp)

    msg = r"Invalid treatment variable\(s\) d_cols. At least one treatment variable is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = ["d1", "d13"]
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = "d13"

    msg = r"The treatment variable\(s\) d_cols must be of str or list type. " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.d_cols = 5

    # check single treatment variable
    d_comp = dml_data.data["d2"].values
    dml_data.d_cols = "d2"
    assert np.array_equal(dml_data.d, d_comp)
    assert dml_data.n_treat == 1


@pytest.mark.ci
def test_z_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(4)] + [f"z{i + 1}" for i in np.arange(3)] + ["y", "d1", "d2"]
    dml_data = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(4)], [f"z{i + 1}" for i in np.arange(3)])

    # check that after changing z_cols, the z array gets updated
    z_comp = dml_data.data[["z1", "z2"]].values
    dml_data.z_cols = ["z1", "z2"]
    assert np.array_equal(dml_data.z, z_comp)

    msg = r"Invalid instrumental variable\(s\) z_cols. At least one instrumental variable is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = ["z1", "a13"]
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = "a13"

    msg = (
        r"The instrumental variable\(s\) z_cols must be of str or list type \(or None\). "
        "5 of type <class 'int'> was passed."
    )
    with pytest.raises(TypeError, match=msg):
        dml_data.z_cols = 5

    # check single instrument
    z_comp = dml_data.data[["z2"]].values
    dml_data.z_cols = "z2"
    assert np.array_equal(dml_data.z, z_comp)

    # check None
    dml_data.z_cols = None
    assert dml_data.n_instr == 0
    assert dml_data.z is None


@pytest.mark.ci
def test_t_col_setter():
    np.random.seed(3141)
    df = make_did_SZ2020(n_obs=100, cross_sectional_data=True, return_type=pd.DataFrame)
    df["t_new"] = np.ones(shape=(100,))
    dml_data = DoubleMLData(df, "y", "d", [f"Z{i + 1}" for i in np.arange(4)], t_col="t")

    # check that after changing t_col, the t array gets updated
    t_comp = dml_data.data["t_new"].values
    dml_data.t_col = "t_new"
    assert np.array_equal(dml_data.t, t_comp)

    msg = r"Invalid time variable t_col. a13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = "a13"

    msg = r"The time variable t_col must be of str type \(or None\). " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.t_col = 5

    # check None
    dml_data.t_col = None
    assert dml_data.t is None


@pytest.mark.ci
def test_s_col_setter():
    np.random.seed(3141)
    df = make_ssm_data(n_obs=100, return_type=pd.DataFrame)
    df["s_new"] = np.ones(shape=(100,))
    dml_data = DoubleMLData(df, "y", "d", [f"X{i + 1}" for i in np.arange(4)], s_col="s")

    # check that after changing s_col, the s array gets updated
    s_comp = dml_data.data["s_new"].values
    dml_data.s_col = "s_new"
    assert np.array_equal(dml_data.s, s_comp)

    msg = r"Invalid score or selection variable s_col. a13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.s_col = "a13"

    msg = r"The score or selection variable s_col must be of str type \(or None\). " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.s_col = 5

    # check None
    dml_data.s_col = None
    assert dml_data.s is None


@pytest.mark.ci
def test_y_col_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(7)] + ["y", "y123", "d"]
    dml_data = DoubleMLData(df, "y", "d", [f"X{i + 1}" for i in np.arange(7)])

    # check that after changing y_col, the y array gets updated
    y_comp = dml_data.data["y123"].values
    dml_data.y_col = "y123"
    assert np.array_equal(dml_data.y, y_comp)

    msg = r"Invalid outcome variable y_col. d13 is no data column."
    with pytest.raises(ValueError, match=msg):
        dml_data.y_col = "d13"

    msg = r"The outcome variable y_col must be of str type. " "5 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.y_col = 5


@pytest.mark.ci
def test_use_other_treat_as_covariate():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f"X{i + 1}" for i in np.arange(7)] + ["y", "d1", "d2"]
    dml_data = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(7)], use_other_treat_as_covariate=True)
    dml_data.set_x_d("d1")
    assert np.array_equal(dml_data.d, df["d1"].values)
    assert np.array_equal(dml_data.x, df[[f"X{i + 1}" for i in np.arange(7)] + ["d2"]].values)
    dml_data.set_x_d("d2")
    assert np.array_equal(dml_data.d, df["d2"].values)
    assert np.array_equal(dml_data.x, df[[f"X{i + 1}" for i in np.arange(7)] + ["d1"]].values)

    dml_data = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(7)], use_other_treat_as_covariate=False)
    dml_data.set_x_d("d1")
    assert np.array_equal(dml_data.d, df["d1"].values)
    assert np.array_equal(dml_data.x, df[[f"X{i + 1}" for i in np.arange(7)]].values)
    dml_data.set_x_d("d2")
    assert np.array_equal(dml_data.d, df["d2"].values)
    assert np.array_equal(dml_data.x, df[[f"X{i + 1}" for i in np.arange(7)]].values)

    dml_data.use_other_treat_as_covariate = True
    assert np.array_equal(dml_data.d, df["d1"].values)
    assert np.array_equal(dml_data.x, df[[f"X{i + 1}" for i in np.arange(7)] + ["d2"]].values)

    msg = "use_other_treat_as_covariate must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData(df, "y", ["d1", "d2"], [f"X{i + 1}" for i in np.arange(7)], use_other_treat_as_covariate=1)

    msg = "Invalid treatment_var. d3 is not in d_cols."
    with pytest.raises(ValueError, match=msg):
        dml_data.set_x_d("d3")

    msg = r"treatment_var must be of str type. \['d1', 'd2'\] of type <class 'list'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.set_x_d(["d1", "d2"])


@pytest.mark.ci
def test_disjoint_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(6), (4, 1)), columns=["yy", "dd1", "xx1", "xx2", "zz", "tt"])

    msg = (
        r"At least one variable/column is set as treatment variable \(``d_cols``\) and as covariate\(``x_cols``\). "
        "Consider using parameter ``use_other_treat_as_covariate``."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1", "xx1"], x_cols=["xx1", "xx2"])
    msg = "yy cannot be set as outcome variable ``y_col`` and treatment variable in ``d_cols``"
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1", "yy"], x_cols=["xx1", "xx2"])
    msg = "yy cannot be set as outcome variable ``y_col`` and covariate in ``x_cols``"
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "yy", "xx2"])

    # instrumental variable
    msg = r"At least one variable/column is set as outcome variable \(``y_col``\) and instrumental variable \(``z_cols``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], z_cols="yy")
    msg = r"At least one variable/column is set as treatment variable \(``d_cols``\) and instrumental variable \(``z_cols``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], z_cols=["dd1"])
    msg = r"At least one variable/column is set as covariate \(``x_cols``\) and instrumental variable \(``z_cols``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], z_cols="xx2")

    # time variable
    msg = r"At least one variable/column is set as outcome variable \(``y_col``\) and time variable \(``t_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="yy")
    msg = r"At least one variable/column is set as treatment variable \(``d_cols``\) and time variable \(``t_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="dd1")
    msg = r"At least one variable/column is set as covariate \(``x_cols``\) and time variable \(``t_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="xx2")
    msg = r"At least one variable/column is set as instrumental variable \(``z_cols``\) and time variable \(``t_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], z_cols="zz", t_col="zz")

    # score or selection variable
    msg = (
        r"At least one variable/column is set as outcome variable \(``y_col``\) and score or selection variable \(``s_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], s_col="yy")
    msg = (
        r"At least one variable/column is set as treatment variable \(``d_cols``\) "
        r"and score or selection variable \(``s_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], s_col="dd1")
    msg = r"At least one variable/column is set as covariate \(``x_cols``\) and score or selection variable \(``s_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], s_col="xx2")
    msg = (
        r"At least one variable/column is set as instrumental variable \(``z_cols``\) "
        r"and score or selection variable \(``s_col``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], z_cols="zz", s_col="zz")
    msg = r"At least one variable/column is set as time variable \(``t_col``\) and score or selection variable \(``s_col``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col="yy", d_cols=["dd1"], x_cols=["xx1", "xx2"], t_col="tt", s_col="tt")


@pytest.mark.ci
def test_duplicates():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)

    msg = r"Invalid treatment variable\(s\) d_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col="y", d_cols=["d", "d", "X1"], x_cols=["X3", "X2"])
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = ["d", "d", "X1"]

    msg = "Invalid covariates x_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col="y", d_cols=["d"], x_cols=["X3", "X2", "X3"])
    with pytest.raises(ValueError, match=msg):
        dml_data.x_cols = ["X3", "X2", "X3"]

    msg = r"Invalid instrumental variable\(s\) z_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col="y", d_cols=["d"], x_cols=["X3", "X2"], z_cols=["X15", "X12", "X12", "X15"])
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = ["X15", "X12", "X12", "X15"]

    msg = "Invalid pd.DataFrame: Contains duplicate column names."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(
            pd.DataFrame(np.zeros((100, 5)), columns=["y", "d", "X3", "X2", "y"]), y_col="y", d_cols=["d"], x_cols=["X3", "X2"]
        )


@pytest.mark.ci
def test_dml_datatype():
    data_array = np.zeros((100, 10))
    # msg = ('data must be of pd.DataFrame type. '
    #        f'{str(data_array)} of type {str(type(data_array))} was passed.')
    with pytest.raises(TypeError):
        _ = DoubleMLData(data_array, y_col="y", d_cols=["d"], x_cols=["X3", "X2"])


@pytest.mark.ci
def test_dml_data_w_missings(generate_data_irm_w_missings):
    (x, y, d) = generate_data_irm_w_missings

    dml_data = DoubleMLData.from_arrays(x, y, d, force_all_x_finite=False)

    _ = DoubleMLData.from_arrays(x, y, d, force_all_x_finite="allow-nan")

    msg = r"Input contains NaN."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(x, y, d, force_all_x_finite=True)

    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(x, x[:, 0], d, force_all_x_finite=False)

    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(x, y, x[:, 0], force_all_x_finite=False)

    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(x, y, d, x[:, 0], force_all_x_finite=False)

    msg = r"Input contains infinity or a value too large for dtype\('float64'\)."
    xx = np.copy(x)
    xx[0, 0] = np.inf
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(xx, y, d, force_all_x_finite="allow-nan")

    msg = "Invalid force_all_x_finite. force_all_x_finite must be True, False or 'allow-nan'."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData.from_arrays(xx, y, d, force_all_x_finite=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col="y", d_cols="d", force_all_x_finite=1)

    msg = "Invalid force_all_x_finite allownan. force_all_x_finite must be True, False or 'allow-nan'."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(xx, y, d, force_all_x_finite="allownan")
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col="y", d_cols="d", force_all_x_finite="allownan")

    msg = r"Input contains NaN."
    with pytest.raises(ValueError, match=msg):
        dml_data.force_all_x_finite = True

    assert dml_data.force_all_x_finite is True
    dml_data.force_all_x_finite = False
    assert dml_data.force_all_x_finite is False
    dml_data.force_all_x_finite = "allow-nan"
    assert dml_data.force_all_x_finite == "allow-nan"


def test_dml_data_w_missing_d(generate_data1):
    data = generate_data1
    np.random.seed(3141)
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()

    pd_args = {
        "data": data,
        "y_col": "y",
        "d_cols": ["d"],
        "x_cols": x_cols,
    }
    dml_data = DoubleMLData(force_all_d_finite=True, **pd_args)

    data["d"] = np.nan
    np_args = {
        "x": data.loc[:, x_cols].values,
        "y": data["y"].values,
        "d": data["d"].values,
    }
    msg = r"Input contains NaN."
    with pytest.raises(ValueError, match=msg):
        dml_data2 = DoubleMLData(force_all_d_finite=False, **pd_args)
        dml_data2.force_all_d_finite = True
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(force_all_d_finite=True, **np_args)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(force_all_d_finite=True, **pd_args)

    data["d"] = np.inf
    np_args = {
        "x": data.loc[:, x_cols].values,
        "y": data["y"].values,
        "d": data["d"].values,
    }
    msg = r"Input contains infinity or a value too large for dtype\('float64'\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData.from_arrays(force_all_d_finite=True, **np_args)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(force_all_d_finite=True, **pd_args)

    msg = "Invalid force_all_d_finite. force_all_d_finite must be True, False or 'allow-nan'."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData(force_all_d_finite=1, **pd_args)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData.from_arrays(force_all_d_finite=1, **np_args)

    data["d"] = 1.0
    assert dml_data.force_all_d_finite is True
    dml_data.force_all_d_finite = False
    assert dml_data.force_all_d_finite is False
    dml_data.force_all_d_finite = "allow-nan"
    assert dml_data.force_all_d_finite == "allow-nan"
