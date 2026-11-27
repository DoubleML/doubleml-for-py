import numpy as np
import pandas as pd
import pytest

from doubleml.data import DoubleMLPanelData


@pytest.fixture
def sample_data():
    n_ids = 3
    n_periods = 4

    data = []
    for id_val in range(n_ids):
        for t in range(n_periods):
            data.append(
                {
                    "id": f"ID_{id_val}",
                    "time": t,
                    "y": np.random.normal(),
                    "treatment": int(t >= 2),
                    "x1": np.random.normal(),
                    "x2": np.random.normal(),
                    "z": np.random.normal(),
                }
            )

    return pd.DataFrame(data)


@pytest.mark.ci
def test_multiple_treatments_exception(sample_data):
    # Test exception when more than one treatment column is provided
    with pytest.raises(ValueError, match="Only one treatment column is allowed for panel data."):
        # Create copy of data with an additional treatment column
        data_multi = sample_data.copy()
        data_multi["treatment2"] = np.random.binomial(1, 0.5, size=len(data_multi))
        DoubleMLPanelData(data=data_multi, y_col="y", d_cols=["treatment", "treatment2"], t_col="time", id_col="id")


@pytest.mark.ci
def test_id_col_type_exception(sample_data):
    # Test exception when id_col is not a string
    with pytest.raises(TypeError, match="The id variable id_col must be of str type."):
        DoubleMLPanelData(data=sample_data, y_col="y", d_cols="treatment", t_col="time", id_col=123)  # Should be a string


@pytest.mark.ci
def test_id_col_not_in_data(sample_data):
    # Test exception when id_col doesn't exist in data
    with pytest.raises(ValueError, match="Invalid id variable id_col. non_existent_id is no data column."):
        DoubleMLPanelData(data=sample_data, y_col="y", d_cols="treatment", t_col="time", id_col="non_existent_id")


@pytest.mark.ci
def test_time_col_none_exception(sample_data):
    # Test exception when t_col is None
    with pytest.raises(TypeError, match="Invalid time variable t_col. Time variable required for panel data."):
        DoubleMLPanelData(
            data=sample_data, y_col="y", d_cols="treatment", t_col=None, id_col="id"  # Should not be None for panel data
        )


@pytest.mark.ci
def test_time_var_data_type_exception(sample_data):
    # Test exception when time var is not int, float or datetime
    msg = (
        r"Invalid data type for time variable: expected one of \(<class 'numpy.integer'>, <class 'numpy.floating'>, "
        r"<class 'numpy.datetime64'>\)"
    )
    with pytest.raises(ValueError, match=msg):
        data_time_type = sample_data.copy()
        data_time_type["time"] = data_time_type["time"].astype(str)
        DoubleMLPanelData(data=data_time_type, y_col="y", d_cols="treatment", t_col="time", id_col="id")


@pytest.mark.ci
def test_overlapping_variables_exception(sample_data):
    # Test exception when id_col overlaps with another variable
    msg = r"At least one variable/column is set as outcome variable \(``y_col``\) and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        DoubleMLPanelData(
            data=sample_data,
            y_col="id",  # Using id as outcome variable
            d_cols="treatment",
            t_col="time",
            id_col="id",  # Same as y_col
        )

    # Test treatment variable overlapping
    msg = r"At least one variable/column is set as treatment variable \(``d_cols``\) and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        DoubleMLPanelData(data=sample_data, y_col="y", d_cols="id", t_col="time", id_col="id")  # Using id as treatment

    # Test time variable overlapping
    # using t_col="id", id_col="id" gives invalid data type for time variable exception first
    msg = r"At least one variable/column is set as time variable \(``t_col``\) and identifier variable \(``id_col``\)."
    with pytest.raises(ValueError, match=msg):
        DoubleMLPanelData(data=sample_data, y_col="y", d_cols="treatment", t_col="time", id_col="time")  # Using time as id


@pytest.mark.ci
@pytest.mark.parametrize(
    ("t_col", "msg"),
    [
        ("y", r"At least one variable/column is set as outcome variable \(``y_col``\) and time variable \(``t_col``\)."),
        (
            "treatment",
            r"At least one variable/column is set as treatment variable \(``d_cols``\) and time variable \(``t_col``\).",
        ),
        ("x1", r"At least one variable/column is set as covariate \(``x_cols``\) and time variable \(``t_col``\)."),
        (
            "z",
            r"At least one variable/column is set as instrumental variable \(``z_cols``\) and time variable \(``t_col``\).",
        ),
        (
            "id",
            r"At least one variable/column is set as time variable \(``t_col``\) and identifier variable \(``id_col``\).",
        ),
    ],
)
def test_t_col_reassignment_disjoint_sets(sample_data, t_col, msg):
    dml_data = DoubleMLPanelData(
        data=sample_data,
        y_col="y",
        d_cols="treatment",
        t_col="time",
        id_col="id",
        x_cols=["x1", "x2"],
        z_cols="z",
    )
    expected_t = dml_data.t.copy()

    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = t_col

    assert dml_data.t_col == "time"
    np.testing.assert_array_equal(dml_data.t, expected_t)


@pytest.mark.ci
def test_t_col_reassignment_invalid_data_type(sample_data):
    data = sample_data.copy()
    data["invalid_time"] = data["time"].astype(str)
    dml_data = DoubleMLPanelData(
        data=data,
        y_col="y",
        d_cols="treatment",
        t_col="time",
        id_col="id",
        x_cols=["x1", "x2"],
    )
    expected_t = dml_data.t.copy()
    expected_t_values = dml_data.t_values.copy()
    msg = (
        r"Invalid data type for time variable: expected one of \(<class 'numpy.integer'>, <class 'numpy.floating'>, "
        r"<class 'numpy.datetime64'>\)"
    )

    with pytest.raises(ValueError, match=msg):
        dml_data.t_col = "invalid_time"

    assert dml_data.t_col == "time"
    np.testing.assert_array_equal(dml_data.t, expected_t)
    np.testing.assert_array_equal(dml_data.t_values, expected_t_values)


@pytest.mark.ci
@pytest.mark.parametrize(
    ("id_col", "msg"),
    [
        (
            "x1",
            r"At least one variable/column is set as covariate \(``x_cols``\) and identifier variable \(``id_col``\).",
        ),
        (
            "time",
            r"At least one variable/column is set as time variable \(``t_col``\) and identifier variable \(``id_col``\).",
        ),
        (
            "treatment",
            r"At least one variable/column is set as treatment variable \(``d_cols``\) and identifier variable "
            r"\(``id_col``\).",
        ),
        (
            "y",
            r"At least one variable/column is set as outcome variable \(``y_col``\) and identifier variable \(``id_col``\).",
        ),
        (
            "z",
            r"At least one variable/column is set as instrumental variable \(``z_cols``\) and identifier variable "
            r"\(``id_col``\).",
        ),
    ],
)
def test_id_col_reassignment_disjoint_sets(sample_data, id_col, msg):
    dml_data = DoubleMLPanelData(
        data=sample_data,
        y_col="y",
        d_cols="treatment",
        t_col="time",
        id_col="id",
        x_cols=["x1", "x2"],
        z_cols="z",
    )
    expected_id_var = dml_data.id_var.copy()

    with pytest.raises(ValueError, match=msg):
        dml_data.id_col = id_col

    assert dml_data.id_col == "id"
    np.testing.assert_array_equal(dml_data.id_var, expected_id_var)


@pytest.mark.ci
def test_id_col_reassignment_non_finite(sample_data):
    data = sample_data.copy()
    data["invalid_id"] = np.arange(len(data), dtype=float)
    data.loc[0, "invalid_id"] = np.nan
    dml_data = DoubleMLPanelData(
        data=data,
        y_col="y",
        d_cols="treatment",
        t_col="time",
        id_col="id",
        x_cols=["x1", "x2"],
    )
    expected_id_var = dml_data.id_var.copy()
    expected_id_var_unique = dml_data.id_var_unique.copy()

    with pytest.raises(ValueError, match="Input contains NaN."):
        dml_data.id_col = "invalid_id"

    assert dml_data.id_col == "id"
    np.testing.assert_array_equal(dml_data.id_var, expected_id_var)
    np.testing.assert_array_equal(dml_data.id_var_unique, expected_id_var_unique)


@pytest.mark.ci
def test_from_arrays_not_implemented():
    # Test that from_arrays raises NotImplementedError
    with pytest.raises(NotImplementedError, match="from_arrays is not implemented for DoubleMLPanelData"):
        DoubleMLPanelData.from_arrays(
            x=np.random.normal(size=(10, 2)),
            y=np.random.normal(size=10),
            d=np.random.binomial(1, 0.5, size=10),
            t=np.arange(10),
            identifier=np.arange(10),
        )


@pytest.mark.ci
def test_invalid_datetime_unit(sample_data):
    with pytest.raises(ValueError, match="Invalid datetime unit."):
        DoubleMLPanelData(
            data=sample_data, y_col="y", d_cols="treatment", t_col="time", id_col="id", datetime_unit="invalid_unit"
        )


# test if no exception is raised
@pytest.mark.ci
def test_no_exception(sample_data):
    DoubleMLPanelData(data=sample_data, y_col="y", d_cols="treatment", t_col="time", id_col="id")
    assert True
