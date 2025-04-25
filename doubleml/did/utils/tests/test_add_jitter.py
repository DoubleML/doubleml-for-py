from datetime import datetime, timedelta

import pandas as pd
import pytest

from doubleml.did.utils._plot import add_jitter


@pytest.fixture
def numeric_df_no_duplicates():
    """Create a DataFrame with numeric x values and no duplicates."""
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


@pytest.fixture
def numeric_df_with_duplicates():
    """Create a DataFrame with numeric x values and duplicates."""
    return pd.DataFrame({"x": [1, 1, 2, 2, 2, 3], "y": [10, 15, 20, 25, 30, 35]})


@pytest.fixture
def datetime_df_with_duplicates():
    """Create a DataFrame with datetime x values and duplicates."""
    base_date = datetime(2023, 1, 1)
    return pd.DataFrame(
        {
            "x": [
                base_date,
                base_date,
                base_date + timedelta(days=1),
                base_date + timedelta(days=1),
                base_date + timedelta(days=2),
            ],
            "y": [10, 15, 20, 25, 30],
        }
    )


@pytest.mark.ci
def test_add_jitter_numeric_no_duplicates(numeric_df_no_duplicates):
    """Test that no jitter is added when there are no duplicates."""
    result = add_jitter(numeric_df_no_duplicates, "x")
    # No jitter should be added when there are no duplicates
    pd.testing.assert_series_equal(result["jittered_x"], result["x"], check_names=False)


@pytest.mark.ci
def test_add_jitter_numeric_with_duplicates(numeric_df_with_duplicates):
    """Test that jitter is added correctly to numeric values with duplicates."""
    result = add_jitter(numeric_df_with_duplicates, "x", jitter_value=0.1)

    # Check that all original x-values have jitter applied
    for x_val in numeric_df_with_duplicates["x"].unique():
        mask = numeric_df_with_duplicates["x"] == x_val
        count = mask.sum()
        if count > 1:
            jittered_x = result.loc[mask, "jittered_x"]
            # Check that jittered values are different from original
            assert not (jittered_x == x_val).all()
            # Check that jittered values are symmetric around original
            assert abs(jittered_x.mean() - x_val) < 1e-10


@pytest.mark.ci
def test_add_jitter_datetime(datetime_df_with_duplicates):
    """Test that jitter is added correctly to datetime values."""
    result = add_jitter(datetime_df_with_duplicates, "x", jitter_value=20)

    # Check that result contains jittered_x column with datetime type
    assert pd.api.types.is_datetime64_dtype(result["jittered_x"])

    # Check that duplicates have different jittered values
    for x_val in datetime_df_with_duplicates["x"].unique():
        mask = datetime_df_with_duplicates["x"] == x_val
        count = mask.sum()
        if count > 1:
            jittered_values = result.loc[mask, "jittered_x"].tolist()
            # All jittered values should be unique
            assert len(set(jittered_values)) == count


@pytest.mark.ci
def test_add_jitter_empty_df():
    """Test behavior with empty DataFrame."""
    empty_df = pd.DataFrame({"x": [], "y": []})
    result = add_jitter(empty_df, "x")
    assert result.empty


@pytest.mark.ci
def test_add_jitter_explicit_value(numeric_df_with_duplicates):
    """Test with explicitly specified jitter value."""
    explicit_jitter = 0.5
    result = add_jitter(numeric_df_with_duplicates, "x", jitter_value=explicit_jitter)

    # Check that maximum jitter is equal to or less than the specified value
    for x_val in numeric_df_with_duplicates["x"].unique():
        mask = numeric_df_with_duplicates["x"] == x_val
        if mask.sum() > 1:
            max_diff = (result.loc[mask, "jittered_x"] - x_val).abs().max()
            assert max_diff <= explicit_jitter


@pytest.mark.ci
def test_add_jitter_single_unique_value():
    """Test with DataFrame having only one unique x value."""
    df = pd.DataFrame({"x": [5, 5, 5], "y": [1, 2, 3]})
    result = add_jitter(df, "x", jitter_value=0.1)

    # Check that jitter was applied
    assert not (result["jittered_x"] == 5).all()

    # Check that jittered values are centered around the original value
    assert abs(result["jittered_x"].mean() - 5) < 1e-10


@pytest.mark.ci
def test_add_jitter_explicit_datetime_flag():
    """Test with explicitly specified is_datetime flag."""
    # Create DataFrame with string dates
    df = pd.DataFrame({"x": ["2023-01-01", "2023-01-01", "2023-01-02"], "y": [10, 15, 20]})

    # Without specifying is_datetime, it would treat as strings
    with pytest.raises(TypeError):
        _ = add_jitter(df, "x")

    # With is_datetime=True, it should convert and jitter as datetimes
    with pytest.raises(TypeError):
        # This should fail because strings can't be converted to datetime implicitly
        add_jitter(df, "x", is_datetime=True)
