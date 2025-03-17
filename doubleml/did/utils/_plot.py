import numpy as np
import pandas as pd


def add_jitter(data, x_col, is_datetime=None, jitter_value=None):
    """
    Adds jitter to duplicate x-values for better visibility.

    Args:
        data (DataFrame): The subset of the dataset to jitter.
        x_col (str): Column name for x values.
        is_datetime (bool): Whether the x-values are datetime objects. If None, will be detected.
        jitter_value (float or timedelta): Jitter amount.

    Returns:
        DataFrame with an additional 'jittered_x' column.
    """
    if data.empty:
        return data

    data = data.copy()

    # Auto-detect datetime if not specified
    if is_datetime is None:
        is_datetime = pd.api.types.is_datetime64_any_dtype(data[x_col])

    # Initialize jittered_x with original values
    data["jittered_x"] = data[x_col]

    for x_val in data[x_col].unique():
        mask = data[x_col] == x_val
        count = mask.sum()
        if count > 1:
            # Create evenly spaced jitter values
            if is_datetime:
                jitters = [pd.Timedelta(seconds=float(j)) for j in np.linspace(-jitter_value, jitter_value, count)]
            else:
                jitters = np.linspace(-jitter_value, jitter_value, count)

            # Apply jitter to each duplicate point
            data.loc[mask, "jitter_index"] = range(count)
            for i, j in enumerate(jitters):
                data.loc[mask & (data["jitter_index"] == i), "jittered_x"] = x_val + j

    return data
