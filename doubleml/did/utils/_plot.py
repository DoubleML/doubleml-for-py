import numpy as np
import pandas as pd


def add_jitter(data, x_col, is_datetime=None, jitter_value=None, default_jitter=0.1):
    """
    Adds jitter to duplicate x-values for better visibility.

    Args:
        data (DataFrame): The subset of the dataset to jitter.
        x_col (str): Column name for x values.
        is_datetime (bool): Whether the x-values are datetime objects. If None, will be detected.
        jitter_value (float or timedelta): Jitter amount. If None, will be auto-calculated, based on default jitter.
        default_jitter (float): Default jitter amount as a fraction of the smallest difference between x-values.

    Returns:
        DataFrame with an additional 'jittered_x' column.
    """
    if data.empty:
        return data

    data = data.copy()

    # Auto-detect datetime if not specified
    if is_datetime is None:
        is_datetime = pd.api.types.is_datetime64_any_dtype(data[x_col])

    # Auto-calculate jitter if not specified
    if jitter_value is None:
        all_values = sorted(data[x_col].unique())
        if len(all_values) > 1:
            if is_datetime:
                jitter_value = (all_values[1] - all_values[0]).total_seconds() * default_jitter
            else:
                jitter_value = (all_values[1] - all_values[0]) * default_jitter
        else:
            jitter_value = default_jitter

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
