import pandas as pd

valid_datetime_units = {"Y", "M", "D", "h", "m", "s", "ms", "us", "ns"}

# Units that can be used with pd.Timedelta (unambiguous)
timedelta_compatible_units = {"D", "h", "m", "s", "ms", "us", "ns"}

# Units that require period arithmetic (ambiguous)
period_only_units = {"Y", "M"}


def _is_valid_datetime_unit(unit):
    if unit not in valid_datetime_units:
        raise ValueError("Invalid datetime unit.")
    else:
        return unit


def _is_timedelta_compatible(unit):
    """Check if a datetime unit can be used with pd.Timedelta."""
    return unit in timedelta_compatible_units


def _subtract_periods_safe(datetime_values, reference_datetime, periods, unit):
    """
    Safely subtract periods from datetime values, handling both timedelta-compatible
    and period-only units.

    Parameters
    ----------
    datetime_values : pandas.Series or numpy.array
        Array of datetime values to compare
    reference_datetime : datetime-like
        Reference datetime to subtract periods from
    periods : int
        Number of periods to subtract
    unit : str
        Datetime unit

    Returns
    -------
    numpy.array
        Boolean array indicating which datetime_values are >= (reference_datetime - periods)
    """
    if periods == 0:
        # No anticipation periods, so no datetime arithmetic needed
        return datetime_values >= reference_datetime

    if _is_timedelta_compatible(unit):
        # Use Timedelta for unambiguous units
        period_offset = pd.Timedelta(periods, unit=unit)
        return datetime_values >= (reference_datetime - period_offset)
    else:
        # Use Period arithmetic for ambiguous units like 'M' and 'Y'
        ref_period = pd.Period(reference_datetime, freq=unit)
        ref_minus_periods = ref_period - periods
        datetime_periods = pd.PeriodIndex(datetime_values, freq=unit)
        return datetime_periods >= ref_minus_periods
