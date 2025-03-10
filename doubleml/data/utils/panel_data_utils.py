valid_datetime_units = {"Y", "M", "D", "h", "m", "s", "ms", "us", "ns"}


def _is_valid_datetime_unit(unit):
    if unit not in valid_datetime_units:
        raise ValueError("Invalid datetime unit.")
    else:
        return unit
