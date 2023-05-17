def _check_in_zero_one(value, name, include_zero=True, include_one=True):
    if not isinstance(value, float):
        raise TypeError(f'{name} must be of float type. '
                        f'{str(value)} of type {str(type(value))} was passed.')
    if include_zero & include_one:
        if (value < 0) | (value > 1):
            raise ValueError(f'{name} must be in [0,1]. '
                             f'{str(value)} was passed.')
    elif (not include_zero) & include_one:
        if (value <= 0) | (value > 1):
            raise ValueError(f'{name} must be in (0,1]. '
                             f'{str(value)} was passed.')
    elif include_zero & (not include_one):
        if (value < 0) | (value >= 1):
            raise ValueError(f'{name} must be in [0,1). '
                             f'{str(value)} was passed.')
    else:
        if (value <= 0) | (value >= 1):
            raise ValueError(f'{name} must be in (0,1). '
                             f'{str(value)} was passed.')
    return


def _check_integer(value, name, lower_bound=None, upper_bound=None):
    if not isinstance(value, int):
        raise TypeError(f'{name} must be an integer.'
                        f' {str(value)} of type {str(type(value))} was passed.')
    if lower_bound is not None:
        if value < lower_bound:
            raise ValueError(f'{name} must be larger or equal to {lower_bound}. '
                             f'{str(value)} was passed.')
    if upper_bound is not None:
        if value > upper_bound:
            raise ValueError(f'{name} must be smaller or equal to {upper_bound}. '
                             f'{str(value)} was passed.')
    return


def _check_float(value, name, lower_bound=None, upper_bound=None):
    if not isinstance(value, float):
        raise TypeError(f'{name} must be of float type.'
                        f' {str(value)} of type {str(type(value))} was passed.')
    if lower_bound is not None:
        if value < lower_bound:
            raise ValueError(f'{name} must be larger or equal to {lower_bound}. '
                             f'{str(value)} was passed.')
    if upper_bound is not None:
        if value > upper_bound:
            raise ValueError(f'{name} must be smaller or equal to {upper_bound}. '
                             f'{str(value)} was passed.')


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f'{name} has to be boolean.'
                        f' {str(value)} of type {str(type(value))} was passed.')
