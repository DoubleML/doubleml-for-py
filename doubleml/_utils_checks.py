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


def _check_pos_integer(value, name):
    if not isinstance(value, int):
        raise TypeError(f'{name} has to be an integer.'
                        f' {str(value)} of type {str(type(value))} was passed.')
    if value < 0:
        raise ValueError(f'{name} has to be larger or equal to zero.'
                         f' {str(value)} was passed.')
    return
