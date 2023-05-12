import warnings

def _check_in_zero_one(value, name):
    if not isinstance(value, float):
        raise TypeError(f'{name} must be of float type. '
                        f'{str(value)} of type {str(type(value))} was passed.')
    if (value < 0) | (value > 1):
        raise ValueError(f'{name} must be in [0,1]. '
                         f'{str(value)} was passed.')
    return