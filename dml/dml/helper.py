import numpy as np
from sklearn.utils.multiclass import type_of_target


def assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x


def check_binary_vector(x, variable_name=''):
    # assure D binary
    assert type_of_target(x) == 'binary', 'variable ' + variable_name  + ' must be binary'
    
    if np.any(np.power(x,2) - x != 0):
        raise ValueError('variable ' + variable_name  + ' must be binary with values 0 and 1')

