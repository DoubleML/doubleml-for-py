"""
Helper function for partially linear IV data generation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def _g(x):
    return np.power(np.sin(x), 2)


def _m(x, nu=0.0, gamma=1.0):
    return 0.5 / np.pi * (np.sinh(gamma)) / (np.cosh(gamma) - np.cos(x - nu))


def _make_pliv_data(n_obs=100, dim_x=20, theta=0.5, gamma_z=0.4, return_type="DoubleMLData"):
    b = [1 / k for k in range(1, dim_x + 1)]
    sigma = make_spd_matrix(dim_x)

    x = np.random.multivariate_normal(
        np.zeros(dim_x),
        sigma,
        size=[
            n_obs,
        ],
    )
    G = _g(np.dot(x, b))
    # instrument
    z = _m(np.dot(x, b)) + np.random.standard_normal(
        size=[
            n_obs,
        ]
    )
    # treatment
    M = _m(gamma_z * z + np.dot(x, b))
    d = M + np.random.standard_normal(
        size=[
            n_obs,
        ]
    )
    y = (
        np.dot(theta, d)
        + G
        + np.random.standard_normal(
            size=[
                n_obs,
            ]
        )
    )

    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)), columns=x_cols + ["y", "d", "z"])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, "y", "d", x_cols, "z")
    else:
        raise ValueError("Invalid return_type.")
