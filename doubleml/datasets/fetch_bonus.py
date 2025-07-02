"""
Data set on the Pennsylvania Reemployment Bonus experiment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from doubleml import DoubleMLData


def _get_array_alias():
    return ["array", "np.array", "np.ndarray"]


def _get_data_frame_alias():
    return ["DataFrame", "pd.DataFrame", "pandas.DataFrame"]


def _get_dml_data_alias():
    return ["DoubleMLData"]


def fetch_bonus(return_type="DoubleMLData", polynomial_features=False):
    """
    Data set on the Pennsylvania Reemployment Bonus experiment.

    Parameters
    ----------
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.
    polynomial_features :
        If ``True`` polynomial features are added (see replication files of Chernozhukov et al. (2018)).

    References
    ----------
    Bilias Y. (2000), Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment.
    Journal of Applied Econometrics, 15(6): 575-594.

    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    _data_frame_alias = _get_data_frame_alias()
    _dml_data_alias = _get_dml_data_alias()

    url = "https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat"
    raw_data = pd.read_csv(url, sep=r"\s+")

    ind = (raw_data["tg"] == 0) | (raw_data["tg"] == 4)
    data = raw_data.copy()[ind]
    data.reset_index(inplace=True)
    data["tg"] = data["tg"].replace(4, 1)
    data["inuidur1"] = np.log(data["inuidur1"])

    # variable dep as factor (dummy encoding)
    dummy_enc = OneHotEncoder(drop="first", categories="auto").fit(data.loc[:, ["dep"]])
    xx = dummy_enc.transform(data.loc[:, ["dep"]]).toarray()
    data["dep1"] = xx[:, 0]
    data["dep2"] = xx[:, 1]

    y_col = "inuidur1"
    d_cols = ["tg"]
    x_cols = [
        "female",
        "black",
        "othrace",
        "dep1",
        "dep2",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "agelt35",
        "agegt54",
        "durable",
        "lusd",
        "husd",
    ]

    if polynomial_features:
        poly = PolynomialFeatures(2, include_bias=False)
        data_transf = poly.fit_transform(data[x_cols])
        x_cols = list(poly.get_feature_names_out(x_cols))

        data_transf = pd.DataFrame(data_transf, columns=x_cols)
        data = pd.concat((data[[y_col] + d_cols], data_transf), axis=1, sort=False)

    if return_type in _data_frame_alias + _dml_data_alias:
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, y_col, d_cols, x_cols)
    else:
        raise ValueError("Invalid return_type.")
