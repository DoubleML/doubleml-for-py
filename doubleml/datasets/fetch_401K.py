"""
Data set on financial wealth and 401(k) plan participation.
"""

import pandas as pd

from doubleml import DoubleMLData


def _get_array_alias():
    return ["array", "np.array", "np.ndarray"]


def _get_data_frame_alias():
    return ["DataFrame", "pd.DataFrame", "pandas.DataFrame"]


def _get_dml_data_alias():
    return ["DoubleMLData"]


def fetch_401K(return_type="DoubleMLData", polynomial_features=False):
    """
    Data set on financial wealth and 401(k) plan participation.

    Parameters
    ----------
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.
    polynomial_features :
        If ``True`` polynomial features are added (see replication files of Chernozhukov et al. (2018)).

    References
    ----------
    Abadie, A. (2003), Semiparametric instrumental variable estimation of treatment response models. Journal of
    Econometrics, 113(2): 231-263.

    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    _data_frame_alias = _get_data_frame_alias()
    _dml_data_alias = _get_dml_data_alias()

    url = "https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta"
    raw_data = pd.read_stata(url)

    y_col = "net_tfa"
    d_cols = ["e401"]
    x_cols = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]

    data = raw_data.copy()

    if polynomial_features:
        raise NotImplementedError("polynomial_features os not implemented yet for fetch_401K.")

    if return_type in _data_frame_alias + _dml_data_alias:
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, y_col, d_cols, x_cols)
    else:
        raise ValueError("Invalid return_type.")
