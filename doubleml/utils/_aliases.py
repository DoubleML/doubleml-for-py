import numpy as np
import pandas as pd

from doubleml.data import (
    DoubleMLClusterData,
    DoubleMLData,
    DoubleMLDIDData,
    DoubleMLPanelData,
    DoubleMLRDDData,
    DoubleMLSSMData,
)

_array_alias = ["array", "np.ndarray", "np.array", np.ndarray]
_data_frame_alias = ["DataFrame", "pd.DataFrame", pd.DataFrame]
_dml_data_alias = ["DoubleMLData", DoubleMLData]
_dml_did_data_alias = ["DoubleMLDIDData", DoubleMLDIDData]
_dml_panel_data_alias = ["DoubleMLPanelData", DoubleMLPanelData]
_dml_rdd_data_alias = ["DoubleMLRDDData", DoubleMLRDDData]
_dml_ssm_data_alias = ["DoubleMLSSMData", DoubleMLSSMData]
_dml_cluster_data_alias = ["DoubleMLClusterData", DoubleMLClusterData]


def _get_array_alias():
    """Returns the list of array aliases."""
    return _array_alias


def _get_data_frame_alias():
    """Returns the list of data frame aliases."""
    return _data_frame_alias


def _get_dml_data_alias():
    """Returns the list of DoubleMLData aliases."""
    return _dml_data_alias


def _get_dml_cluster_data_alias():
    """Returns the list of DoubleMLClusterData aliases."""
    return _dml_cluster_data_alias


def _get_dml_did_data_alias():
    """Returns the list of DoubleMLDIDData aliases."""
    return _dml_did_data_alias


def _get_dml_panel_data_alias():
    """Returns the list of DoubleMLPanelData aliases."""
    return _dml_panel_data_alias


def _get_dml_rdd_data_alias():
    """Returns the list of DoubleMLRDDData aliases."""
    return _dml_rdd_data_alias


def _get_dml_ssm_data_alias():
    """Returns the list of DoubleMLSSMData aliases."""
    return _dml_ssm_data_alias
