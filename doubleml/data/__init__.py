"""
The :mod:`doubleml.data` module implements data classes for double machine learning.
"""

import warnings

from .base_data import DoubleMLData
from .did_data import DoubleMLDIDData
from .panel_data import DoubleMLPanelData
from .rdd_data import DoubleMLRDDData
from .ssm_data import DoubleMLSSMData


class DoubleMLClusterData(DoubleMLData):
    """
    Backwards compatibility wrapper for DoubleMLData with cluster_cols.
    This class is deprecated and will be removed in a future version.
    Use DoubleMLData with cluster_cols instead.
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        cluster_cols,
        x_cols=None,
        z_cols=None,
        t_col=None,
        s_col=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
    ):
        warnings.warn(
            "DoubleMLClusterData is deprecated and will be removed with version 0.12.0. "
            "Use DoubleMLData with cluster_cols instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            data=data,
            y_col=y_col,
            d_cols=d_cols,
            x_cols=x_cols,
            z_cols=z_cols,
            cluster_cols=cluster_cols,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=True,
        )

    @classmethod
    def from_arrays(
        cls, x, y, d, cluster_vars, z=None, t=None, s=None, use_other_treat_as_covariate=True, force_all_x_finite=True
    ):
        """
        Initialize :class:`DoubleMLClusterData` from :class:`numpy.ndarray`'s.
        This method is deprecated and will be removed with version 0.12.0,
        use DoubleMLData.from_arrays with cluster_vars instead.
        """
        warnings.warn(
            "DoubleMLClusterData is deprecated and will be removed with version 0.12.0. "
            "Use DoubleMLData.from_arrays with cluster_vars instead.",
            FutureWarning,
            stacklevel=2,
        )
        return DoubleMLData.from_arrays(
            x=x,
            y=y,
            d=d,
            z=z,
            cluster_vars=cluster_vars,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=True,
        )


__all__ = ["DoubleMLData", "DoubleMLClusterData", "DoubleMLDIDData", "DoubleMLPanelData", "DoubleMLRDDData", "DoubleMLSSMData"]
