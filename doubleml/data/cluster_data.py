import io

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_array

from doubleml.data.base_data import DoubleMLBaseData, DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLClusterData(DoubleMLData):
    """Double machine learning data-backend for data with cluster variables.

    :class:`DoubleMLClusterData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s).

    cluster_cols : str or list
        The cluster variable(s).

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols`` are used as covariates.
        Default is ``None``.

    z_cols : None, str or list
        The instrumental variable(s).
        Default is ``None``.

    t_col : None or str
        The time variable (only relevant/used for DiD Estimators).
        Default is ``None``.

    s_col : None or str
        The score or selection variable (only relevant/used for RDD and SSM Estimatiors).
        Default is ``None``.

    use_other_treat_as_covariate : bool
        Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
        Default is ``True``.

    force_all_x_finite : bool or str
        Indicates whether to raise an error on infinite values and / or missings in the covariates ``x``.
        Possible values are: ``True`` (neither missings ``np.nan``, ``pd.NA`` nor infinite values ``np.inf`` are
        allowed), ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        Note that the choice ``False`` and ``'allow-nan'`` are only reasonable if the machine learning methods used
        for the nuisance functions are capable to provide valid predictions with missings and / or infinite values
        in the covariates ``x``.
        Default is ``True``.

    Examples
    --------
    >>> from doubleml import DoubleMLClusterData
    >>> from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021
    >>> # initialization from pandas.DataFrame
    >>> df = make_pliv_multiway_cluster_CKMS2021(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLClusterData(df, 'Y', 'D', ['cluster_var_i', 'cluster_var_j'], z_cols='Z')
    >>> # initialization from np.ndarray
    >>> (x, y, d, cluster_vars, z) = make_pliv_multiway_cluster_CKMS2021(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLClusterData.from_arrays(x, y, d, cluster_vars, z)
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
        DoubleMLBaseData.__init__(self, data)

        # we need to set cluster_cols (needs _data) before call to the super __init__ because of the x_cols setter
        self.cluster_cols = cluster_cols
        self._set_cluster_vars()
        DoubleMLData.__init__(
            self, data, y_col, d_cols, x_cols, z_cols, t_col, s_col, use_other_treat_as_covariate, force_all_x_finite
        )
        self._check_disjoint_sets_cluster_cols()

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            "================== DoubleMLClusterData Object ==================\n"
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ DataFrame info    ------------------\n"
            + df_info
        )
        return res

    def _data_summary_str(self):
        data_summary = (
            f"Outcome variable: {self.y_col}\n"
            f"Treatment variable(s): {self.d_cols}\n"
            f"Cluster variable(s): {self.cluster_cols}\n"
            f"Covariates: {self.x_cols}\n"
            f"Instrument variable(s): {self.z_cols}\n"
        )
        if self.t_col is not None:
            data_summary += f"Time variable: {self.t_col}\n"
        if self.s_col is not None:
            data_summary += f"Score/Selection variable: {self.s_col}\n"

        data_summary += f"No. Observations: {self.n_obs}\n"
        return data_summary

    @classmethod
    def from_arrays(
        cls, x, y, d, cluster_vars, z=None, t=None, s=None, use_other_treat_as_covariate=True, force_all_x_finite=True
    ):
        """
        Initialize :class:`DoubleMLClusterData` from :class:`numpy.ndarray`'s.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of covariates.

        y : :class:`numpy.ndarray`
            Array of the outcome variable.

        d : :class:`numpy.ndarray`
            Array of treatment variables.

        cluster_vars : :class:`numpy.ndarray`
            Array of cluster variables.

        z : None or :class:`numpy.ndarray`
            Array of instrumental variables.
            Default is ``None``.

        t : :class:`numpy.ndarray`
            Array of the time variable (only relevant/used for DiD models).
            Default is ``None``.

        s : :class:`numpy.ndarray`
            Array of the score or selection variable (only relevant/used for RDD or SSM models).
            Default is ``None``.

        use_other_treat_as_covariate : bool
            Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
            Default is ``True``.

        force_all_x_finite : bool or str
            Indicates whether to raise an error on infinite values and / or missings in the covariates ``x``.
            Possible values are: ``True`` (neither missings ``np.nan``, ``pd.NA`` nor infinite values ``np.inf`` are
            allowed), ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
            Note that the choice ``False`` and ``'allow-nan'`` are only reasonable if the machine learning methods used
            for the nuisance functions are capable to provide valid predictions with missings and / or infinite values
            in the covariates ``x``.
            Default is ``True``.

        Examples
        --------
        >>> from doubleml import DoubleMLClusterData
        >>> from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021
        >>> (x, y, d, cluster_vars, z) = make_pliv_multiway_cluster_CKMS2021(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLClusterData.from_arrays(x, y, d, cluster_vars, z)
        """
        dml_data = DoubleMLData.from_arrays(x, y, d, z, t, s, use_other_treat_as_covariate, force_all_x_finite)
        cluster_vars = check_array(cluster_vars, ensure_2d=False, allow_nd=False)
        cluster_vars = _assure_2d_array(cluster_vars)
        if cluster_vars.shape[1] == 1:
            cluster_cols = ["cluster_var"]
        else:
            cluster_cols = [f"cluster_var{i + 1}" for i in np.arange(cluster_vars.shape[1])]

        data = pd.concat((pd.DataFrame(cluster_vars, columns=cluster_cols), dml_data.data), axis=1)

        return cls(
            data,
            dml_data.y_col,
            dml_data.d_cols,
            cluster_cols,
            dml_data.x_cols,
            dml_data.z_cols,
            dml_data.t_col,
            dml_data.s_col,
            dml_data.use_other_treat_as_covariate,
            dml_data.force_all_x_finite,
        )

    @property
    def cluster_cols(self):
        """
        The cluster variable(s).
        """
        return self._cluster_cols

    @cluster_cols.setter
    def cluster_cols(self, value):
        reset_value = hasattr(self, "_cluster_cols")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                "The cluster variable(s) cluster_cols must be of str or list type. "
                f"{str(value)} of type {str(type(value))} was passed."
            )
        if not len(set(value)) == len(value):
            raise ValueError("Invalid cluster variable(s) cluster_cols: Contains duplicate values.")
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError("Invalid cluster variable(s) cluster_cols. At least one cluster variable is no data column.")
        self._cluster_cols = value
        if reset_value:
            self._check_disjoint_sets()
            self._set_cluster_vars()

    @property
    def n_cluster_vars(self):
        """
        The number of cluster variables.
        """
        return len(self.cluster_cols)

    @property
    def cluster_vars(self):
        """
        Array of cluster variable(s).
        """
        return self._cluster_vars.values

    def _get_optional_col_sets(self):
        base_optional_col_sets = super()._get_optional_col_sets()
        cluster_cols_set = set(self.cluster_cols)
        return [cluster_cols_set] + base_optional_col_sets

    def _check_disjoint_sets(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLClusterData, self)._check_disjoint_sets()
        self._check_disjoint_sets_cluster_cols()

    def _check_disjoint_sets_cluster_cols(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLClusterData, self)._check_disjoint_sets()

        # special checks for the additional cluster variables
        cluster_cols_set = set(self.cluster_cols)
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        z_cols_set = set(self.z_cols or [])
        t_col_set = {self.t_col} if self.t_col else set()
        s_col_set = {self.s_col} if self.s_col else set()

        # TODO: X can not be used as cluster variable
        cluster_checks_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
            (t_col_set, "time variable", "``t_col``"),
            (s_col_set, "score or selection variable", "``s_col``"),
        ]
        for set1, name, argument in cluster_checks_args:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=argument,
                set2=cluster_cols_set,
                name2="cluster variable(s)",
                arg2="``cluster_cols``",
            )

    def _set_cluster_vars(self):
        assert_all_finite(self.data.loc[:, self.cluster_cols])
        self._cluster_vars = self.data.loc[:, self.cluster_cols]
