import io
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

from doubleml.data.base_data import DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLDIDData(DoubleMLData):
    """Double machine learning data-backend for Difference-in-Differences models.

    :class:`DoubleMLDIDData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s).

    t_col : str
        The time variable for DiD models.

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols``, nor time variable ``t_col`` are used as covariates.
        Default is ``None``.

    z_cols : None, str or list
        The instrumental variable(s).
        Default is ``None``.

    cluster_cols : None, str or list
        The cluster variable(s).
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

    force_all_d_finite : bool
        Indicates whether to raise an error on infinite values and / or missings in the treatment variables ``d``.
        Default is ``True``.

    Examples
    --------    >>> from doubleml import DoubleMLDIDData
    >>> from doubleml.did.datasets import make_did_SZ2020
    >>> # initialization from pandas.DataFrame
    >>> df = make_did_SZ2020(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLDIDData(df, 'y', 'd', 't')
    >>> # initialization from np.ndarray
    >>> (x, y, d, t) = make_did_SZ2020(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLDIDData.from_arrays(x, y, d, t=t)
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        t_col,
        x_cols=None,
        z_cols=None,
        cluster_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        # Set time column before calling parent constructor
        self.t_col = t_col
        
        # Call parent constructor
        super().__init__(
            data=data,
            y_col=y_col,
            d_cols=d_cols,
            x_cols=x_cols,
            z_cols=z_cols,
            cluster_cols=cluster_cols,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=force_all_d_finite,
        )
        
        # Set time variable array after data is loaded
        self._set_time_var()

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        t,
        z=None,
        cluster_vars=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        """
        Initialize :class:`DoubleMLDIDData` object from :class:`numpy.ndarray`'s.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of covariates.

        y : :class:`numpy.ndarray`
            Array of the outcome variable.

        d : :class:`numpy.ndarray`
            Array of treatment variables.

        t : :class:`numpy.ndarray`
            Array of the time variable for DiD models.

        z : None or :class:`numpy.ndarray`
            Array of instrumental variables.
            Default is ``None``.

        cluster_vars : None or :class:`numpy.ndarray`
            Array of cluster variables.
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

        force_all_d_finite : bool
            Indicates whether to raise an error on infinite values and / or missings in the treatment variables ``d``.
            Default is ``True``.

        Examples
        --------        >>> from doubleml import DoubleMLDIDData
        >>> from doubleml.did.datasets import make_did_SZ2020
        >>> (x, y, d, t) = make_did_SZ2020(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLDIDData.from_arrays(x, y, d, t=t)
        """
        # Prepare time variable
        t = check_array(t, ensure_2d=False, allow_nd=False)
        t = _assure_2d_array(t)
        if t.shape[1] != 1:
            raise ValueError("t must be a single column.")
        t_col = "t"
        
        # Create base data using parent class method
        base_data = DoubleMLData.from_arrays(
            x, y, d, z, cluster_vars, use_other_treat_as_covariate, force_all_x_finite, force_all_d_finite
        )
        
        # Add time variable to the DataFrame
        data = pd.concat((base_data.data, pd.DataFrame(t, columns=[t_col])), axis=1)
        
        return cls(
            data,
            base_data.y_col,
            base_data.d_cols,
            t_col,
            base_data.x_cols,
            base_data.z_cols,
            base_data.cluster_cols,
            base_data.use_other_treat_as_covariate,
            base_data.force_all_x_finite,
            base_data.force_all_d_finite,
        )

    @property
    def t_col(self):
        """
        The time variable.
        """
        return self._t_col

    @t_col.setter
    def t_col(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "The time variable t_col must be of str type. "
                f"{str(value)} of type {str(type(value))} was passed."
            )
        # Check if data exists (during initialization it might not)
        if hasattr(self, '_data') and value not in self.all_variables:
            raise ValueError("Invalid time variable t_col. The time variable is no data column.")
        self._t_col = value
        # Update time variable array if data is already loaded
        if hasattr(self, '_data'):
            self._set_time_var()

    @property
    def t(self):
        """
        Array of time variable.
        """
        return self._t.values

    def _get_optional_col_sets(self):
        """Get optional column sets including time column."""
        base_optional_col_sets = super()._get_optional_col_sets()
        t_col_set = {self.t_col}
        return [t_col_set] + base_optional_col_sets

    def _check_disjoint_sets(self):
        """Check that time column doesn't overlap with other variables."""
        # Apply standard checks from parent class
        super()._check_disjoint_sets()
        self._check_disjoint_sets_t_col()

    def _check_disjoint_sets_t_col(self):
        """Check that time column is disjoint from other variable sets."""
        t_col_set = {self.t_col}
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)
        z_cols_set = set(self.z_cols or [])
        cluster_cols_set = set(self.cluster_cols or [])

        t_checks_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
            (cluster_cols_set, "cluster variable(s)", "``cluster_cols``"),
        ]
        for set1, name, argument in t_checks_args:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=argument,
                set2=t_col_set,
                name2="time variable",
                arg2="``t_col``",
            )

    def _set_time_var(self):
        """Set the time variable array."""
        if hasattr(self, '_data') and self.t_col in self.data.columns:
            self._t = self.data.loc[:, [self.t_col]]

    def __str__(self):
        """String representation."""
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        print("================== DoubleMLDIDData Object ==================", file=buf)
        print(f"Time variable: {self.t_col}", file=buf)
        print(data_summary, file=buf)
        return buf.getvalue()
