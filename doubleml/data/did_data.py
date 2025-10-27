import io
import warnings

import pandas as pd
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_consistent_length, column_or_1d

from doubleml.data.base_data import DoubleMLData


# TODO: Remove DoubleMLDIDData with version 0.12.0
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
        treatment variables ``d_cols``, nor instrumental variables ``z_cols``, nor time variable ``t_col``
        are used as covariates.
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
        Default is ``True``.    Examples
    --------
    >>> from doubleml import DoubleMLDIDData
    >>> from doubleml.did.datasets import make_did_SZ2020
    >>> # initialization from pandas.DataFrame
    >>> df = make_did_SZ2020(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLDIDData(df, 'y', 'd')
    >>> # initialization from np.ndarray
    >>> (x, y, d, t) = make_did_SZ2020(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLDIDData.from_arrays(x, y, d, t=t)
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        x_cols=None,
        z_cols=None,
        t_col=None,
        cluster_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        warnings.warn(
            "DoubleMLDIDData is deprecated and will be removed with version 0.12.0." "Use DoubleMLPanelData instead.",
            FutureWarning,
            stacklevel=2,
        )
        # Initialize _t_col to None first to avoid AttributeError during parent init
        self._t_col = None

        # Store whether x_cols was originally None to reset it later
        x_cols_was_none = x_cols is None

        # Call parent constructor first to set _data
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

        # Set time column directly to avoid triggering checks during init
        if t_col is not None:
            if not isinstance(t_col, str):
                raise TypeError(
                    "The time variable t_col must be of str type (or None). "
                    f"{str(t_col)} of type {str(type(t_col))} was passed."
                )
            if t_col not in self.all_variables:
                raise ValueError(f"Invalid time variable t_col. {t_col} is no data column.")
        self._t_col = t_col

        # If x_cols was originally None, reset it to exclude the time column
        if x_cols_was_none and t_col is not None:
            self.x_cols = None

        # Now run the checks and set variables
        if t_col is not None:
            self._check_disjoint_sets()
            self._set_y_z_t()

        # Set time variable array after data is loaded
        self._set_time_var()

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        z=None,
        t=None,
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
        --------
        >>> from doubleml import DoubleMLDIDData
        >>> from doubleml.did.datasets import make_did_SZ2020
        >>> (x, y, d, t) = make_did_SZ2020(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLDIDData.from_arrays(x, y, d, t=t)
        """
        # Prepare time variable

        if t is None:
            t_col = None
        else:
            t = column_or_1d(t, warn=True)
            check_consistent_length(x, y, d, t)
            t_col = "t"

        # Create base data using parent class method
        base_data = DoubleMLData.from_arrays(
            x, y, d, z, cluster_vars, use_other_treat_as_covariate, force_all_x_finite, force_all_d_finite
        )

        # Add time variable to the DataFrame
        data = pd.concat((base_data.data, pd.DataFrame(t, columns=[t_col])), axis=1)

        if t is not None:
            data[t_col] = t

        return cls(
            data,
            base_data.y_col,
            base_data.d_cols,
            base_data.x_cols,
            base_data.z_cols,
            t_col,
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
        reset_value = hasattr(self, "_t_col")
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    "The time variable t_col must be of str type (or None). "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if value not in self.all_variables:
                raise ValueError(f"Invalid time variable t_col. {value} is no data column.")
            self._t_col = value
        else:
            self._t_col = None
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_t()

    @property
    def t(self):
        """
        Array of time variable.
        """
        if self.t_col is not None:
            return self._t.values
        else:
            return None

    def _get_optional_col_sets(self):
        """Get optional column sets including time column."""
        base_optional_col_sets = super()._get_optional_col_sets()
        if self.t_col is not None:
            t_col_set = {self.t_col}
            return [t_col_set] + base_optional_col_sets
        return base_optional_col_sets

    def _check_disjoint_sets(self):
        """Check that time column doesn't overlap with other variables."""
        # Apply standard checks from parent class
        super()._check_disjoint_sets()
        if self.t_col is not None:
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
        if hasattr(self, "_data") and self.t_col in self.data.columns:
            self._t = self.data.loc[:, self.t_col]

    def _set_y_z_t(self):
        def _set_attr(col):
            if col is None:
                return None
            assert_all_finite(self.data.loc[:, col])
            return self.data.loc[:, col]

        self._y = _set_attr(self.y_col)
        self._z = _set_attr(self.z_cols)
        self._t = _set_attr(self.t_col)

    def __str__(self):
        """String representation."""
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        print("================== DoubleMLDIDData Object ==================", file=buf)
        print(f"Time variable: {self.t_col}", file=buf)
        print(data_summary, file=buf)
        return buf.getvalue()
