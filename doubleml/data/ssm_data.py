import io

import pandas as pd
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_array

from doubleml.data.base_data import DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLSSMData(DoubleMLData):
    """Double machine learning data-backend for Sample Selection Models.

    :class:`DoubleMLSSMData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s).

    s_col : str
        The selection variable for SSM models.

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols``, nor selection variable ``s_col``
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
        Default is ``True``.

    Examples
    --------
    >>> from doubleml import DoubleMLSSMData
    >>> from doubleml.irm.datasets import make_ssm_data
    >>> # initialization from pandas.DataFrame
    >>> df = make_ssm_data(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLSSMData(df, 'y', 'd', s_col='s')
    >>> # initialization from np.ndarray
    >>> (x, y, d, _, s) = make_ssm_data(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLSSMData.from_arrays(x, y, d, s=s)
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        x_cols=None,
        z_cols=None,
        s_col=None,
        cluster_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        # Initialize _s_col to None first to avoid AttributeError during parent init
        self._s_col = None

        # Store whether x_cols was originally None to reset it later
        x_cols_was_none = x_cols is None

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

        # Set selection column directly to avoid triggering checks during init
        if s_col is not None:
            if not isinstance(s_col, str):
                raise TypeError(
                    "The selection variable s_col must be of str type (or None). "
                    f"{str(s_col)} of type {str(type(s_col))} was passed."
                )
            if s_col not in self.all_variables:
                raise ValueError(f"Invalid selection variable s_col. {s_col} is no data column.")
        self._s_col = s_col

        # If x_cols was originally None, reset it to exclude the selection column
        if x_cols_was_none and s_col is not None:
            self.x_cols = None

        # Now run the checks and set variables
        if s_col is not None:
            self._check_disjoint_sets()
            self._set_y_z_s()

        # Set selection variable array after data is loaded
        self._set_selection_var()

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        z=None,
        s=None,
        cluster_vars=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        """
        Initialize :class:`DoubleMLSSMData` object from :class:`numpy.ndarray`'s.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of covariates.

        y : :class:`numpy.ndarray`
            Array of the outcome variable.

        d : :class:`numpy.ndarray`
            Array of treatment variables.

        s : :class:`numpy.ndarray`
            Array of the selection variable for SSM models.

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
        >>> from doubleml import DoubleMLSSMData
        >>> from doubleml.irm.datasets import make_ssm_data
        >>> (x, y, d, _, s) = make_ssm_data(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLSSMData.from_arrays(x, y, d, s=s)
        """
        # Prepare selection variable
        s = check_array(s, ensure_2d=False, allow_nd=False)
        s = _assure_2d_array(s)
        if s.shape[1] != 1:
            raise ValueError("s must be a single column.")
        s_col = "s"

        # Create base data using parent class method
        base_data = DoubleMLData.from_arrays(
            x, y, d, z, cluster_vars, use_other_treat_as_covariate, force_all_x_finite, force_all_d_finite
        )

        # Add selection variable to the DataFrame
        data = pd.concat((base_data.data, pd.DataFrame(s, columns=[s_col])), axis=1)

        return cls(
            data,
            base_data.y_col,
            base_data.d_cols,
            base_data.x_cols,
            base_data.z_cols,
            s_col,
            base_data.cluster_cols,
            base_data.use_other_treat_as_covariate,
            base_data.force_all_x_finite,
            base_data.force_all_d_finite,
        )

    @property
    def s(self):
        """
        Array of score or selection variable.
        """
        if self.s_col is not None:
            return self._s.values
        else:
            return None

    @property
    def s_col(self):
        """
        The selection variable.
        """
        return self._s_col

    @s_col.setter
    def s_col(self, value):
        reset_value = hasattr(self, "_s_col")
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    "The selection variable s_col must be of str type (or None). "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if value not in self.all_variables:
                raise ValueError(f"Invalid selection variable s_col. {value} is no data column.")
            self._s_col = value
        else:
            self._s_col = None
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_s()

    def _get_optional_col_sets(self):
        """Get optional column sets including selection column."""
        base_optional_col_sets = super()._get_optional_col_sets()
        if self.s_col is not None:
            s_col_set = {self.s_col}
            return [s_col_set] + base_optional_col_sets
        return base_optional_col_sets

    def _check_disjoint_sets(self):
        """Check that selection column doesn't overlap with other variables."""
        # Apply standard checks from parent class
        super()._check_disjoint_sets()
        self._check_disjoint_sets_s_col()

    def _check_disjoint_sets_s_col(self):
        """Check that selection column is disjoint from other variable sets."""
        s_col_set = {self.s_col}
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)
        z_cols_set = set(self.z_cols or [])
        cluster_cols_set = set(self.cluster_cols or [])

        s_checks_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
            (cluster_cols_set, "cluster variable(s)", "``cluster_cols``"),
        ]
        for set1, name, argument in s_checks_args:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=argument,
                set2=s_col_set,
                name2="selection variable",
                arg2="``s_col``",
            )

    def _set_selection_var(self):
        """Set the selection variable array."""
        if hasattr(self, "_data") and self.s_col in self.data.columns:
            self._s = self.data.loc[:, [self.s_col]].squeeze()

    def _set_y_z_s(self):
        def _set_attr(col):
            if col is None:
                return None
            assert_all_finite(self.data.loc[:, col])
            return self.data.loc[:, col]

        self._y = _set_attr(self.y_col)
        self._z = _set_attr(self.z_cols)
        self._s = _set_attr(self.s_col)

    def __str__(self):
        """String representation."""
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        print("================== DoubleMLSSMData Object ==================", file=buf)
        print(f"Selection variable: {self.s_col}", file=buf)
        print(data_summary, file=buf)
        return buf.getvalue()
