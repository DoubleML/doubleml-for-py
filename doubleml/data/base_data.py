import io
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d

from doubleml.utils._estimation import _assure_2d_array


class DoubleMLBaseData(ABC):
    """Base Class Double machine learning data-backends"""

    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be of pd.DataFrame type. {str(data)} of type {str(type(data))} was passed.")
        if not data.columns.is_unique:
            raise ValueError("Invalid pd.DataFrame: Contains duplicate column names.")
        self._data = data

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            "================== DoubleMLBaseData Object ==================\n"
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ DataFrame info    ------------------\n"
            + df_info
        )
        return res

    def _data_summary_str(self):
        data_summary = f"No. Observations: {self.n_obs}\n"
        return data_summary

    @property
    def data(self):
        """
        The data.
        """
        return self._data

    @property
    def all_variables(self):
        """
        All variables available in the dataset.
        """
        return self.data.columns

    @property
    def n_obs(self):
        """
        The number of observations.
        """
        return self.data.shape[0]

    # TODO: This and the following property does not make sense but the base class DoubleML needs it (especially for the
    #  multiple treatment variables case) and other things are also build around it, see for example DoubleML._params
    @property
    def d_cols(self):
        return ["theta"]

    @property
    def n_treat(self):
        """
        The number of treatment variables.
        """
        return 1

    @property
    @abstractmethod
    def n_coefs(self):
        pass


class DoubleMLData(DoubleMLBaseData):
    """Double machine learning data-backend.

    :class:`DoubleMLData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s).

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols``, nor cluster variables ``cluster_cols``
        are used as covariates.
        Default is ``None``.

    z_cols : None, str or list
        The instrumental variable(s).
        Default is ``None``.    cluster_cols : None, str or list
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

    force_all_d_finite : bool or str
        Indicates whether to raise an error on infinite values and / or missings in the treatment variables ``d``.
        Possible values are: ``True`` (neither missings ``np.nan``, ``pd.NA`` nor infinite values ``np.inf`` are
        allowed), ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        Note that the choice ``False`` and ``'allow-nan'`` are only reasonable if the model used allows for missing
        and / or infinite values in the treatment variables ``d`` (e.g. panel data models).
        Default is ``True``.

    Examples
    --------
    >>> from doubleml import DoubleMLData
    >>> from doubleml.plm.datasets import make_plr_CCDDHNR2018
    >>> # initialization from pandas.DataFrame
    >>> df = make_plr_CCDDHNR2018(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLData(df, 'y', 'd')
    >>> # initialization from np.ndarray
    >>> (x, y, d) = make_plr_CCDDHNR2018(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLData.from_arrays(x, y, d)
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        x_cols=None,
        z_cols=None,
        cluster_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        DoubleMLBaseData.__init__(self, data)

        self.y_col = y_col
        self.d_cols = d_cols
        self.z_cols = z_cols
        self.cluster_cols = cluster_cols
        self.x_cols = x_cols
        self._check_disjoint_sets()
        self.use_other_treat_as_covariate = use_other_treat_as_covariate
        self.force_all_x_finite = force_all_x_finite
        self.force_all_d_finite = force_all_d_finite
        self._binary_treats = self._check_binary_treats()
        self._binary_outcome = self._check_binary_outcome()
        self._set_y_z()
        if self.cluster_cols is not None:
            self._set_cluster_vars()
        # by default, we initialize to the first treatment variable
        self.set_x_d(self.d_cols[0])

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            "================== DoubleMLData Object ==================\n"
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
            f"Covariates: {self.x_cols}\n"
            f"Instrument variable(s): {self.z_cols}\n"
        )

        if self.cluster_cols is not None:
            data_summary += f"Cluster variable(s): {self.cluster_cols}\n"

        if hasattr(self, "is_cluster_data") and self.is_cluster_data:
            data_summary += f"Is cluster data: {self.is_cluster_data}\n"
        data_summary += f"No. Observations: {self.n_obs}\n"
        return data_summary

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        z=None,
        cluster_vars=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        """
        Initialize :class:`DoubleMLData` from :class:`numpy.ndarray`'s.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of covariates.

        y : :class:`numpy.ndarray`
            Array of the outcome variable.

        d : :class:`numpy.ndarray`
            Array of treatment variables.

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

        force_all_d_finite : bool or str
            Indicates whether to raise an error on infinite values and / or missings in the treatment variables ``d``.
            Possible values are: ``True`` (neither missings ``np.nan``, ``pd.NA`` nor infinite values ``np.inf`` are
            allowed), ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
            Note that the choice ``False`` and ``'allow-nan'`` are only reasonable if the model used allows for missing
            and / or infinite values in the treatment variables ``d`` (e.g. panel data models).
            Default is ``True``.

        Examples
        --------
        >>> from doubleml import DoubleMLData
        >>> from doubleml.plm.datasets import make_plr_CCDDHNR2018
        >>> (x, y, d) = make_plr_CCDDHNR2018(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLData.from_arrays(x, y, d)
        """
        if isinstance(force_all_x_finite, str):
            if force_all_x_finite != "allow-nan":
                raise ValueError(
                    "Invalid force_all_x_finite "
                    + force_all_x_finite
                    + ". "
                    + "force_all_x_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(force_all_x_finite, bool):
            raise TypeError("Invalid force_all_x_finite. " + "force_all_x_finite must be True, False or 'allow-nan'.")

        if isinstance(force_all_d_finite, str):
            if force_all_d_finite != "allow-nan":
                raise ValueError(
                    "Invalid force_all_d_finite "
                    + force_all_d_finite
                    + ". "
                    + "force_all_d_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(force_all_d_finite, bool):
            raise TypeError("Invalid force_all_d_finite. " + "force_all_d_finite must be True, False or 'allow-nan'.")

        x = check_array(x, ensure_2d=False, allow_nd=False, ensure_all_finite=force_all_x_finite)
        d = check_array(d, ensure_2d=False, allow_nd=False, ensure_all_finite=force_all_x_finite)
        y = column_or_1d(y, warn=True)

        x = _assure_2d_array(x)
        d = _assure_2d_array(d)

        y_col = "y"

        if z is None:
            check_consistent_length(x, y, d)
            z_cols = None
        else:
            z = check_array(z, ensure_2d=False, allow_nd=False)
            z = _assure_2d_array(z)
            check_consistent_length(x, y, d, z)
            if z.shape[1] == 1:
                z_cols = ["z"]
            else:
                z_cols = [f"z{i + 1}" for i in np.arange(z.shape[1])]

        if cluster_vars is None:
            cluster_cols = None
        else:
            cluster_vars = check_array(cluster_vars, ensure_2d=False, allow_nd=False)
            cluster_vars = _assure_2d_array(cluster_vars)
            check_consistent_length(x, y, d, cluster_vars)
            if cluster_vars.shape[1] == 1:
                cluster_cols = ["cluster_var"]
            else:
                cluster_cols = [f"cluster_var{i + 1}" for i in np.arange(cluster_vars.shape[1])]

        if d.shape[1] == 1:
            d_cols = ["d"]
        else:
            d_cols = [f"d{i + 1}" for i in np.arange(d.shape[1])]

        x_cols = [f"X{i + 1}" for i in np.arange(x.shape[1])]  # baseline version with features, outcome and treatments
        data = pd.DataFrame(np.column_stack((x, y, d)), columns=x_cols + [y_col] + d_cols)
        if z is not None:
            df_z = pd.DataFrame(z, columns=z_cols)
            data = pd.concat([data, df_z], axis=1)
        if cluster_vars is not None:
            df_cluster = pd.DataFrame(cluster_vars, columns=cluster_cols)
            data = pd.concat([data, df_cluster], axis=1)

        return cls(
            data,
            y_col,
            d_cols,
            x_cols,
            z_cols,
            cluster_cols,
            use_other_treat_as_covariate,
            force_all_x_finite,
            force_all_d_finite,
        )

    @property
    def x(self):
        """
        Array of covariates;
        Dynamic! May depend on the currently set treatment variable;
        To get an array of all covariates (independent of the currently set treatment variable)
        call ``obj.data[obj.x_cols].values``.
        """
        return self._X.values

    @property
    def y(self):
        """
        Array of outcome variable.
        """
        return self._y.values

    @property
    def d(self):
        """
        Array of treatment variable;
        Dynamic! Depends on the currently set treatment variable;
        To get an array of all treatment variables (independent of the currently set treatment variable)
        call ``obj.data[obj.d_cols].values``.
        """
        return self._d.values

    @property
    def z(self):
        """
        Array of instrumental variables.
        """
        if self.z_cols is not None:
            return self._z.values
        else:
            return None

    @property
    def cluster_cols(self):
        """
        The cluster variable(s).
        """
        return self._cluster_cols

    @cluster_cols.setter
    def cluster_cols(self, value):
        reset_value = hasattr(self, "_cluster_cols")
        if value is not None:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError(
                    "The cluster variable(s) cluster_cols must be of str or list type (or None). "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if not len(set(value)) == len(value):
                raise ValueError("Invalid cluster variable(s) cluster_cols: Contains duplicate values.")
            if not set(value).issubset(set(self.all_variables)):
                raise ValueError("Invalid cluster variable(s) cluster_cols. At least one cluster variable is no data column.")
            self._cluster_cols = value
        else:
            self._cluster_cols = None

        self._is_cluster_data = self._cluster_cols is not None

        if reset_value:
            self._check_disjoint_sets()
            if self.cluster_cols is not None:
                self._set_cluster_vars()

    @property
    def n_treat(self):
        """
        The number of treatment variables.
        """
        return len(self.d_cols)

    @property
    def n_coefs(self):
        """
        The number of coefficients to be estimated.
        """
        return self.n_treat

    @property
    def n_instr(self):
        """
        The number of instruments.
        """
        if self.z_cols is not None:
            n_instr = len(self.z_cols)
        else:
            n_instr = 0
        return n_instr

    @property
    def binary_treats(self):
        """
        Series with logical(s) indicating whether the treatment variable(s) are binary with values 0 and 1.
        """
        return self._binary_treats

    @property
    def binary_outcome(self):
        """
        Logical indicating whether the outcome variable is binary with values 0 and 1.
        """
        return self._binary_outcome

    @property
    def x_cols(self):
        """
        The covariates.
        """
        return self._x_cols

    @x_cols.setter
    def x_cols(self, value):
        reset_value = hasattr(self, "_x_cols")

        if value is not None:
            # Basic checks
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError(
                    "The covariates x_cols must be of str or list type (or None). "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if not len(set(value)) == len(value):
                raise ValueError("Invalid covariates x_cols: Contains duplicate values.")
            if not set(value).issubset(set(self.all_variables)):
                raise ValueError("Invalid covariates x_cols. At least one covariate is no data column.")
            assert set(value).issubset(set(self.all_variables))

            if not reset_value:
                self._x_cols = value
            else:
                previous_value = self._x_cols
                self._x_cols = value
                try:
                    self._check_disjoint_sets()
                except ValueError:
                    self._x_cols = previous_value
                    raise

        else:
            excluded_cols = {self.y_col} | set(self.d_cols)
            optional_col_sets = self._get_optional_col_sets()
            for optional_col_set in optional_col_sets:
                excluded_cols |= optional_col_set
            self._x_cols = [col for col in self.data.columns if col not in excluded_cols]

        if reset_value:
            self.set_x_d(self.d_cols[0])

    @property
    def d_cols(self):
        """
        The treatment variable(s).
        """
        return self._d_cols

    @d_cols.setter
    def d_cols(self, value):
        reset_value = hasattr(self, "_d_cols")

        # Basic checks
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                "The treatment variable(s) d_cols must be of str or list type. "
                f"{str(value)} of type {str(type(value))} was passed."
            )
        if not len(set(value)) == len(value):
            raise ValueError("Invalid treatment variable(s) d_cols: Contains duplicate values.")
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError("Invalid treatment variable(s) d_cols. At least one treatment variable is no data column.")

        if not reset_value:
            self._d_cols = value
        else:
            previous_value = self._d_cols
            self._d_cols = value
            try:
                self._check_disjoint_sets()
            except ValueError:
                self._d_cols = previous_value
                raise

        if reset_value:
            self.set_x_d(self.d_cols[0])

    @property
    def y_col(self):
        """
        The outcome variable.
        """
        return self._y_col

    @y_col.setter
    def y_col(self, value):
        reset_value = hasattr(self, "_y_col")

        # Basic checks
        if not isinstance(value, str):
            raise TypeError(
                f"The outcome variable y_col must be of str type. {str(value)} of type {str(type(value))} was passed."
            )
        if value not in self.all_variables:
            raise ValueError(f"Invalid outcome variable y_col. {value} is no data column.")

        if not reset_value:
            self._y_col = value
        else:
            previous_value = self._y_col
            self._y_col = value
            try:
                self._check_disjoint_sets()
            except ValueError:
                self._y_col = previous_value
                raise

        if reset_value:
            self._set_y_z()

    @property
    def z_cols(self):
        """
        The instrumental variable(s).
        """
        return self._z_cols

    @z_cols.setter
    def z_cols(self, value):
        reset_value = hasattr(self, "_z_cols")

        if value is not None:
            # Basic validation
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError(
                    "The instrumental variable(s) z_cols must be of str or list type (or None). "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if not len(set(value)) == len(value):
                raise ValueError("Invalid instrumental variable(s) z_cols: Contains duplicate values.")
            if not set(value).issubset(set(self.all_variables)):
                raise ValueError(
                    "Invalid instrumental variable(s) z_cols. At least one instrumental variable is no data column."
                )

            if not reset_value:
                self._z_cols = value
            else:
                previous_value = self._z_cols
                self._z_cols = value
                try:
                    self._check_disjoint_sets()
                except ValueError:
                    self._z_cols = previous_value
                    raise

        else:
            self._z_cols = None

        if reset_value:
            self._set_y_z()

    @property
    def n_cluster_vars(self):
        """
        The number of cluster variables.
        """
        if self.cluster_cols is not None:
            return len(self.cluster_cols)
        else:
            return 0

    @property
    def cluster_vars(self):
        """
        Array of cluster variable(s).
        """
        if self.cluster_cols is not None:
            return self._cluster_vars.values
        else:
            return None

    @property
    def use_other_treat_as_covariate(self):
        """
        Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
        """
        return self._use_other_treat_as_covariate

    @use_other_treat_as_covariate.setter
    def use_other_treat_as_covariate(self, value):
        reset_value = hasattr(self, "_use_other_treat_as_covariate")
        if not isinstance(value, bool):
            raise TypeError(f"use_other_treat_as_covariate must be True or False. Got {str(value)}.")
        self._use_other_treat_as_covariate = value
        if reset_value:
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    @property
    def force_all_x_finite(self):
        """
        Indicates whether to raise an error on infinite values and / or missings in the covariates ``x``.
        """
        return self._force_all_x_finite

    @force_all_x_finite.setter
    def force_all_x_finite(self, value):
        reset_value = hasattr(self, "_force_all_x_finite")
        if isinstance(value, str):
            if value != "allow-nan":
                raise ValueError(
                    "Invalid force_all_x_finite " + value + ". " + "force_all_x_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(value, bool):
            raise TypeError("Invalid force_all_x_finite. " + "force_all_x_finite must be True, False or 'allow-nan'.")
        self._force_all_x_finite = value
        if reset_value:
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    @property
    def force_all_d_finite(self):
        """
        Indicates whether to raise an error on infinite values and / or missings in the treatment variables ``d``.
        Possible values are: ``True`` (neither missings ``np.nan``, ``pd.NA`` nor infinite values ``np.inf`` are
        allowed), ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        """
        return self._force_all_d_finite

    @force_all_d_finite.setter
    def force_all_d_finite(self, value):
        reset_value = hasattr(self, "_force_all_d_finite")
        if isinstance(value, str):
            if value != "allow-nan":
                raise ValueError(
                    "Invalid force_all_d_finite " + value + ". " + "force_all_d_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(value, bool):
            raise TypeError("Invalid force_all_d_finite. " + "force_all_d_finite must be True, False or 'allow-nan'.")
        self._force_all_d_finite = value
        if reset_value:
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    def _set_y_z(self):
        def _set_attr(col):
            if col is None:
                return None
            if isinstance(col, list):
                converted_data = self.data.loc[:, col].apply(pd.to_numeric, errors="raise")
            else:
                converted_data = pd.to_numeric(self.data.loc[:, col], errors="raise")
            assert_all_finite(converted_data)
            return converted_data

        self._y = _set_attr(self.y_col)
        self._z = _set_attr(self.z_cols)

    def _set_cluster_vars(self):
        """Set cluster variables."""
        if self.cluster_cols is not None:
            assert_all_finite(self.data.loc[:, self.cluster_cols])
            self._cluster_vars = self.data.loc[:, self.cluster_cols]

    def set_x_d(self, treatment_var):
        """
        Function that assigns the role for the treatment variables in the multiple-treatment case.

        Parameters
        ----------
        treatment_var : str
            Active treatment variable that will be set to d.
        """
        if not isinstance(treatment_var, str):
            raise TypeError(
                f"treatment_var must be of str type. {str(treatment_var)} of type {str(type(treatment_var))} was passed."
            )
        if treatment_var not in self.d_cols:
            raise ValueError(f"Invalid treatment_var. {treatment_var} is not in d_cols.")
        if self.use_other_treat_as_covariate:
            # note that the following line needs to be adapted in case an intersection of x_cols and d_cols as allowed
            # (see https://github.com/DoubleML/doubleml-for-py/issues/83)
            xd_list = self.x_cols + self.d_cols
            xd_list.remove(treatment_var)
        else:
            xd_list = self.x_cols
        if self.force_all_d_finite:
            assert_all_finite(self.data.loc[:, self.d_cols], allow_nan=self.force_all_d_finite == "allow-nan")
        if self.force_all_x_finite:
            assert_all_finite(self.data.loc[:, xd_list], allow_nan=self.force_all_x_finite == "allow-nan")

        treatment_data = self.data.loc[:, treatment_var]
        # For panel data, preserve datetime type for treatment variables
        if pd.api.types.is_datetime64_any_dtype(treatment_data):
            self._d = treatment_data
        else:
            self._d = pd.to_numeric(treatment_data, errors="raise")
        self._X = self.data.loc[:, xd_list]

    def _get_optional_col_sets(self):
        # this function can be extended in inherited subclasses
        z_cols_set = set(self.z_cols or [])
        cluster_cols_set = set(self.cluster_cols or [])
        return [cluster_cols_set, z_cols_set]

    def _check_binary_treats(self):
        is_binary = pd.Series(dtype=bool, index=self.d_cols)
        if not self.force_all_d_finite:
            is_binary[:] = False  # if we allow infinite values, we cannot check for binary
        else:
            for treatment_var in self.d_cols:
                this_d = self.data.loc[:, treatment_var]
                binary_treat = type_of_target(this_d) == "binary"
                zero_one_treat = np.all((np.power(this_d, 2) - this_d) == 0)
                is_binary[treatment_var] = binary_treat & zero_one_treat
        return is_binary

    def _check_binary_outcome(self):
        y = self.data.loc[:, self.y_col]
        binary_outcome = type_of_target(y) == "binary"
        zero_one_outcome = np.all((np.power(y, 2) - y) == 0)
        is_binary = binary_outcome & zero_one_outcome
        return is_binary

    @staticmethod
    def _check_disjoint(set1, set2, name1, arg1, name2, arg2):
        """Helper method to check for disjoint sets."""
        if not set1.isdisjoint(set2):
            raise ValueError(f"At least one variable/column is set as {name1} ({arg1}) and {name2} ({arg2}).")

    def _check_disjoint_sets(self):
        # this function can be extended in inherited subclasses
        self._check_disjoint_sets_y_d_x()
        self._check_disjoint_sets_z()
        if self.cluster_cols is not None:
            self._check_disjoint_sets_cluster_cols()

    def _check_disjoint_sets_y_d_x(self):
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        if not y_col_set.isdisjoint(x_cols_set):
            raise ValueError(f"{str(self.y_col)} cannot be set as outcome variable ``y_col`` and covariate in ``x_cols``.")
        if not y_col_set.isdisjoint(d_cols_set):
            raise ValueError(
                f"{str(self.y_col)} cannot be set as outcome variable ``y_col`` and treatment variable in ``d_cols``."
            )
        # note that the line xd_list = self.x_cols + self.d_cols in method set_x_d needs adaption if an intersection of
        # x_cols and d_cols as allowed (see https://github.com/DoubleML/doubleml-for-py/issues/83)
        if not d_cols_set.isdisjoint(x_cols_set):
            raise ValueError(
                "At least one variable/column is set as treatment variable (``d_cols``) and as covariate"
                "(``x_cols``). Consider using parameter ``use_other_treat_as_covariate``."
            )

    def _check_disjoint_sets_z(self):
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        z_cols_set = set(self.z_cols or [])

        instrument_checks_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
        ]
        for set1, name, argument in instrument_checks_args:
            self._check_disjoint(
                set1=set1, name1=name, arg1=argument, set2=z_cols_set, name2="instrumental variable", arg2="``z_cols``"
            )

    def _check_disjoint_sets_cluster_cols(self):
        """Check that cluster columns are disjoint from other variable sets."""
        cluster_cols_set = set(self.cluster_cols)
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)
        z_cols_set = set(self.z_cols or [])
        checks = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
        ]
        for set1, name, arg in checks:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=arg,
                set2=cluster_cols_set,
                name2="cluster variable(s)",
                arg2="``cluster_cols``",
            )

    @property
    def is_cluster_data(self):
        """
        Flag indicating whether this data object is being used for cluster data.
        """
        return self._is_cluster_data
