import io

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite

from doubleml.data.base_data import DoubleMLBaseData, DoubleMLData
from doubleml.data.utils.panel_data_utils import _is_valid_datetime_unit


class DoubleMLPanelData(DoubleMLData):
    """Double machine learning data-backend for panel data in long format.

    :class:`DoubleMLPanelData` objects can be initialized from
    :class:`pandas.DataFrame` as well as :class:`numpy.ndarray` objects.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s) indicating the treatment groups in terms of first time of treatment exposure.

    t_col : str
        The time variable indicating the time.

    id_col : str
        Unique unit identifier.

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols`` are used as covariates.
        Default is ``None``.

    z_cols : None, str or list
        The instrumental variable(s).
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

    datetime_unit : str
        The unit of the time and treatment variable (if datetime type).

    Examples
    --------
    >>> from doubleml.did.datasets import make_did_CS2021
    >>> from doubleml import DoubleMLPanelData
    >>> df = make_did_CS2021(n_obs=500)
    >>> dml_data = DoubleMLPanelData(
    ...     df,
    ...     y_col="y",
    ...     d_cols="d",
    ...     id_col="id",
    ...     t_col="t",
    ...     x_cols=["Z1", "Z2", "Z3", "Z4"],
    ...     datetime_unit="M"
    ... )
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        t_col,
        id_col,
        x_cols=None,
        z_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        datetime_unit="M",
    ):
        DoubleMLBaseData.__init__(self, data)

        # we need to set id_col (needs _data) before call to the super __init__ because of the x_cols setter
        self.id_col = id_col
        self._datetime_unit = _is_valid_datetime_unit(datetime_unit)
        self._set_id_var()

        DoubleMLData.__init__(
            self,
            data=data,
            y_col=y_col,
            d_cols=d_cols,
            x_cols=x_cols,
            z_cols=z_cols,
            t_col=t_col,
            s_col=None,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=False,
        )
        if self.n_treat != 1:
            raise ValueError("Only one treatment column is allowed for panel data.")

        self._check_disjoint_sets_id_col()

        # intialize the unique values of g and t
        self._g_values = np.sort(np.unique(self.d))  # unique values of g
        self._t_values = np.sort(np.unique(self.t))  # unique values of t

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            "================== DoubleMLPanelData Object ==================\n"
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
            f"Time variable: {self.t_col}\n"
            f"Id variable: {self.id_col}\n"
        )

        data_summary += f"No. Observations: {self.n_obs}\n"
        return data_summary

    @classmethod
    def from_arrays(cls, x, y, d, t, identifier, z=None, s=None, use_other_treat_as_covariate=True, force_all_x_finite=True):
        # TODO: Implement initialization from arrays
        raise NotImplementedError("from_arrays is not implemented for DoubleMLPanelData")

    @property
    def datetime_unit(self):
        """
        The unit of the time variable.
        """
        return self._datetime_unit

    @property
    def d(self):
        """
        Array of treatment variable;
        Dynamic! Depends on the currently set treatment variable;
        To get an array of all treatment variables (independent of the currently set treatment variable)
        call ``obj.data[obj.d_cols].values``.
        """
        if pd.api.types.is_datetime64_any_dtype(self._d):
            return self._d.values.astype(f"datetime64[{self.datetime_unit}]")
        else:
            return self._d.values

    @property
    def t(self):
        """
        Array of time variable.
        """
        if pd.api.types.is_datetime64_any_dtype(self._d):
            return self._t.values.astype(f"datetime64[{self.datetime_unit}]")
        else:
            return self._t.values

    @property
    def id_col(self):
        """
        The id variable.
        """
        return self._id_col

    @id_col.setter
    def id_col(self, value):
        reset_value = hasattr(self, "_id_col")
        if not isinstance(value, str):
            raise TypeError(
                "The id variable id_col must be of str type. " f"{str(value)} of type {str(type(value))} was passed."
            )
        if value not in self.all_variables:
            raise ValueError("Invalid id variable id_col. " f"{value} is no data column.")
        self._id_col = value
        if reset_value:
            self._check_disjoint_sets()
            self._set_id_var()

    @property
    def id_var(self):
        """
        Array of id variable.
        """
        return self._id_var.values

    @property
    def id_var_unique(self):
        """
        Unique values of id variable.
        """
        return self._id_var_unique

    @property
    def n_obs(self):
        """
        The number of observations. For panel data, the number of unique values for id_col.
        """
        return len(self._id_var_unique)

    @property
    def g_col(self):
        """
        The treatment variable indicating the time of treatment exposure.
        """
        return self._d_cols[0]

    @DoubleMLData.d_cols.setter
    def d_cols(self, value):
        super(self.__class__, self.__class__).d_cols.__set__(self, value)
        if hasattr(self, "_g_values"):
            self._g_values = np.sort(np.unique(self.d))  # update unique values of g

    @property
    def g_values(self):
        """
        The unique values of the treatment variable (groups) ``d``.
        """
        return self._g_values

    @property
    def n_groups(self):
        """
        The number of groups.
        """
        return len(self.g_values)

    @DoubleMLData.t_col.setter
    def t_col(self, value):
        if value is None:
            raise TypeError("Invalid time variable t_col. Time variable required for panel data.")
        super(self.__class__, self.__class__).t_col.__set__(self, value)
        if hasattr(self, "_t_values"):
            self._t_values = np.sort(np.unique(self.t))  # update unique values of t

    @property
    def t_values(self):
        """
        The unique values of the time variable ``t``.
        """
        return self._t_values

    @property
    def n_t_periods(self):
        """
        The number of time periods.
        """
        return len(self.t_values)

    def _get_optional_col_sets(self):
        base_optional_col_sets = super()._get_optional_col_sets()
        id_col_set = {self.id_col}
        return [id_col_set] + base_optional_col_sets

    def _check_disjoint_sets(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLPanelData, self)._check_disjoint_sets()
        self._check_disjoint_sets_id_col()

    def _check_disjoint_sets_id_col(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLPanelData, self)._check_disjoint_sets()

        # special checks for the additional id variable (and the time variable)
        id_col_set = {self.id_col}
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        z_cols_set = set(self.z_cols or [])
        t_col_set = {self.t_col}  # t_col is not None for panel data
        # s_col not tested as not relevant for panel data

        id_col_check_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
            (t_col_set, "time variable", "``t_col``"),
        ]
        for set1, name, argument in id_col_check_args:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=argument,
                set2=id_col_set,
                name2="identifier variable",
                arg2="``id_col``",
            )

    def _set_id_var(self):
        assert_all_finite(self.data.loc[:, self.id_col])
        self._id_var = self.data.loc[:, self.id_col]
        self._id_var_unique = np.unique(self._id_var.values)
