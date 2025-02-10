import io

import numpy as np
from sklearn.utils import assert_all_finite

from doubleml.data.base_data import DoubleMLBaseData, DoubleMLData


class DoubleMLPanelData(DoubleMLData):
    """Double machine learning data-backend for panel data in long format.
    :class:`DoubleMLPanelData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.
    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s) indicating the treatment groups in terms of first time of treatment exposure.

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols`` are used as covariates.
        Default is ``None``.

    t_col : None or str
        The time variable (only relevant/used for DiD Estimators).
        Default is ``None``.
        # TODO: Check defaults for panel data setting

    z_cols : None, str or list
        The instrumental variable(s).
        Default is ``None``.

    s_col : None or str
        The selection variable (only relevant/used for SSM Estimatiors).
        Default is ``None``.

    id_col : None or str
        The id variable (only relevant/used for DiD estimators).
        Default is ``None``.
        # TODO: Check defaults for panel data setting

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
    # TODO: Add examples
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        x_cols=None,
        z_cols=None,
        t_col=None,
        s_col=None,
        id_col=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
    ):
        DoubleMLBaseData.__init__(self, data)

        # we need to set id_col (needs _data) before call to the super __init__ because of the x_cols setter
        self.id_col = id_col
        self._set_id_var()

        DoubleMLData.__init__(
            self, data, y_col, d_cols, x_cols, z_cols, t_col, s_col, use_other_treat_as_covariate, force_all_x_finite
        )
        self._check_disjoint_sets_id_col()

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
        )
        if self.t_col is not None:
            data_summary += f"Time variable: {self.t_col}\n"
        if self.id_col is not None:
            data_summary += f"Id variable: {self.id_col}\n"
        if self.s_col is not None:
            data_summary += f"Selection variable: {self.s_col}\n"

        data_summary += f"No. Observations: {self.n_obs}\n"
        return data_summary

    @classmethod
    def from_arrays(cls, x, y, d, t, id, z=None, s=None, use_other_treat_as_covariate=True, force_all_x_finite=True):
        # TODO: Implement initialization from arrays
        raise NotImplementedError("from_arrays is not implemented for DoubleMLPanelData")

    @property
    def id_col(self):
        """
        The id variable.
        """
        return self._id_col

    @property
    def g_col(self):
        """
        The treatment variable indicating the time of treatment exposure.
        """
        return self._d_cols[0]

    @id_col.setter
    def id_col(self, value):
        reset_value = hasattr(self, "_id_col")
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    "The id variable id_col must be of str type. "
                    f"{str(value)} of type {str(type(value))} was passed."
                )
            if value not in self.all_variables:
                raise ValueError("Invalid id variable id_col. " f"{value} is no data column.")
            self._id_col = value
        else:
            self._id_col = None
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
        if self.id_col is not None:
            return len(np.unique(self.id_var))
        else:
            return self.data.shape[0]

    @property
    def n_t_periods(self):
        """
        The number of time periods.
        """
        return len(self.t_values)

    @property
    def n_groups(self):
        """
        The number of groups.
        """
        return len(self.g_values)

    @property
    def g_values(self):
        """
        The unique values of the treatment variable (groups) ``d``.
        """
        return np.unique(self.d)

    @property
    def t_values(self):
        """
        The unique values of the time variable ``t``.
        """
        return np.unique(self.t)

    @DoubleMLData.x_cols.setter
    def x_cols(self, value):
        if value is not None:
            # this call might become much easier with https://github.com/python/cpython/pull/26194
            super(self.__class__, self.__class__).x_cols.__set__(self, value)
        else:
            if self.s_col is None:
                if (self.z_cols is not None) & (self.t_col is not None):
                    y_d_z_t = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), {self.t_col}, set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_z_t]
                elif self.z_cols is not None:
                    y_d_z = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_z]
                elif self.t_col is not None:
                    y_d_t = set.union({self.y_col}, set(self.d_cols), {self.t_col}, set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_t]
                else:
                    y_d = set.union({self.y_col}, set(self.d_cols), set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d]
            else:
                if (self.z_cols is not None) & (self.t_col is not None):
                    y_d_z_t_s = set.union(
                        {self.y_col}, set(self.d_cols), set(self.z_cols), {self.t_col}, {self.s_col}, set(self.id_col)
                    )
                    x_cols = [col for col in self.data.columns if col not in y_d_z_t_s]
                elif self.z_cols is not None:
                    y_d_z_s = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), {self.s_col}, set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_z_s]
                elif self.t_col is not None:
                    y_d_t_s = set.union({self.y_col}, set(self.d_cols), {self.t_col}, {self.s_col}, set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_t_s]
                else:
                    y_d_s = set.union({self.y_col}, set(self.d_cols), {self.s_col}, set(self.id_col))
                    x_cols = [col for col in self.data.columns if col not in y_d_s]
            # this call might become much easier with https://github.com/python/cpython/pull/26194
            super(self.__class__, self.__class__).x_cols.__set__(self, x_cols)

    def _check_disjoint_sets(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLPanelData, self)._check_disjoint_sets()
        self._check_disjoint_sets_id_col()

    def _check_disjoint_sets_id_col(self):
        # apply the standard checks from the DoubleMLData class
        super(DoubleMLPanelData, self)._check_disjoint_sets()

        # special checks for the additional cluster variables
        id_col_set = set(self.id_col)
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)
        t_col_set = {self.t_col}
        s_col_set = {self.s_col}

        if not y_col_set.isdisjoint(id_col_set):
            raise ValueError(
                f"{str(self.y_col)} cannot be set as outcome variable ``y_col`` and id " "variable in ``id_col``."
            )
        if not d_cols_set.isdisjoint(id_col_set):
            raise ValueError(
                "At least one variable/column is set as treatment variable (``d_cols``) and " "id variable in ``id_col``."
            )
        if not x_cols_set.isdisjoint(id_col_set):
            raise ValueError("At least one variable/column is set as covariate (``x_cols``) and id " "variable in ``id_col``.")
        if self.z_cols is not None:
            z_cols_set = set(self.z_cols)
            if not z_cols_set.isdisjoint(id_col_set):
                raise ValueError(
                    "At least one variable/column is set as instrumental variable (``z_cols``) and "
                    "id variable in ``id_col``."
                )
        if self.t_col is not None:
            if not t_col_set.isdisjoint(id_col_set):
                raise ValueError(
                    f"{str(self.t_col)} cannot be set as time variable ``t_col`` and " "id variable in ``id_col``."
                )
        if self.s_col is not None:
            if not s_col_set.isdisjoint(id_col_set):
                raise ValueError(
                    f"{str(self.s_col)} cannot be set as selection variable ``s_col`` and " "id variable in ``id_col``."
                )

    def _set_id_var(self):
        assert_all_finite(self.data.loc[:, self.id_col])
        self._id_var = self.data.loc[:, self.id_col]
        self._id_var_unique = np.unique(self._id_var.values)
