import numpy as np
import pandas as pd
import io

from abc import ABC, abstractmethod

from sklearn.utils.validation import check_array, column_or_1d,  check_consistent_length
from sklearn.utils import assert_all_finite
from sklearn.utils.multiclass import type_of_target
from .utils._estimation import _assure_2d_array
from .utils._checks import _check_set


class DoubleMLBaseData(ABC):
    """Base Class Double machine learning data-backends
    """
    def __init__(self,
                 data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be of pd.DataFrame type. '
                            f'{str(data)} of type {str(type(data))} was passed.')
        if not data.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')
        self._data = data

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = '================== DoubleMLBaseData Object ==================\n' + \
              '\n------------------ Data summary      ------------------\n' + data_summary + \
              '\n------------------ DataFrame info    ------------------\n' + df_info
        return res

    def _data_summary_str(self):
        data_summary = f'No. Observations: {self.n_obs}\n'
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
        return ['theta']

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
        treatment variables ``d_cols``, nor instrumental variables ``z_cols`` are used as covariates.
        Default is ``None``.

    z_cols : None, str or list
        The instrumental variable(s).
        Default is ``None``.

    t_col : None or str
        The time variable (only relevant/used for DiD Estimators).
        Default is ``None``.

    s_col : None or str
        The selection variable (only relevant/used for SSM Estimatiors).
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
    >>> from doubleml import DoubleMLData
    >>> from doubleml.datasets import make_plr_CCDDHNR2018
    >>> # initialization from pandas.DataFrame
    >>> df = make_plr_CCDDHNR2018(return_type='DataFrame')
    >>> obj_dml_data_from_df = DoubleMLData(df, 'y', 'd')
    >>> # initialization from np.ndarray
    >>> (x, y, d) = make_plr_CCDDHNR2018(return_type='array')
    >>> obj_dml_data_from_array = DoubleMLData.from_arrays(x, y, d)
    """
    def __init__(self,
                 data,
                 y_col,
                 d_cols,
                 x_cols=None,
                 z_cols=None,
                 t_col=None,
                 s_col=None,
                 use_other_treat_as_covariate=True,
                 force_all_x_finite=True):
        DoubleMLBaseData.__init__(self, data)

        self.y_col = y_col
        self.d_cols = d_cols
        self.z_cols = z_cols
        self.t_col = t_col
        self.s_col = s_col
        self.x_cols = x_cols
        self._check_disjoint_sets_y_d_x_z_t_s()
        self.use_other_treat_as_covariate = use_other_treat_as_covariate
        self.force_all_x_finite = force_all_x_finite
        self._binary_treats = self._check_binary_treats()
        self._binary_outcome = self._check_binary_outcome()
        self._set_y_z_t_s()
        # by default, we initialize to the first treatment variable
        self.set_x_d(self.d_cols[0])

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = '================== DoubleMLData Object ==================\n' + \
              '\n------------------ Data summary      ------------------\n' + data_summary + \
              '\n------------------ DataFrame info    ------------------\n' + df_info
        return res

    def _data_summary_str(self):
        data_summary = f'Outcome variable: {self.y_col}\n' \
                       f'Treatment variable(s): {self.d_cols}\n' \
                       f'Covariates: {self.x_cols}\n' \
                       f'Instrument variable(s): {self.z_cols}\n'
        if self.t_col is not None:
            data_summary += f'Time variable: {self.t_col}\n'
        if self.s_col is not None:
            data_summary += f'Selection variable: {self.s_col}\n'
        data_summary += f'No. Observations: {self.n_obs}\n'
        return data_summary

    @classmethod
    def from_arrays(cls, x, y, d, z=None, t=None, s=None, use_other_treat_as_covariate=True,
                    force_all_x_finite=True):
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

        t : :class:`numpy.ndarray`
            Array of the time variable (only relevant/used for DiD models).
            Default is ``None``.

        s : :class:`numpy.ndarray`
            Array of the selection variable (only relevant/used for SSM models).
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
        >>> from doubleml import DoubleMLData
        >>> from doubleml.datasets import make_plr_CCDDHNR2018
        >>> (x, y, d) = make_plr_CCDDHNR2018(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLData.from_arrays(x, y, d)
        """
        if isinstance(force_all_x_finite, str):
            if force_all_x_finite != 'allow-nan':
                raise ValueError("Invalid force_all_x_finite " + force_all_x_finite + ". " +
                                 "force_all_x_finite must be True, False or 'allow-nan'.")
        elif not isinstance(force_all_x_finite, bool):
            raise TypeError("Invalid force_all_x_finite. " +
                            "force_all_x_finite must be True, False or 'allow-nan'.")

        x = check_array(x, ensure_2d=False, allow_nd=False,
                        force_all_finite=force_all_x_finite)
        d = check_array(d, ensure_2d=False, allow_nd=False)
        y = column_or_1d(y, warn=True)

        x = _assure_2d_array(x)
        d = _assure_2d_array(d)

        y_col = 'y'
        if z is None:
            check_consistent_length(x, y, d)
            z_cols = None
        else:
            z = check_array(z, ensure_2d=False, allow_nd=False)
            z = _assure_2d_array(z)
            check_consistent_length(x, y, d, z)
            if z.shape[1] == 1:
                z_cols = ['z']
            else:
                z_cols = [f'z{i + 1}' for i in np.arange(z.shape[1])]

        if t is None:
            t_col = None
        else:
            t = column_or_1d(t, warn=True)
            check_consistent_length(x, y, d, t)
            t_col = 't'

        if s is None:
            s_col = None
        else:
            s = column_or_1d(s, warn=True)
            check_consistent_length(x, y, d, s)
            s_col = 's'

        if d.shape[1] == 1:
            d_cols = ['d']
        else:
            d_cols = [f'd{i+1}' for i in np.arange(d.shape[1])]

        x_cols = [f'X{i+1}' for i in np.arange(x.shape[1])]

        # basline version with features, outcome and treatments
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + [y_col] + d_cols)

        if z is not None:
            df_z = pd.DataFrame(z, columns=z_cols)
            data = pd.concat([data, df_z], axis=1)

        if t is not None:
            data[t_col] = t

        if s is not None:
            data[s_col] = s

        return cls(data, y_col, d_cols, x_cols, z_cols, t_col, s_col, use_other_treat_as_covariate, force_all_x_finite)

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
    def t(self):
        """
        Array of time variable.
        """
        if self.t_col is not None:
            return self._t.values
        else:
            return None

    @property
    def s(self):
        """
        Array of selection variable.
        """
        if self.s_col is not None:
            return self._s.values
        else:
            return None

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
        reset_value = hasattr(self, '_x_cols')
        if value is not None:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError('The covariates x_cols must be of str or list type (or None). '
                                f'{str(value)} of type {str(type(value))} was passed.')
            if not len(set(value)) == len(value):
                raise ValueError('Invalid covariates x_cols: '
                                 'Contains duplicate values.')
            if not set(value).issubset(set(self.all_variables)):
                raise ValueError('Invalid covariates x_cols. '
                                 'At least one covariate is no data column.')
            assert set(value).issubset(set(self.all_variables))
            self._x_cols = value
        else:
            excluded_cols = set.union({self.y_col}, set(self.d_cols))
            if (self.z_cols is not None):
                excluded_cols = set.union(excluded_cols, set(self.z_cols))
            for col in [self.t_col, self.s_col]:
                col = _check_set(col)
                excluded_cols = set.union(excluded_cols, col)
            self._x_cols = [col for col in self.data.columns if col not in excluded_cols]
        if reset_value:
            self._check_disjoint_sets()
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    @property
    def d_cols(self):
        """
        The treatment variable(s).
        """
        return self._d_cols

    @d_cols.setter
    def d_cols(self, value):
        reset_value = hasattr(self, '_d_cols')
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError('The treatment variable(s) d_cols must be of str or list type. '
                            f'{str(value)} of type {str(type(value))} was passed.')
        if not len(set(value)) == len(value):
            raise ValueError('Invalid treatment variable(s) d_cols: '
                             'Contains duplicate values.')
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError('Invalid treatment variable(s) d_cols. '
                             'At least one treatment variable is no data column.')
        self._d_cols = value
        if reset_value:
            self._check_disjoint_sets()
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    @property
    def y_col(self):
        """
        The outcome variable.
        """
        return self._y_col

    @y_col.setter
    def y_col(self, value):
        reset_value = hasattr(self, '_y_col')
        if not isinstance(value, str):
            raise TypeError('The outcome variable y_col must be of str type. '
                            f'{str(value)} of type {str(type(value))} was passed.')
        if value not in self.all_variables:
            raise ValueError('Invalid outcome variable y_col. '
                             f'{value} is no data column.')
        self._y_col = value
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_t_s()

    @property
    def z_cols(self):
        """
        The instrumental variable(s).
        """
        return self._z_cols

    @z_cols.setter
    def z_cols(self, value):
        reset_value = hasattr(self, '_z_cols')
        if value is not None:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError('The instrumental variable(s) z_cols must be of str or list type (or None). '
                                f'{str(value)} of type {str(type(value))} was passed.')
            if not len(set(value)) == len(value):
                raise ValueError('Invalid instrumental variable(s) z_cols: '
                                 'Contains duplicate values.')
            if not set(value).issubset(set(self.all_variables)):
                raise ValueError('Invalid instrumental variable(s) z_cols. '
                                 'At least one instrumental variable is no data column.')
            self._z_cols = value
        else:
            self._z_cols = None
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_t_s()

    @property
    def t_col(self):
        """
        The time variable.
        """
        return self._t_col

    @t_col.setter
    def t_col(self, value):
        reset_value = hasattr(self, '_t_col')
        if value is not None:
            if not isinstance(value, str):
                raise TypeError('The time variable t_col must be of str type (or None). '
                                f'{str(value)} of type {str(type(value))} was passed.')
            if value not in self.all_variables:
                raise ValueError('Invalid time variable t_col. '
                                 f'{value} is no data column.')
            self._t_col = value
        else:
            self._t_col = None
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_t_s()

    @property
    def s_col(self):
        """
        The selection variable.
        """
        return self._s_col

    @s_col.setter
    def s_col(self, value):
        reset_value = hasattr(self, '_s_col')
        if value is not None:
            if not isinstance(value, str):
                raise TypeError('The selection variable s_col must be of str type (or None). '
                                f'{str(value)} of type {str(type(value))} was passed.')
            if value not in self.all_variables:
                raise ValueError('Invalid selection variable s_col. '
                                 f'{value} is no data column.')
            self._s_col = value
        else:
            self._s_col = None
        if reset_value:
            self._check_disjoint_sets()
            self._set_y_z_t_s()

    @property
    def use_other_treat_as_covariate(self):
        """
        Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
        """
        return self._use_other_treat_as_covariate

    @use_other_treat_as_covariate.setter
    def use_other_treat_as_covariate(self, value):
        reset_value = hasattr(self, '_use_other_treat_as_covariate')
        if not isinstance(value, bool):
            raise TypeError('use_other_treat_as_covariate must be True or False. '
                            f'Got {str(value)}.')
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
        reset_value = hasattr(self, '_force_all_x_finite')
        if isinstance(value, str):
            if value != 'allow-nan':
                raise ValueError("Invalid force_all_x_finite " + value + ". " +
                                 "force_all_x_finite must be True, False or 'allow-nan'.")
        elif not isinstance(value, bool):
            raise TypeError("Invalid force_all_x_finite. " +
                            "force_all_x_finite must be True, False or 'allow-nan'.")
        self._force_all_x_finite = value
        if reset_value:
            # by default, we initialize to the first treatment variable
            self.set_x_d(self.d_cols[0])

    def _set_y_z_t_s(self):
        assert_all_finite(self.data.loc[:, self.y_col])
        self._y = self.data.loc[:, self.y_col]
        if self.z_cols is None:
            self._z = None
        else:
            assert_all_finite(self.data.loc[:, self.z_cols])
            self._z = self.data.loc[:, self.z_cols]

        if self.t_col is None:
            self._t = None
        else:
            assert_all_finite(self.data.loc[:, self.t_col])
            self._t = self.data.loc[:, self.t_col]

        if self.s_col is None:
            self._s = None
        else:
            assert_all_finite(self.data.loc[:, self.s_col])
            self._s = self.data.loc[:, self.s_col]

    def set_x_d(self, treatment_var):
        """
        Function that assigns the role for the treatment variables in the multiple-treatment case.

        Parameters
        ----------
        treatment_var : str
            Active treatment variable that will be set to d.
        """
        if not isinstance(treatment_var, str):
            raise TypeError('treatment_var must be of str type. '
                            f'{str(treatment_var)} of type {str(type(treatment_var))} was passed.')
        if treatment_var not in self.d_cols:
            raise ValueError('Invalid treatment_var. '
                             f'{treatment_var} is not in d_cols.')
        if self.use_other_treat_as_covariate:
            # note that the following line needs to be adapted in case an intersection of x_cols and d_cols as allowed
            # (see https://github.com/DoubleML/doubleml-for-py/issues/83)
            xd_list = self.x_cols + self.d_cols
            xd_list.remove(treatment_var)
        else:
            xd_list = self.x_cols
        assert_all_finite(self.data.loc[:, treatment_var])
        if self.force_all_x_finite:
            assert_all_finite(self.data.loc[:, xd_list],
                              allow_nan=self.force_all_x_finite == 'allow-nan')
        self._d = self.data.loc[:, treatment_var]
        self._X = self.data.loc[:, xd_list]

    def _check_binary_treats(self):
        is_binary = pd.Series(dtype=bool, index=self.d_cols)
        for treatment_var in self.d_cols:
            this_d = self.data.loc[:, treatment_var]
            binary_treat = (type_of_target(this_d) == 'binary')
            zero_one_treat = np.all((np.power(this_d, 2) - this_d) == 0)
            is_binary[treatment_var] = (binary_treat & zero_one_treat)
        return is_binary

    def _check_binary_outcome(self):
        y = self.data.loc[:, self.y_col]
        binary_outcome = (type_of_target(y) == 'binary')
        zero_one_outcome = np.all((np.power(y, 2) - y) == 0)
        is_binary = (binary_outcome & zero_one_outcome)
        return is_binary

    def _check_disjoint_sets(self):
        # this function can be extended in inherited subclasses
        self._check_disjoint_sets_y_d_x_z_t_s()

    def _check_disjoint_sets_y_d_x_z_t_s(self):
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        if not y_col_set.isdisjoint(x_cols_set):
            raise ValueError(f'{str(self.y_col)} cannot be set as outcome variable ``y_col`` and covariate in '
                             '``x_cols``.')
        if not y_col_set.isdisjoint(d_cols_set):
            raise ValueError(f'{str(self.y_col)} cannot be set as outcome variable ``y_col`` and treatment variable in '
                             '``d_cols``.')
        # note that the line xd_list = self.x_cols + self.d_cols in method set_x_d needs adaption if an intersection of
        # x_cols and d_cols as allowed (see https://github.com/DoubleML/doubleml-for-py/issues/83)
        if not d_cols_set.isdisjoint(x_cols_set):
            raise ValueError('At least one variable/column is set as treatment variable (``d_cols``) and as covariate'
                             '(``x_cols``). Consider using parameter ``use_other_treat_as_covariate``.')

        if self.z_cols is not None:
            z_cols_set = set(self.z_cols)
            if not y_col_set.isdisjoint(z_cols_set):
                raise ValueError(f'{str(self.y_col)} cannot be set as outcome variable ``y_col`` and instrumental '
                                 'variable in ``z_cols``.')
            if not d_cols_set.isdisjoint(z_cols_set):
                raise ValueError('At least one variable/column is set as treatment variable (``d_cols``) and '
                                 'instrumental variable in ``z_cols``.')
            if not x_cols_set.isdisjoint(z_cols_set):
                raise ValueError('At least one variable/column is set as covariate (``x_cols``) and instrumental '
                                 'variable in ``z_cols``.')

        self._check_disjoint_sets_t_s()

    def _check_disjoint_sets_t_s(self):
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)

        if self.t_col is not None:
            t_col_set = {self.t_col}
            if not t_col_set.isdisjoint(x_cols_set):
                raise ValueError(f'{str(self.t_col)} cannot be set as time variable ``t_col`` and covariate in '
                                 '``x_cols``.')
            if not t_col_set.isdisjoint(d_cols_set):
                raise ValueError(f'{str(self.t_col)} cannot be set as time variable ``t_col`` and treatment variable in '
                                 '``d_cols``.')
            if not t_col_set.isdisjoint(y_col_set):
                raise ValueError(f'{str(self.t_col)} cannot be set as time variable ``t_col`` and outcome variable '
                                 '``y_col``.')
            if self.z_cols is not None:
                z_cols_set = set(self.z_cols)
                if not t_col_set.isdisjoint(z_cols_set):
                    raise ValueError(f'{str(self.t_col)} cannot be set as time variable ``t_col`` and instrumental '
                                     'variable in ``z_cols``.')

        if self.s_col is not None:
            s_col_set = {self.s_col}
            if not s_col_set.isdisjoint(x_cols_set):
                raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and covariate in '
                                 '``x_cols``.')
            if not s_col_set.isdisjoint(d_cols_set):
                raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and treatment variable in '
                                 '``d_cols``.')
            if not s_col_set.isdisjoint(y_col_set):
                raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and outcome variable '
                                 '``y_col``.')
            if self.z_cols is not None:
                z_cols_set = set(self.z_cols)
                if not s_col_set.isdisjoint(z_cols_set):
                    raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and instrumental '
                                     'variable in ``z_cols``.')
            if self.t_col is not None:
                t_col_set = {self.t_col}
                if not s_col_set.isdisjoint(t_col_set):
                    raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and time variable '
                                     '``t_col``.')


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
        The selection variable (only relevant/used for SSM Estimatiors).
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
    def __init__(self,
                 data,
                 y_col,
                 d_cols,
                 cluster_cols,
                 x_cols=None,
                 z_cols=None,
                 t_col=None,
                 s_col=None,
                 use_other_treat_as_covariate=True,
                 force_all_x_finite=True):
        DoubleMLBaseData.__init__(self, data)

        # we need to set cluster_cols (needs _data) before call to the super __init__ because of the x_cols setter
        self.cluster_cols = cluster_cols
        self._set_cluster_vars()
        DoubleMLData.__init__(self,
                              data,
                              y_col,
                              d_cols,
                              x_cols,
                              z_cols,
                              t_col,
                              s_col,
                              use_other_treat_as_covariate,
                              force_all_x_finite)
        self._check_disjoint_sets_cluster_cols()

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = '================== DoubleMLClusterData Object ==================\n' + \
              '\n------------------ Data summary      ------------------\n' + data_summary + \
              '\n------------------ DataFrame info    ------------------\n' + df_info
        return res

    def _data_summary_str(self):
        data_summary = f'Outcome variable: {self.y_col}\n' \
                       f'Treatment variable(s): {self.d_cols}\n' \
                       f'Cluster variable(s): {self.cluster_cols}\n' \
                       f'Covariates: {self.x_cols}\n' \
                       f'Instrument variable(s): {self.z_cols}\n'
        if self.t_col is not None:
            data_summary += f'Time variable: {self.t_col}\n'
        if self.s_col is not None:
            data_summary += f'Selection variable: {self.s_col}\n'

        data_summary += f'No. Observations: {self.n_obs}\n'
        return data_summary

    @classmethod
    def from_arrays(cls, x, y, d, cluster_vars, z=None, t=None, s=None, use_other_treat_as_covariate=True,
                    force_all_x_finite=True):
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
            Array of the selection variable (only relevant/used for SSM models).
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
            cluster_cols = ['cluster_var']
        else:
            cluster_cols = [f'cluster_var{i + 1}' for i in np.arange(cluster_vars.shape[1])]

        data = pd.concat((pd.DataFrame(cluster_vars, columns=cluster_cols), dml_data.data), axis=1)

        return (cls(data, dml_data.y_col, dml_data.d_cols, cluster_cols,
                    dml_data.x_cols, dml_data.z_cols, dml_data.t_col, dml_data.s_col,
                    dml_data.use_other_treat_as_covariate, dml_data.force_all_x_finite))

    @property
    def cluster_cols(self):
        """
        The cluster variable(s).
        """
        return self._cluster_cols

    @cluster_cols.setter
    def cluster_cols(self, value):
        reset_value = hasattr(self, '_cluster_cols')
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError('The cluster variable(s) cluster_cols must be of str or list type. '
                            f'{str(value)} of type {str(type(value))} was passed.')
        if not len(set(value)) == len(value):
            raise ValueError('Invalid cluster variable(s) cluster_cols: '
                             'Contains duplicate values.')
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError('Invalid cluster variable(s) cluster_cols. '
                             'At least one cluster variable is no data column.')
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

    @DoubleMLData.x_cols.setter
    def x_cols(self, value):
        if value is not None:
            # this call might become much easier with https://github.com/python/cpython/pull/26194
            super(self.__class__, self.__class__).x_cols.__set__(self, value)
        else:
            if self.s_col is None:
                if (self.z_cols is not None) & (self.t_col is not None):
                    y_d_z_t = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), {self.t_col}, set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_z_t]
                elif self.z_cols is not None:
                    y_d_z = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_z]
                elif self.t_col is not None:
                    y_d_t = set.union({self.y_col}, set(self.d_cols), {self.t_col}, set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_t]
                else:
                    y_d = set.union({self.y_col}, set(self.d_cols), set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d]
            else:
                if (self.z_cols is not None) & (self.t_col is not None):
                    y_d_z_t_s = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), {self.t_col}, {self.s_col},
                                          set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_z_t_s]
                elif self.z_cols is not None:
                    y_d_z_s = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), {self.s_col}, set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_z_s]
                elif self.t_col is not None:
                    y_d_t_s = set.union({self.y_col}, set(self.d_cols), {self.t_col}, {self.s_col}, set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_t_s]
                else:
                    y_d_s = set.union({self.y_col}, set(self.d_cols), {self.s_col}, set(self.cluster_cols))
                    x_cols = [col for col in self.data.columns if col not in y_d_s]
            # this call might become much easier with https://github.com/python/cpython/pull/26194
            super(self.__class__, self.__class__).x_cols.__set__(self, x_cols)

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
        t_col_set = {self.t_col}
        s_col_set = {self.s_col}

        if not y_col_set.isdisjoint(cluster_cols_set):
            raise ValueError(f'{str(self.y_col)} cannot be set as outcome variable ``y_col`` and cluster '
                             'variable in ``cluster_cols``.')
        if not d_cols_set.isdisjoint(cluster_cols_set):
            raise ValueError('At least one variable/column is set as treatment variable (``d_cols``) and '
                             'cluster variable in ``cluster_cols``.')
        # TODO: Is the following combination allowed, or not?
        if not x_cols_set.isdisjoint(cluster_cols_set):
            raise ValueError('At least one variable/column is set as covariate (``x_cols``) and cluster '
                             'variable in ``cluster_cols``.')
        if self.z_cols is not None:
            z_cols_set = set(self.z_cols)
            if not z_cols_set.isdisjoint(cluster_cols_set):
                raise ValueError('At least one variable/column is set as instrumental variable (``z_cols``) and '
                                 'cluster variable in ``cluster_cols``.')
        if self.t_col is not None:
            if not t_col_set.isdisjoint(cluster_cols_set):
                raise ValueError(f'{str(self.t_col)} cannot be set as time variable ``t_col`` and '
                                 'cluster variable in ``cluster_cols``.')
        if self.s_col is not None:
            if not s_col_set.isdisjoint(cluster_cols_set):
                raise ValueError(f'{str(self.s_col)} cannot be set as selection variable ``s_col`` and '
                                 'cluster variable in ``cluster_cols``.')

    def _set_cluster_vars(self):
        assert_all_finite(self.data.loc[:, self.cluster_cols])
        self._cluster_vars = self.data.loc[:, self.cluster_cols]
