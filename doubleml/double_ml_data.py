import numpy as np
import pandas as pd
import io

from sklearn.utils.validation import check_array, column_or_1d,  check_consistent_length
from sklearn.utils.multiclass import type_of_target
from ._utils import _assure_2d_array


class DoubleMLData:
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

    use_other_treat_as_covariate : bool
        Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
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
                 use_other_treat_as_covariate=True):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be of pd.DataFrame type. '
                            f'{str(data)} of type {str(type(data))} was passed.')
        if not data.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')
        self._data = data

        self.y_col = y_col
        self.d_cols = d_cols
        self.z_cols = z_cols
        self.x_cols = x_cols
        self._check_disjoint_sets()
        self.use_other_treat_as_covariate = use_other_treat_as_covariate
        self._binary_treats = self._check_binary_treats()
        self._set_y_z()
        # by default, we initialize to the first treatment variable
        self.set_x_d(self.d_cols[0])

    def __str__(self):
        data_info = f'Outcome variable: {self.y_col}\n' \
                    f'Treatment variable(s): {self.d_cols}\n' \
                    f'Covariates: {self.x_cols}\n' \
                    f'Instrument variable(s): {self.z_cols}\n' \
                    f'No. Observations: {self.n_obs}\n'
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = '================== DoubleMLData Object ==================\n' + \
              '\n------------------ Data summary      ------------------\n' + data_info + \
              '\n------------------ DataFrame info    ------------------\n' + df_info
        return res

    @classmethod
    def from_arrays(cls, x, y, d, z=None, use_other_treat_as_covariate=True):
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

        use_other_treat_as_covariate : bool
            Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
            Default is ``True``.

        Examples
        --------
        >>> from doubleml import DoubleMLData
        >>> from doubleml.datasets import make_plr_CCDDHNR2018
        >>> (x, y, d) = make_plr_CCDDHNR2018(return_type='array')
        >>> obj_dml_data_from_array = DoubleMLData.from_arrays(x, y, d)
        """
        x = check_array(x, ensure_2d=False, allow_nd=False)
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

        if d.shape[1] == 1:
            d_cols = ['d']
        else:
            d_cols = [f'd{i+1}' for i in np.arange(d.shape[1])]

        x_cols = [f'X{i+1}' for i in np.arange(x.shape[1])]

        if z is None:
            data = pd.DataFrame(np.column_stack((x, y, d)),
                                columns=x_cols + [y_col] + d_cols)
        else:
            data = pd.DataFrame(np.column_stack((x, y, d, z)),
                                columns=x_cols + [y_col] + d_cols + z_cols)

        return cls(data, y_col, d_cols, x_cols, z_cols, use_other_treat_as_covariate)

    @property
    def data(self):
        """
        The data.
        """
        return self._data

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
    def all_variables(self):
        """
        All variables available in the dataset.
        """
        return self.data.columns

    @property
    def n_treat(self):
        """
        The number of treatment variables.
        """
        return len(self.d_cols)

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
    def n_obs(self):
        """
        The number of observations.
        """
        return self.data.shape[0]

    @property
    def binary_treats(self):
        """
        Series with logical(s) indicating whether the treatment variable(s) are binary with values 0 and 1.
        """
        return self._binary_treats

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
            # x_cols defaults to all columns but y_col, d_cols and z_cols
            if self.z_cols is not None:
                y_d_z = set.union({self.y_col}, set(self.d_cols), set(self.z_cols))
                x_cols = [col for col in self.data.columns if col not in y_d_z]
            else:
                y_d = set.union({self.y_col}, set(self.d_cols))
                x_cols = [col for col in self.data.columns if col not in y_d]
            self._x_cols = x_cols
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
            self._set_y_z()

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
            self._set_y_z()

    @property
    def use_other_treat_as_covariate(self):
        """
        Indicates whether in the multiple-treatment case the other treatment variables should be added as covariates.
        """
        return self._use_other_treat_as_covariate

    @use_other_treat_as_covariate.setter
    def use_other_treat_as_covariate(self, value):
        if not isinstance(value, bool):
            raise TypeError('use_other_treat_as_covariate must be True or False. '
                            f'Got {str(value)}.')
        self._use_other_treat_as_covariate = value

    def _set_y_z(self):
        self._y = self.data.loc[:, self.y_col]
        if self.z_cols is None:
            self._z = None
        else:
            self._z = self.data.loc[:, self.z_cols]

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

    def _check_disjoint_sets(self):
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
