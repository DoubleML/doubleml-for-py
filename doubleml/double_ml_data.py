import numpy as np
import pandas as pd
import io

from sklearn.utils.validation import check_array, column_or_1d,  check_consistent_length
from sklearn.utils import assert_all_finite
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
                 use_other_treat_as_covariate=True,
                 force_all_x_finite=True):
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
        self._check_disjoint_sets_y_d_x_z()
        self.use_other_treat_as_covariate = use_other_treat_as_covariate
        self.force_all_x_finite = force_all_x_finite
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
    def from_arrays(cls, x, y, d, z=None, use_other_treat_as_covariate=True,
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

        return cls(data, y_col, d_cols, x_cols, z_cols, use_other_treat_as_covariate, force_all_x_finite)

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

    def _set_y_z(self):
        assert_all_finite(self.data.loc[:, self.y_col])
        self._y = self.data.loc[:, self.y_col]
        if self.z_cols is None:
            self._z = None
        else:
            assert_all_finite(self.data.loc[:, self.z_cols])
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

    def _check_disjoint_sets(self):
        # this function can be extended in inherited subclasses
        self._check_disjoint_sets_y_d_x_z()

    def _check_disjoint_sets_y_d_x_z(self):
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
                 use_other_treat_as_covariate=True,
                 force_all_x_finite=True):
        # we need to set cluster_cols (needs _data) before call to the super __init__ because of the x_cols setter
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be of pd.DataFrame type. '
                            f'{str(data)} of type {str(type(data))} was passed.')
        if not data.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')
        self._data = data
        self.cluster_cols = cluster_cols
        self._set_cluster_vars()
        super().__init__(data,
                         y_col,
                         d_cols,
                         x_cols,
                         z_cols,
                         use_other_treat_as_covariate,
                         force_all_x_finite)
        self._check_disjoint_sets_cluster_cols()

    def __str__(self):
        data_info = f'Outcome variable: {self.y_col}\n' \
                    f'Treatment variable(s): {self.d_cols}\n' \
                    f'Cluster variable(s): {self.cluster_cols}\n' \
                    f'Covariates: {self.x_cols}\n' \
                    f'Instrument variable(s): {self.z_cols}\n' \
                    f'No. Observations: {self.n_obs}\n'
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = '================== DoubleMLClusterData Object ==================\n' + \
              '\n------------------ Data summary      ------------------\n' + data_info + \
              '\n------------------ DataFrame info    ------------------\n' + df_info
        return res

    @classmethod
    def from_arrays(cls, x, y, d, cluster_vars, z=None, use_other_treat_as_covariate=True,
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
        dml_data = DoubleMLData.from_arrays(x, y, d, z, use_other_treat_as_covariate, force_all_x_finite)
        cluster_vars = check_array(cluster_vars, ensure_2d=False, allow_nd=False)
        cluster_vars = _assure_2d_array(cluster_vars)
        if cluster_vars.shape[1] == 1:
            cluster_cols = ['cluster_var']
        else:
            cluster_cols = [f'cluster_var{i + 1}' for i in np.arange(cluster_vars.shape[1])]

        data = pd.concat((pd.DataFrame(cluster_vars, columns=cluster_cols), dml_data.data), axis=1)

        return(cls(data, dml_data.y_col, dml_data.d_cols,
                   cluster_cols,
                   dml_data.x_cols, dml_data.z_cols,
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
            if self.z_cols is not None:
                y_d_z = set.union({self.y_col}, set(self.d_cols), set(self.z_cols), set(self.cluster_cols))
                x_cols = [col for col in self.data.columns if col not in y_d_z]
            else:
                y_d = set.union({self.y_col}, set(self.d_cols), set(self.cluster_cols))
                x_cols = [col for col in self.data.columns if col not in y_d]
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

    def _set_cluster_vars(self):
        assert_all_finite(self.data.loc[:, self.cluster_cols])
        self._cluster_vars = self.data.loc[:, self.cluster_cols]
