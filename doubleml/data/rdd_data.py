import io

import pandas as pd
from sklearn.utils.validation import check_array

from doubleml.data.base_data import DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLRDDData(DoubleMLData):
    """Double machine learning data-backend for Regression Discontinuity Design models.

    :class:`DoubleMLRDDData` objects can be initialized from
    :class:`pandas.DataFrame`'s as well as :class:`numpy.ndarray`'s.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.

    y_col : str
        The outcome variable.

    d_cols : str or list
        The treatment variable(s).

    score_col : str
        The score/running variable for RDD models.

    x_cols : None, str or list
        The covariates.
        If ``None``, all variables (columns of ``data``) which are neither specified as outcome variable ``y_col``, nor
        treatment variables ``d_cols``, nor instrumental variables ``z_cols``, nor score variable ``score_col`` are
        used as covariates.
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> from doubleml import DoubleMLRDDData
    >>> from doubleml.rdd.datasets import make_simple_rdd_data
    >>> # initialization from pandas.DataFrame
    >>> data = make_simple_rdd_data(return_type='DataFrame')
    >>> columns = ["y", "d", "score"] + ["x" + str(i) for i in range(data["X"].shape[1])]
    >>> df = pd.DataFrame(np.column_stack((data["Y"], data["D"], data["score"], data["X"])), columns=columns)
    >>> obj_dml_data_from_df = DoubleMLRDDData(df, 'y', 'd', score_col='s')
    >>> # initialization from np.ndarray
    >>> obj_dml_data_from_array = DoubleMLRDDData.from_arrays(data["X"], data["Y"], data["D"], score=data["score"])
    """

    def __init__(
        self,
        data,
        y_col,
        d_cols,
        score_col,
        x_cols=None,
        z_cols=None,
        cluster_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        # Set score column before calling parent constructor
        self.score_col = score_col

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

        # Set score variable array after data is loaded
        self._set_score_var()

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        score,
        z=None,
        cluster_vars=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
    ):
        """
        Initialize :class:`DoubleMLRDDData` object from :class:`numpy.ndarray`'s.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of covariates.

        y : :class:`numpy.ndarray`
            Array of the outcome variable.

        d : :class:`numpy.ndarray`
            Array of treatment variables.

        score : :class:`numpy.ndarray`
            Array of the score/running variable for RDD models.

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
        >>> import numpy as np
        >>> import pandas as pd
        >>> from doubleml import DoubleMLRDDData
        >>> from doubleml.rdd.datasets import make_simple_rdd_data
        >>> # initialization from pandas.DataFrame
        >>> data = make_simple_rdd_data(return_type='DataFrame')
        >>> obj_dml_data_from_array = DoubleMLRDDData.from_arrays(data["X"], data["Y"], data["D"], score=data["score"])
        """
        # Prepare score variable
        score = check_array(score, ensure_2d=False, allow_nd=False)
        score = _assure_2d_array(score)
        if score.shape[1] != 1:
            raise ValueError("score must be a single column.")
        score_col = "score"

        # Create base data using parent class method
        base_data = DoubleMLData.from_arrays(
            x, y, d, z, cluster_vars, use_other_treat_as_covariate, force_all_x_finite, force_all_d_finite
        )

        # Add score variable to the DataFrame
        data = pd.concat((base_data.data, pd.DataFrame(score, columns=[score_col])), axis=1)

        return cls(
            data,
            base_data.y_col,
            base_data.d_cols,
            score_col,
            base_data.x_cols,
            base_data.z_cols,
            base_data.cluster_cols,
            base_data.use_other_treat_as_covariate,
            base_data.force_all_x_finite,
            base_data.force_all_d_finite,
        )

    @property
    def score_col(self):
        """
        The score/running variable.
        """
        return self._score_col

    @score_col.setter
    def score_col(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "The score variable score_col must be of str type. " f"{str(value)} of type {str(type(value))} was passed."
            )
        # Check if data exists (during initialization it might not)
        if hasattr(self, "_data") and value not in self.all_variables:
            raise ValueError("Invalid score variable score_col. The score variable is no data column.")
        self._score_col = value
        # Update score variable array if data is already loaded
        if hasattr(self, "_data"):
            self._set_score_var()

    @property
    def score(self):
        """
        Array of score/running variable.
        """
        return self._score.values

    def _get_optional_col_sets(self):
        """Get optional column sets including score column."""
        base_optional_col_sets = super()._get_optional_col_sets()
        score_col_set = {self.score_col}
        return [score_col_set] + base_optional_col_sets

    def _check_disjoint_sets(self):
        """Check that score column doesn't overlap with other variables."""
        # Apply standard checks from parent class
        super()._check_disjoint_sets()
        self._check_disjoint_sets_score_col()

    def _check_disjoint_sets_score_col(self):
        """Check that score column is disjoint from other variable sets."""
        score_col_set = {self.score_col}
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
                set2=score_col_set,
                name2="score variable",
                arg2="``score_col``",
            )

    def _set_score_var(self):
        """Set the score variable array."""
        if hasattr(self, "_data") and self.score_col in self.data.columns:
            self._score = self.data.loc[:, self.score_col]

    def __str__(self):
        """String representation."""
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        print("================== DoubleMLRDDData Object ==================", file=buf)
        print(f"Score variable: {self.score_col}", file=buf)
        print(data_summary, file=buf)
        return buf.getvalue()
