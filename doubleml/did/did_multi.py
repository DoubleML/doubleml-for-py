from doubleml.data import DoubleMLPanelData


class DoubleMLDIDMulti:
    """Double machine learning for multi-period difference-in-differences models.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLPanelData` object
        The :class:`DoubleMLPanelData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(d,X) = E[Y_1-Y_0|D=d, X]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=1|X]`.
        Only relevant for ``score='observational'``. Default is ``None``.

    g_values : array-like
        Array of treatment group values specifying the first treatment period for each group.
        Default is ``None``, which uses all available treatment groups.

    t_values : array-like
        Array of time period values for evaluation.
        Default is ``None``, which uses all available time periods.

    control_group : str
        Specifies the control group. Either ``'never_treated'`` or ``'not_yet_treated'``.
        Default is ``'never_treated'``.

    n_folds : int
        Number of folds for cross-fitting.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'observational'`` or ``'experimental'``) specifying the score function.
        The ``'experimental'`` scores refers to an A/B setting, where the treatment is independent
        from the pretreatment covariates.
        Default is ``'observational'``.

    in_sample_normalization : bool
        Indicates whether to use in-sample normalization of weights.
        Default is ``True``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization.
        Default is ``True``.

    print_periods : bool
        Indicates whether to print information about the evaluated periods.
        Default is ``False``.

    Examples
    --------
    TODO: Add example
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m=None,
                 g_values=None,
                 t_values=None,
                 control_group='never_treated',
                 n_folds=5,
                 n_rep=1,
                 score='observational',
                 in_sample_normalization=True,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True,
                 print_periods=False):

        self._dml_data = obj_dml_data
        self._is_cluster_data = False
        self._is_panel_data = isinstance(obj_dml_data, DoubleMLPanelData)
        self._check_data(self._dml_data)
        self._print_periods = print_periods

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError('The data has to be a DoubleMLPanelData object. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise NotImplementedError(
                'Incompatible data. ' +
                ' and '.join(obj_dml_data.z_cols) +
                ' have been set as instrumental variable(s). '
                'At the moment there are not DiD models with instruments implemented.')
        return
