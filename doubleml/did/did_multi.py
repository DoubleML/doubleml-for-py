from sklearn.base import clone

from doubleml.did import DoubleMLDIDBINARY

from doubleml.data import DoubleMLPanelData
from doubleml.double_ml import DoubleML
from doubleml.utils._checks import _check_score, _check_trimming


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

    gt_combinations : array-like
        TODO: Add description

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

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m=None,
        gt_combinations=None,
        control_group="never_treated",
        n_folds=5,
        n_rep=1,
        score="observational",
        in_sample_normalization=True,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
        print_periods=False,
    ):

        self._dml_data = obj_dml_data
        self._is_cluster_data = False
        self._is_panel_data = isinstance(obj_dml_data, DoubleMLPanelData)
        self._check_data(self._dml_data)
        self._print_periods = print_periods

        valid_control_groups = ["never_treated", "not_yet_treated"]
        if control_group not in valid_control_groups:
            raise ValueError(f"The control group has to be one of {valid_control_groups}. " + f"{control_group} was passed.")
        self._control_group = control_group

        # TODO: Allow for different gt_combinations (use only combinations)
        self._gt_combinations = gt_combinations
        # TODO: ADD CHECKS FOR gt_combinations
        self._all_gt_labels = [f"ATT({g},{t_eval})" for g, t_pre, t_eval in self.gt_combinations]

        # TODO: Check what to export and what not
        self._in_sample_normalization = in_sample_normalization
        if not isinstance(self.in_sample_normalization, bool):
            raise TypeError(
                "in_sample_normalization indicator has to be boolean. "
                + f"Object of type {str(type(self.in_sample_normalization))} passed."
            )

        self._n_folds = n_folds
        self._n_rep = n_rep

        # check score
        self._score = score
        valid_scores = ["observational", "experimental"]
        _check_score(self.score, valid_scores, allow_callable=False)

        # initialize framework which is constructed after the fit method is called
        self._framework = None
        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        ml_g_is_classifier = DoubleML._check_learner(ml_g, "ml_g", regressor=True, classifier=True)
        _ = DoubleML._check_learner(ml_m, "ml_m", regressor=False, classifier=True)
        self._learner = {"ml_g": clone(ml_g), "ml_m": clone(ml_m)}
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {"ml_g": "predict_proba", "ml_m": "predict_proba"}
            else:
                raise ValueError(
                    f"The ml_g learner {str(ml_g)} was identified as classifier "
                    "but the outcome variable is not binary with values 0 and 1."
                )
        else:
            self._predict_method = {"ml_g": "predict", "ml_m": "predict_proba"}

        # perform sample splitting
        self._smpls = None

        # TODO: Check draw_sample_splitting here vs. DoubleMLDIDBINARY
        self._draw_sample_splitting = draw_sample_splitting

        # initialize all models if splits are known
        self._modellist = self._initialize_models()

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def control_group(self):
        """
        The control group.
        """
        return self._control_group

    # TODO: Define a setter for gt_combinations
    @property
    def gt_combinations(self):
        """
        The combinations of g and t values.
        """
        return self._gt_combinations

    @property
    def in_sample_normalization(self):
        """
        Indicates whether the in sample normalization of weights are used.
        """
        return self._in_sample_normalization

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

    @property
    def n_folds(self):
        """
        Number of folds.
        """
        return self._n_folds

    @property
    def n_rep(self):
        """
        Number of repetitions for the sample splitting.
        """
        return self._n_rep

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError(
                "The data has to be a DoubleMLPanelData object. "
                f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise NotImplementedError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "At the moment there are not DiD models with instruments implemented."
            )
        return

    def _initialize_models(self):
        modellist = [None] * len(self.gt_combinations)
        kwargs = {
            'obj_dml_data': self._dml_data,
            'ml_g': self._learner['ml_g'],
            'ml_m': self._learner['ml_m'],
            'control_group': self._control_group,
            'score': self.score,
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'trimming_rule': self.trimming_rule,
            'trimming_threshold': self.trimming_threshold,
            'in_sample_normalization': self.in_sample_normalization,
            'draw_sample_splitting': True,
            'print_periods': self._print_periods
        }
        for i_model, (g_value, t_value_pre, t_value_eval) in enumerate(self.gt_combinations):
            # initialize models for all levels
            model = DoubleMLDIDBINARY(
                g_value=g_value,
                t_value_pre=t_value_pre,
                t_value_eval=t_value_eval,
                **kwargs
            )

            modellist[i_model] = model

        return modellist
