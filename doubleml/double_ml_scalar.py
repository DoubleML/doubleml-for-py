"""
Abstract base class for scalar DoubleML models (single parameter estimation).
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np

from .data.base_data import DoubleMLBaseData
from .double_ml_base import DoubleMLBase
from .double_ml_framework import DoubleMLCore as DoubleMLCoreData
from .double_ml_framework import DoubleMLFramework
from .utils._checks import _check_sample_splitting
from .utils._learner import LearnerInfo, LearnerSpec, validate_learner
from .utils.resampling import DoubleMLClusterResampling, DoubleMLResampling


class DoubleMLScalar(DoubleMLBase, ABC):
    """
    Abstract base class for scalar DoubleML models.

    Defines the fit() method for a single parameter based on abstract private methods
    such as nuisance_est(). Solves either linear or non-linear score functions.
    Requires a single treatment column in DoubleMLData.

    This class implements the template method pattern: the fit() method orchestrates
    the estimation process by calling abstract methods that subclasses must implement.

    Parameters
    ----------
    obj_dml_data : DoubleMLBaseData
        The data object for the double machine learning model.
        Must contain exactly one treatment variable.
    score : str, optional
        The score function to use. Default is model-specific.

    Attributes
    ----------
    n_folds : int
        Number of folds for cross-fitting (set via draw_sample_splitting).
    n_rep : int
        Number of repetitions for sample splitting (set via draw_sample_splitting).
    score : str
        The score function being used.
    """

    # Subclasses define all possible learners for the model
    _LEARNER_SPECS: ClassVar[dict[str, LearnerSpec]]

    def __init__(
        self,
        obj_dml_data: DoubleMLBaseData,
        score: str = "default",
    ):
        """
        Initialize DoubleMLScalar.

        Parameters
        ----------
        obj_dml_data : DoubleMLBaseData
            The data object. Must have exactly one treatment column.
        score : str, optional
            The score function to use. Default is 'default'.

        Raises
        ------
        ValueError
            If obj_dml_data contains more than one treatment column.
        """
        # Validate single treatment column
        if len(obj_dml_data.d_cols) != 1:
            raise ValueError(
                f"DoubleMLScalar requires exactly one treatment column. "
                f"Got {len(obj_dml_data.d_cols)}: {obj_dml_data.d_cols}. "
                f"For multiple treatments, use DoubleMLVector."
            )

        # Call parent constructor
        super().__init__(obj_dml_data)

        self._score = score

        # Learner storage: single dict for all learner state
        self._learners: dict[str, LearnerInfo] = {}

        # Resampling parameters (set via draw_sample_splitting)
        self._n_folds: int | None = None
        self._n_folds_per_cluster: int | None = None
        self._n_rep: int | None = None
        self._smpls: list | None = None
        self._smpls_cluster: list | None = None

        # Initialize storage for predictions and results
        self._predictions: dict[str, np.ndarray] | None = None
        self._all_thetas: np.ndarray | None = None
        self._all_ses: np.ndarray | None = None
        self._psi: np.ndarray | None = None
        self._psi_deriv: np.ndarray | None = None
        self._var_scaling_factors: np.ndarray | None = None

        # For iteration (used during fit)
        self._i_rep: int | None = None
        self._i_fold: int | None = None

    # ==================== Properties ====================

    @property
    def n_folds(self) -> int:
        """
        Number of folds for cross-fitting.

        Returns
        -------
        int
            Number of folds.

        Raises
        ------
        ValueError
            If sample splitting has not been performed yet.
        """
        if self._n_folds is None:
            raise ValueError("n_folds not set. Call draw_sample_splitting() first.")
        return self._n_folds

    @property
    def n_rep(self) -> int:
        """
        Number of repetitions for sample splitting.

        Returns
        -------
        int
            Number of repetitions.

        Raises
        ------
        ValueError
            If sample splitting has not been performed yet.
        """
        if self._n_rep is None:
            raise ValueError("n_rep not set. Call draw_sample_splitting() first.")
        return self._n_rep

    @property
    def score(self) -> str:
        """
        The score function being used.

        Returns
        -------
        str
            Score function name.
        """
        return self._score

    @property
    def predictions(self) -> dict[str, np.ndarray]:
        """
        Predictions from nuisance models.

        Returns
        -------
        dict
            Dictionary with predictions for each nuisance component.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._predictions is None:
            raise ValueError("Predictions not available. Call fit() first.")
        return self._predictions

    @property
    def smpls(self) -> list:
        """
        Sample splitting indices used for cross-fitting.

        Returns
        -------
        list
            List of sample splitting indices for each repetition.
        """
        if self._smpls is None:
            raise ValueError("Sample splitting has not been performed. Call draw_sample_splitting() first.")
        return self._smpls

    @property
    def smpls_cluster(self) -> list | None:
        """
        Cluster-based sample splitting indices used for cross-fitting.

        Returns
        -------
        list or None
            List of cluster sample splitting indices for each repetition, or None.

        Raises
        ------
        ValueError
            If cluster data is used but cluster splitting is not available.
        """
        if self._dml_data.is_cluster_data and self._smpls_cluster is None:
            raise ValueError("Cluster sample splitting has not been provided. Call set_sample_splitting() first.")
        return self._smpls_cluster

    @property
    @abstractmethod
    def required_learners(self) -> list[str]:
        """
        Names of the required learners for current configuration.

        Subclasses implement this as a property that returns the learner names
        needed based on the current score function or model configuration.

        Returns
        -------
        list of str
            List of required learner names.
        """
        pass

    @property
    def learners(self) -> dict[str, object]:
        """
        Access registered learner objects by name.

        Returns
        -------
        dict
            Dictionary mapping learner names to estimator instances.
        """
        return {name: info.learner for name, info in self._learners.items()}

    def get_params(self, learner_name: str) -> dict:
        """
        Get parameters of a registered learner.

        Parameters
        ----------
        learner_name : str
            Name of the learner.

        Returns
        -------
        dict
            Dictionary of learner parameters.

        Raises
        ------
        ValueError
            If the learner is not registered.
        """
        if learner_name not in self._learners:
            raise ValueError(f"Learner '{learner_name}' not registered.")
        return self._learners[learner_name].learner.get_params()

    def set_params(self, learner_name: str, **params: object) -> Self:
        """
        Set parameters of a registered learner.

        Parameters
        ----------
        learner_name : str
            Name of the learner.
        **params
            Parameters to set on the learner.

        Returns
        -------
        self : Self
            The estimator with updated learner parameters.

        Raises
        ------
        ValueError
            If the learner is not registered.
        """
        if learner_name not in self._learners:
            raise ValueError(f"Learner '{learner_name}' not registered.")
        self._learners[learner_name].learner.set_params(**params)
        return self

    def _register_learner(self, name: str, learner: object) -> None:
        """
        Validate and register a single learner.

        Parameters
        ----------
        name : str
            Name of the learner (must be in _LEARNER_SPECS).
        learner : object
            The learner instance to register.

        Raises
        ------
        ValueError
            If the learner name is not defined in _LEARNER_SPECS.
        """
        if name not in self._LEARNER_SPECS:
            raise ValueError(f"Learner '{name}' not defined for this model.")

        spec = self._LEARNER_SPECS[name]
        info = validate_learner(
            learner,
            spec,
            binary_outcome=self._dml_data.binary_outcome,
            binary_treatment=self._dml_data.binary_treats.all(),
        )
        self._learners[name] = info

    @abstractmethod
    def set_learners(self, **kwargs: object) -> Self:
        """
        Set the learners for nuisance estimation.

        Subclasses must implement this method with explicit keyword arguments
        for each learner (e.g., ``ml_l``, ``ml_m``, ``ml_g`` for PLR).

        Parameters
        ----------
        **kwargs
            Learner keyword arguments specific to the subclass.

        Returns
        -------
        self : Self
            The estimator with learners set.
        """
        pass

    # ==================== Concrete fit() Method (Template) ====================

    def fit(
        self,
        n_folds: int = 5,
        n_rep: int = 1,
        n_jobs_cv: int | None = None,
        external_predictions: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> Self:
        """
        Estimate the DoubleML model.

        Calls :meth:`draw_sample_splitting` (if not yet done),
        :meth:`fit_nuisance_models`, and :meth:`estimate_causal_parameters`.

        Parameters
        ----------
        n_folds : int, optional
            Number of folds for cross-fitting. Default is 5.
            Only used if sample splitting has not been drawn yet.
        n_rep : int, optional
            Number of repetitions for sample splitting. Default is 1.
            Only used if sample splitting has not been drawn yet.
        n_jobs_cv : int, optional
            Number of jobs for parallel processing during cross-validation.
            Currently not used (reserved for future parallelization).
        external_predictions : dict or None, optional
            Dictionary of pre-computed nuisance predictions to use instead of fitting
            learners. Keys are learner names (e.g., ``'ml_l'``, ``'ml_m'``), values are
            arrays of shape ``(n_obs, n_rep)``. Learners not in the dict are fitted normally.
            Default is ``None``.
        **kwargs : dict
            Additional keyword arguments (for future extensibility).

        Returns
        -------
        self : Self
            The fitted estimator.
        """
        if self._smpls is None:
            self.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
        self.fit_nuisance_models(n_jobs_cv=n_jobs_cv, external_predictions=external_predictions)
        self.estimate_causal_parameters()
        return self

    def fit_nuisance_models(
        self,
        n_jobs_cv: int | None = None,
        external_predictions: dict[str, np.ndarray] | None = None,
    ) -> Self:
        """
        Fit nuisance models via cross-fitting.

        Requires sample splitting to be initialized via :meth:`draw_sample_splitting`
        before calling this method.

        Parameters
        ----------
        n_jobs_cv : int, optional
            Number of jobs for parallel processing during cross-validation.
            Currently not used (reserved for future parallelization).
        external_predictions : dict or None, optional
            Dictionary of pre-computed nuisance predictions. Keys are learner names,
            values are arrays of shape ``(n_obs, n_rep)``. Default is ``None``.

        Returns
        -------
        self : Self
            The estimator with fitted nuisance models and stored predictions.

        Raises
        ------
        ValueError
            If sample splitting has not been initialized.
        """
        if self._smpls is None:
            raise ValueError("Sample splitting has not been initialized. Call draw_sample_splitting() first.")

        if external_predictions is not None:
            self._check_external_predictions(external_predictions)

        # Validate that all required learners are available
        self._check_learners_available(external_predictions)

        # Initialize prediction arrays
        self._predictions = self._initialize_predictions_dict()

        # Pre-fill external predictions
        if external_predictions is not None:
            for key, values in external_predictions.items():
                if key in self._predictions:
                    self._predictions[key][:] = values

        # Cross-fitting loop over repetitions and folds
        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            for i_fold in range(self.n_folds):
                self._i_fold = i_fold

                # Get train/test indices for this fold
                train_idx, test_idx = self._smpls[i_rep][i_fold]

                # Call abstract method - subclass implements nuisance estimation
                self._nuisance_est(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    i_rep=i_rep,
                    i_fold=i_fold,
                    external_predictions=external_predictions,
                )

        return self

    def estimate_causal_parameters(self) -> Self:
        """
        Estimate causal parameters from nuisance predictions.

        Computes score elements, estimates parameters and standard errors, and
        constructs the DoubleMLFramework. Must be called after :meth:`fit_nuisance_models`.

        Returns
        -------
        self : Self
            The estimator with estimated causal parameters.

        Raises
        ------
        ValueError
            If nuisance models have not been fitted yet.
        """
        if self._predictions is None:
            raise ValueError("Predictions not available. Call fit_nuisance_models() first.")

        # Initialize result arrays
        self._initialize_result_arrays()

        # Get score elements - subclass implements
        psi_elements = self._get_score_elements()

        # Estimate causal parameters - from score mixin
        self._est_causal_pars_and_se(psi_elements)

        # Construct framework
        self._framework = self._construct_framework()

        return self

    def draw_sample_splitting(self, n_folds: int = 5, n_rep: int = 1) -> Self:
        """
        Draw sample splitting for cross-fitting.

        Uses DoubleMLResampling to generate K-fold cross-validation splits
        with multiple repetitions.

        Parameters
        ----------
        n_folds : int, optional
            Number of folds for cross-fitting. Default is 5.
        n_rep : int, optional
            Number of repetitions for sample splitting. Default is 1.

        Returns
        -------
        self : Self
            The estimator with initialized sample splits.

        Raises
        ------
        ValueError
            If n_folds or n_rep have invalid values.
        """
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2. Got {n_folds}.")
        if not isinstance(n_rep, int) or n_rep < 1:
            raise ValueError(f"n_rep must be an integer >= 1. Got {n_rep}.")

        if self._dml_data.is_cluster_data:
            self._n_folds_per_cluster = n_folds
            self._n_rep = n_rep
            self._n_folds = n_folds**self._dml_data.n_cluster_vars

            resampler = DoubleMLClusterResampling(
                n_folds=n_folds,
                n_rep=n_rep,
                n_obs=self._n_obs,
                n_cluster_vars=self._dml_data.n_cluster_vars,
                cluster_vars=self._dml_data.cluster_vars,
            )
            self._smpls, self._smpls_cluster = resampler.split_samples()
        else:
            self._n_folds = n_folds
            self._n_folds_per_cluster = None
            self._n_rep = n_rep

            # Create resampler
            resampler = DoubleMLResampling(
                n_folds=n_folds,
                n_rep=n_rep,
                n_obs=self._n_obs,
            )

            # Generate splits
            self._smpls = resampler.split_samples()
            self._smpls_cluster = None

        self._reset_fit_state()

        return self

    def set_sample_splitting(self, all_smpls: list, all_smpls_cluster: list | None = None) -> Self:
        """
        Set the sample splitting for DoubleMLScalar models.

        Parameters
        ----------
        all_smpls : list
            List of tuples (train_ind, test_ind) per fold, or list of lists of tuples
            for repeated sample splitting.
        all_smpls_cluster : list or None
            Nested list for cluster sample splitting. Required for cluster data.
            Default is ``None``.

        Returns
        -------
        self : Self

        Raises
        ------
        TypeError
            If ``all_smpls`` is not a list or if tuple shorthand is used.
        ValueError
            If the partition is invalid or cluster splitting is missing.
        """
        if isinstance(all_smpls, tuple):
            raise TypeError("all_smpls must be a list of folds; tuple shorthand is not supported for DoubleMLScalar.")
        if not isinstance(all_smpls, list):
            raise TypeError(f"all_smpls must be of list type. {str(all_smpls)} of type {str(type(all_smpls))} was passed.")

        smpls, smpls_cluster, n_rep, n_folds = _check_sample_splitting(
            all_smpls,
            all_smpls_cluster,
            self._dml_data,
            self._dml_data.is_cluster_data,
            n_obs=self._n_obs,
        )

        self._smpls = smpls
        self._smpls_cluster = smpls_cluster
        self._n_rep = n_rep
        self._n_folds = n_folds
        if self._dml_data.is_cluster_data:
            n_cluster_vars = self._dml_data.n_cluster_vars
            n_folds_per_cluster = int(round(n_folds ** (1.0 / n_cluster_vars)))
            if n_folds_per_cluster**n_cluster_vars != n_folds:
                raise ValueError(
                    "Invalid cluster sample splitting. n_folds must be a power of n_folds_per_cluster "
                    "for the number of cluster variables."
                )
            self._n_folds_per_cluster = n_folds_per_cluster
        else:
            self._n_folds_per_cluster = None

        self._reset_fit_state()

        return self

    # ==================== Private Helper Methods ====================

    def _initialize_result_arrays(self) -> None:
        """Initialize storage arrays for causal parameter estimation results."""
        n_obs = self._n_obs
        n_rep = self.n_rep
        n_thetas = 1  # Scalar model estimates single parameter

        # Shapes follow framework: (n_thetas, n_rep) for params, (n_obs, n_thetas, n_rep) for scores
        self._all_thetas = np.zeros((n_thetas, n_rep))
        self._all_ses = np.zeros((n_thetas, n_rep))
        self._psi = np.zeros((n_obs, n_thetas, n_rep))
        self._psi_deriv = np.zeros((n_obs, n_thetas, n_rep))

    def _initialize_predictions_dict(self) -> dict[str, np.ndarray]:
        """
        Initialize dictionary for storing predictions.

        Creates a prediction array of shape ``(n_obs, n_rep)`` for each learner
        in :attr:`required_learners`, filled with ``NaN``. Subclasses can override
        this for custom prediction storage.

        Returns
        -------
        dict
            Dictionary mapping learner names to NaN-filled arrays.
        """
        n_obs = self._n_obs
        n_rep = self.n_rep
        return {name: np.full((n_obs, n_rep), np.nan) for name in self.required_learners}

    def _check_external_predictions(self, external_predictions: dict[str, np.ndarray]) -> None:
        """
        Validate external prediction arrays.

        Parameters
        ----------
        external_predictions : dict
            Dictionary of external predictions keyed by learner name.

        Raises
        ------
        TypeError
            If a value is not a numpy array.
        ValueError
            If a value does not match shape (n_obs, n_rep).
        """
        n_obs = self._n_obs
        n_rep = self.n_rep
        required = set(self.required_learners)

        for key, values in external_predictions.items():
            if key not in required:
                raise ValueError(
                    f"External predictions provided for unknown learner '{key}'. " f"Allowed learners: {sorted(required)}."
                )
            if not isinstance(values, np.ndarray):
                raise TypeError(f"External predictions for '{key}' must be a numpy array. Got {type(values).__name__}.")
            if values.shape != (n_obs, n_rep):
                raise ValueError(f"External predictions for '{key}' must have shape ({n_obs}, {n_rep}). Got {values.shape}.")

    def _check_learners_available(self, external_predictions: dict[str, np.ndarray] | None = None) -> None:
        """
        Validate that all required learners are set or covered by external predictions.

        Parameters
        ----------
        external_predictions : dict or None
            External predictions that may cover some learners.

        Raises
        ------
        ValueError
            If a required learner is missing and not covered by external predictions.
        """
        ext_keys = set(external_predictions.keys()) if external_predictions is not None else set()

        for name in self.required_learners:
            if name not in self._learners and name not in ext_keys:
                raise ValueError(
                    f"Learner '{name}' is required but not set and no external predictions provided for it. "
                    f"Call set_learners({name}=...) or provide external_predictions."
                )

    def _construct_framework(self) -> DoubleMLFramework:
        """
        Construct DoubleMLFramework from estimation results.

        Returns
        -------
        DoubleMLFramework
            The framework object with estimation results.
        """
        # Standardize the score function: psi / E[psi_deriv]
        # Both already in framework shape: (n_obs, n_thetas, n_rep)
        scaled_psi = np.divide(self._psi, np.mean(self._psi_deriv, axis=0, keepdims=True))

        cluster_dict = None
        if self._dml_data.is_cluster_data:
            cluster_dict = {
                "smpls": self.smpls,
                "smpls_cluster": self.smpls_cluster,
                "cluster_vars": self._dml_data.cluster_vars,
                "n_folds_per_cluster": self._n_folds_per_cluster,
            }

        # Create data container (no transpose needed - already in framework convention!)
        framework_data = DoubleMLCoreData(
            all_thetas=self._all_thetas,  # (n_thetas, n_rep)
            all_ses=self._all_ses,  # (n_thetas, n_rep)
            var_scaling_factors=self._var_scaling_factors,  # (n_thetas,)
            scaled_psi=scaled_psi,  # (n_obs, n_thetas, n_rep)
            is_cluster_data=self._dml_data.is_cluster_data,
            cluster_dict=cluster_dict,
        )

        # Create and return framework
        return DoubleMLFramework(
            dml_core=framework_data,
            treatment_names=self._dml_data.d_cols,
        )

    def _reset_fit_state(self) -> None:
        """Clear fit-dependent state after changing the sample splitting."""
        self._predictions = None
        self._framework = None
        self._all_thetas = None
        self._all_ses = None
        self._psi = None
        self._psi_deriv = None
        self._var_scaling_factors = None
        self._i_rep = None
        self._i_fold = None

    # ==================== Abstract Methods (Must be Implemented by Subclasses) ====================

    @abstractmethod
    def _nuisance_est(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        i_rep: int,
        i_fold: int,
        external_predictions: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Estimate nuisance parameters for one fold.

        This is the main method subclasses must implement. It should:
        1. Check external_predictions for pre-computed values (skip fitting if present)
        2. Extract training and test data using train_idx and test_idx
        3. Fit nuisance models on training data
        4. Predict on test data
        5. Store predictions in self._predictions

        Parameters
        ----------
        train_idx : np.ndarray
            Indices of training observations for this fold.
        test_idx : np.ndarray
            Indices of test observations for this fold.
        i_rep : int
            Repetition index (0 to n_rep-1).
        i_fold : int
            Fold index (0 to n_folds-1).
        external_predictions : dict or None, optional
            If provided, a dictionary of external predictions. Learners whose names
            appear as keys should not be fitted; their predictions are already
            pre-filled in self._predictions.
        """
        pass

    @abstractmethod
    def _get_score_elements(self) -> dict[str, np.ndarray]:
        """
        Compute score function elements from nuisance predictions.

        This method should use the predictions stored in self._predictions
        to compute the components of the score function.

        Returns
        -------
        dict
            Dictionary with score elements.
            For LinearScoreMixin: {'psi_a': array, 'psi_b': array}
            For NonLinearScoreMixin: model-specific elements

        Notes
        -----
        The score elements should have shape (n_obs, n_rep) for scalar models.

        Example for PLR (linear score):
            psi_a = (D - m_hat) ** 2  # shape: (n_obs, n_rep)
            psi_b = (D - m_hat) * (Y - l_hat)  # shape: (n_obs, n_rep)
            return {'psi_a': psi_a, 'psi_b': psi_b}
        """
        pass

    @abstractmethod
    def _est_causal_pars_and_se(self, psi_elements: dict[str, np.ndarray]) -> None:
        """
        Estimate causal parameters and standard errors from score elements.

        This method is implemented by score mixins (LinearScoreMixin or NonLinearScoreMixin).
        It should:
        1. Compute parameter estimates (self._all_thetas)
        2. Compute standard errors (self._all_ses)
        3. Compute influence function (self._psi)
        4. Compute score derivative (self._psi_deriv)
        5. Compute variance scaling factors (self._var_scaling_factors)

        Parameters
        ----------
        psi_elements : dict
            Dictionary with score function elements from _get_score_elements().

        Notes
        -----
        After this method, all arrays must follow framework convention:
        - self._all_thetas should have shape (n_thetas, n_rep)
        - self._all_ses should have shape (n_thetas, n_rep)
        - self._psi should have shape (n_obs, n_thetas, n_rep)
        - self._psi_deriv should have shape (n_obs, n_thetas, n_rep)
        - self._var_scaling_factors should have shape (n_thetas,)
        """
        pass

    def __str__(self) -> str:
        """
        String representation of the DoubleMLScalar object.

        Returns
        -------
        str
            A formatted string summary of the model.
        """
        class_name = self.__class__.__name__
        header = f"{'=' * 20} {class_name} Object {'=' * 20}"

        info = f"Score function: {self.score}\n"
        if self._n_folds is not None:
            info += f"Resampling: {self._n_folds}-fold CV, {self._n_rep} repetitions\n"

        if self._framework is not None:
            summary_str = str(self.summary)
            return f"{header}\n\n{info}\n{summary_str}"
        else:
            return f"{header}\n\n{info}\nModel not yet fitted. Call fit() first."
