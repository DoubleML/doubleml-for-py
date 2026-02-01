"""
Abstract base class for scalar DoubleML models (single parameter estimation).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Self

import numpy as np

from .data.base_data import DoubleMLBaseData
from .double_ml_base import DoubleMLBase
from .double_ml_framework import DoubleMLCore as DoubleMLCoreData
from .double_ml_framework import DoubleMLFramework
from .utils.resampling import DoubleMLResampling


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

        # Resampling parameters (set via draw_sample_splitting)
        self._n_folds: Optional[int] = None
        self._n_rep: Optional[int] = None
        self._smpls: Optional[List] = None

        # Initialize storage for predictions and results
        self._predictions: Optional[Dict[str, np.ndarray]] = None
        self._all_thetas: Optional[np.ndarray] = None
        self._all_ses: Optional[np.ndarray] = None
        self._psi: Optional[np.ndarray] = None
        self._psi_deriv: Optional[np.ndarray] = None
        self._var_scaling_factors: Optional[np.ndarray] = None

        # For iteration (used during fit)
        self._i_rep: Optional[int] = None
        self._i_fold: Optional[int] = None

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
    def predictions(self) -> Dict[str, np.ndarray]:
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
    def smpls(self) -> List:
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

    # ==================== Concrete fit() Method (Template) ====================

    def fit(
        self,
        n_folds: int = 5,
        n_rep: int = 1,
        n_jobs_cv: Optional[int] = None,
        external_predictions: Optional[Dict[str, np.ndarray]] = None,
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
        n_jobs_cv: Optional[int] = None,
        external_predictions: Optional[Dict[str, np.ndarray]] = None,
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

        self._n_folds = n_folds
        self._n_rep = n_rep

        # Create resampler
        resampler = DoubleMLResampling(
            n_folds=n_folds,
            n_rep=n_rep,
            n_obs=self._n_obs,
        )

        # Generate splits
        self._smpls = resampler.split_samples()

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

    def _initialize_predictions_dict(self) -> Dict[str, np.ndarray]:
        """
        Initialize dictionary for storing predictions.

        Subclasses can override this to define their specific prediction storage structure.

        Returns
        -------
        dict
            Empty dictionary (subclasses should override).
        """
        return {}

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

        # Create data container (no transpose needed - already in framework convention!)
        framework_data = DoubleMLCoreData(
            all_thetas=self._all_thetas,  # (n_thetas, n_rep)
            all_ses=self._all_ses,  # (n_thetas, n_rep)
            var_scaling_factors=self._var_scaling_factors,  # (n_thetas,)
            scaled_psi=scaled_psi,  # (n_obs, n_thetas, n_rep)
            is_cluster_data=False,  # TODO: Add cluster data support
        )

        # Create and return framework
        return DoubleMLFramework(
            dml_core=framework_data,
            treatment_names=self._dml_data.d_cols,
        )

    # ==================== Abstract Methods (Must be Implemented by Subclasses) ====================

    @abstractmethod
    def _nuisance_est(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        i_rep: int,
        i_fold: int,
        external_predictions: Optional[Dict[str, np.ndarray]] = None,
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
    def _get_score_elements(self) -> Dict[str, np.ndarray]:
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
    def _est_causal_pars_and_se(self, psi_elements: Dict[str, np.ndarray]) -> None:
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
