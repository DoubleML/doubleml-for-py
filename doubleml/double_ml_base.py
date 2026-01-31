"""
Abstract base class for Double Machine Learning estimators.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data.base_data import DoubleMLBaseData
from .double_ml_framework import DoubleMLFramework


class DoubleMLBase(ABC):
    """
    Abstract base class for Double Machine Learning.

    Provides basic properties and abstract methods, e.g. the fit() method. Mainly handles
    properties and methods which rely on an initialized DoubleMLFramework object.

    This class serves as the foundation for both DoubleMLScalar (single parameter estimation)
    and DoubleMLVector (parameter vector estimation).

    Parameters
    ----------
    obj_dml_data : DoubleMLBaseData
        The data object for the double machine learning model.

    Attributes
    ----------
    framework : DoubleMLFramework
        The DoubleMLFramework object containing estimation results and providing inference methods.
    thetas : np.ndarray
        Estimated parameter values (aggregated across repetitions, shape: (n_thetas,)).
    all_thetas : np.ndarray
        Estimated parameter values for each repetition (shape: (n_thetas, n_rep)).
    ses : np.ndarray
        Standard errors of parameter estimates (aggregated across repetitions, shape: (n_thetas,)).
    all_ses : np.ndarray
        Standard errors for each repetition (shape: (n_thetas, n_rep)).
    summary : pd.DataFrame
        Summary table with estimates, standard errors, confidence intervals, and p-values.
    psi : np.ndarray
        Influence function values (shape: (n_obs, n_thetas, n_rep)).
    smpls : list
        Sample splitting indices used for cross-fitting.
    n_folds : int
        Number of folds used for cross-fitting.
    n_rep : int
        Number of repetitions for sample splitting.
    """

    def __init__(
        self,
        obj_dml_data: DoubleMLBaseData,
    ):
        """
        Initialize DoubleMLBase base class.

        Parameters
        ----------
        obj_dml_data : DoubleMLBaseData
            The data object for the double machine learning model.
        """
        # Validate and store data
        if not isinstance(obj_dml_data, DoubleMLBaseData):
            raise TypeError(f"obj_dml_data must be a DoubleMLBaseData instance. " f"Got {type(obj_dml_data)}.")

        self._dml_data = obj_dml_data
        self._n_obs = obj_dml_data.n_obs

        # Framework is initialized after fit()
        self._framework: Optional[DoubleMLFramework] = None

        # Sample splits are initialized via draw_sample_splitting()
        self._smpls: Optional[List] = None

    # ==================== Properties (Delegating to Framework) ====================

    @property
    def framework(self) -> DoubleMLFramework:
        """
        The DoubleMLFramework object containing estimation results.

        This object is created after calling fit() and provides methods for
        statistical inference (confidence intervals, bootstrap, sensitivity analysis).

        Returns
        -------
        DoubleMLFramework
            The framework object with estimation results.

        Raises
        ------
        ValueError
            If framework is not yet initialized (fit() has not been called).
        """
        if self._framework is None:
            raise ValueError("The framework is not yet initialized. " "Call fit() before accessing estimation results.")
        return self._framework

    @property
    def thetas(self) -> np.ndarray:
        """
        Estimated parameter values (aggregated across repetitions).

        Returns
        -------
        np.ndarray
            Parameter estimates (shape: (n_thetas,)).
        """
        return self.framework.thetas

    @property
    def coef(self) -> np.ndarray:
        """
        Alias for thetas. Estimated parameter values (aggregated across repetitions).

        Returns
        -------
        np.ndarray
            Parameter estimates (shape: (n_thetas,)).
        """
        return self.thetas

    @property
    def all_thetas(self) -> np.ndarray:
        """
        Estimated parameter values for each repetition.

        Returns
        -------
        np.ndarray
            Parameter estimates for all repetitions (shape: (n_thetas, n_rep)).
        """
        return self.framework.all_thetas

    @property
    def all_coef(self) -> np.ndarray:
        """
        Alias for all_thetas. Estimated parameter values for each repetition.

        Returns
        -------
        np.ndarray
            Parameter estimates for all repetitions (shape: (n_thetas, n_rep)).
        """
        return self.all_thetas

    @property
    def se(self) -> np.ndarray:
        """
        Standard errors of parameter estimates (aggregated across repetitions).

        Returns
        -------
        np.ndarray
            Standard errors (shape: (n_thetas,)).
        """
        return self.framework.ses

    @property
    def all_ses(self) -> np.ndarray:
        """
        Standard errors for each repetition.

        Returns
        -------
        np.ndarray
            Standard errors for all repetitions (shape: (n_thetas, n_rep)).
        """
        return self.framework.all_ses

    @property
    def summary(self) -> pd.DataFrame:
        """
        Summary table with estimates, standard errors, confidence intervals, and p-values.

        Returns
        -------
        pd.DataFrame
            Summary statistics for all parameters.
        """
        return self.framework.summary

    @property
    def psi(self) -> np.ndarray:
        """
        Normalized influence function values (scaled score function).

        Returns
        -------
        np.ndarray
            Influence function values (shape: (n_obs, n_thetas, n_rep)).
        """
        return self.framework.scaled_psi

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
            raise ValueError("Sample splitting has not been performed. " "Call draw_sample_splitting() first.")
        return self._smpls

    @property
    def n_obs(self) -> int:
        """
        Number of observations.

        Returns
        -------
        int
            Number of observations in the dataset.
        """
        return self._n_obs

    # ==================== Concrete Methods (Delegating to Framework) ====================

    def confint(self, joint: bool = False, level: float = 0.95) -> pd.DataFrame:
        """
        Confidence intervals for DoubleML models.

        Parameters
        ----------
        joint : bool, optional
            Indicates whether joint confidence intervals are computed.
            Default is False.
        level : float, optional
            The confidence level for the confidence interval.
            Default is 0.95.

        Returns
        -------
        pd.DataFrame
            A DataFrame with confidence intervals.
        """
        return self.framework.confint(joint=joint, level=level)

    def bootstrap(self, method: str = "normal", n_rep_boot: int = 500) -> "DoubleMLBase":
        """
        Multiplier bootstrap for DoubleML models.

        Parameters
        ----------
        method : str, optional
            The bootstrap method ('normal', 'Bayes', or 'wild').
            Default is 'normal'.
        n_rep_boot : int, optional
            The number of bootstrap replications.
            Default is 500.

        Returns
        -------
        self : DoubleMLBase
            The DoubleML estimator with bootstrap results.
        """
        self.framework.bootstrap(method=method, n_rep_boot=n_rep_boot)
        return self

    def p_adjust(self, method: str = "romano-wolf") -> pd.DataFrame:
        """
        Multiple testing adjustment of p-values.

        Parameters
        ----------
        method : str, optional
            The p-value adjustment method. Default is 'romano-wolf'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with adjusted p-values.
        """
        return self.framework.p_adjust(method=method)

    def sensitivity_analysis(
        self,
        cf_y: float = 0.03,
        cf_d: float = 0.03,
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: float = 0.0,
    ) -> Dict:
        """
        Sensitivity analysis for DoubleML models.

        Parameters
        ----------
        cf_y : float, optional
            Percentage of residual variation in outcome explained by unobserved confounders.
            Default is 0.03.
        cf_d : float, optional
            Percentage of residual variation in treatment explained by unobserved confounders.
            Default is 0.03.
        rho : float, optional
            Correlation between unobserved confounders affecting outcome and treatment.
            Default is 1.0.
        level : float, optional
            The confidence level for robustness analysis.
            Default is 0.95.
        null_hypothesis : float, optional
            The null hypothesis value for the parameter.
            Default is 0.0.

        Returns
        -------
        dict
            A dictionary with sensitivity analysis results.
        """
        return self.framework.sensitivity_analysis(
            cf_y=cf_y,
            cf_d=cf_d,
            rho=rho,
            level=level,
            null_hypothesis=null_hypothesis,
        )

    # ==================== Abstract Methods ====================

    @abstractmethod
    def fit(self, **kwargs) -> "DoubleMLBase":
        """
        Estimate the DoubleML model.

        This method must be implemented by subclasses (DoubleMLScalar or DoubleMLVector).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for fitting.

        Returns
        -------
        self : DoubleMLBase
            The fitted DoubleML estimator.
        """
        pass

    @abstractmethod
    def draw_sample_splitting(self) -> "DoubleMLBase":
        """
        Draw sample splitting for cross-fitting.

        This method must be implemented by subclasses to generate sample splits
        using an appropriate resampling strategy.

        Returns
        -------
        self : DoubleMLBase
            The DoubleML estimator with initialized sample splits.
        """
        pass

    def __str__(self) -> str:
        """
        String representation of the DoubleMLBase object.

        Returns
        -------
        str
            A formatted string summary of the model.
        """
        class_name = self.__class__.__name__
        header = f"{'=' * 20} {class_name} Object {'=' * 20}"

        if self._framework is not None:
            summary_str = str(self.summary)
            return f"{header}\n\n{summary_str}"
        else:
            return f"{header}\n\nModel not yet fitted. Call fit() first."

    def __repr__(self) -> str:
        """
        Representation of the DoubleMLBase object.

        Returns
        -------
        str
            A string representation of the object.
        """
        return self.__str__()
