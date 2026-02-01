"""
Mixin for DoubleML models with linear score functions.
"""

from typing import Dict

import numpy as np

from .double_ml_scalar import DoubleMLScalar


class LinearScoreMixin(DoubleMLScalar):
    """
    Mixin for score functions linear in the target parameter.

    This class extends DoubleMLScalar and implements the _est_causal_pars_and_se() method
    for score functions that are linear in the target parameter θ.

    Score form:
        ψ(W; θ, η) = θ · ψ_a(W; η) + ψ_b(W; η)

    The solution has a closed form:
        θ̂ = -E[ψ_b] / E[ψ_a]

    This applies to many common DoubleML models including:
    - Partially Linear Regression (PLR)
    - Partially Linear IV Regression (PLIV)
    - Interactive Regression Model (IRM)
    - Difference-in-Differences (DID)
    - and others

    Notes
    -----
    Subclasses must implement:
    - _nuisance_est(): Estimate nuisance parameters for one fold
    - _get_score_elements(): Return dict with 'psi_a' and 'psi_b' arrays of shape (n_obs, n_rep)
    """

    def _est_causal_pars_and_se(self, psi_elements: Dict[str, np.ndarray]) -> None:
        """
        Estimate causal parameters and standard errors for linear score.

        This method implements the closed-form solution for linear score functions
        and computes standard errors using the influence function.

        All computations use framework convention: (n_obs, n_thetas, n_rep).

        Parameters
        ----------
        psi_elements : dict
            Dictionary with score elements. Must contain:
            - 'psi_a': np.ndarray of shape (n_obs, n_rep)
            - 'psi_b': np.ndarray of shape (n_obs, n_rep)

        Notes
        -----
        Updates the following attributes (all in framework convention):
        - self._all_thetas: Parameter estimates for each repetition (n_thetas=1, n_rep)
        - self._all_ses: Standard errors for each repetition (n_thetas=1, n_rep)
        - self._psi: Influence function values (n_obs, n_thetas=1, n_rep)
        - self._psi_deriv: Score derivative w.r.t. θ (n_obs, n_thetas=1, n_rep)
        - self._var_scaling_factors: Variance scaling factors (n_thetas=1,)
        """
        # Extract score elements
        if "psi_a" not in psi_elements or "psi_b" not in psi_elements:
            raise ValueError(
                "LinearScoreMixin requires 'psi_a' and 'psi_b' in psi_elements. " f"Got keys: {list(psi_elements.keys())}"
            )

        psi_a = psi_elements["psi_a"]  # Shape: (n_obs, n_rep)
        psi_b = psi_elements["psi_b"]  # Shape: (n_obs, n_rep)

        # Validate shapes
        if psi_a.shape != psi_b.shape:
            raise ValueError(f"psi_a and psi_b must have the same shape. " f"Got psi_a: {psi_a.shape}, psi_b: {psi_b.shape}")

        n_obs, n_rep = psi_a.shape

        if n_rep != self.n_rep:
            raise ValueError(f"Score elements have {n_rep} repetitions, but model expects {self.n_rep}.")

        # Compute parameter estimates using closed-form solution
        # θ̂ = -E[ψ_b] / E[ψ_a]
        mean_psi_a = np.mean(psi_a, axis=0)  # (n_rep,)
        mean_psi_b = np.mean(psi_b, axis=0)  # (n_rep,)

        # Check for zero denominator
        if np.any(np.abs(mean_psi_a) < 1e-12):
            raise ValueError(
                "Division by near-zero detected in linear score estimation. "
                "E[psi_a] is very close to zero. This may indicate issues with "
                "the nuisance models or data."
            )

        thetas = -mean_psi_b / mean_psi_a  # (n_rep,)

        # Store parameter estimates in framework shape: (n_thetas=1, n_rep)
        self._all_thetas = thetas[np.newaxis, :]  # (1, n_rep)

        # Compute influence function (score evaluated at θ̂)
        # ψ(W; θ̂, η) = θ̂ · ψ_a + ψ_b
        # Shape: (n_obs, n_rep)
        psi = thetas[np.newaxis, :] * psi_a + psi_b  # Broadcasting: (1, n_rep) * (n_obs, n_rep)

        # Store influence function in framework shape: (n_obs, n_thetas=1, n_rep)
        self._psi = psi[:, np.newaxis, :]  # (n_obs, 1, n_rep)

        # Compute score derivative w.r.t. θ
        # ∂ψ/∂θ = ψ_a
        # Store in framework shape: (n_obs, n_thetas=1, n_rep)
        self._psi_deriv = psi_a[:, np.newaxis, :]  # (n_obs, 1, n_rep)

        # Compute standard errors using sandwich variance estimator
        # Var(θ̂) = E[ψ²] / (n · J²), where J = E[ψ_a]
        # SE = sqrt(E[ψ²]) / (|J| · sqrt(n))
        gamma_hat = np.mean(psi**2, axis=0)  # (n_rep,)
        se = np.sqrt(gamma_hat) / (np.abs(mean_psi_a) * np.sqrt(n_obs))  # (n_rep,)
        self._all_ses = se[np.newaxis, :]  # (1, n_rep)

        # Variance scaling factor: n / J² (used by framework for aggregation)
        self._var_scaling_factors = np.array([n_obs])  # (1,)

    def _compute_score(self, psi_elements: Dict[str, np.ndarray], coef: float) -> np.ndarray:
        """
        Compute the score function value for a given coefficient.

        This is primarily used for verification and diagnostic purposes.

        Parameters
        ----------
        psi_elements : dict
            Dictionary with 'psi_a' and 'psi_b' of shape (n_obs, n_rep).
        coef : float
            The coefficient value at which to evaluate the score.

        Returns
        -------
        np.ndarray
            Score function values, shape (n_obs, n_rep).
        """
        psi_a = psi_elements["psi_a"]
        psi_b = psi_elements["psi_b"]

        return coef * psi_a + psi_b

    def _score_element_names(self) -> list:
        """
        Get the names of score elements for this model.

        Returns
        -------
        list
            List of score element names: ['psi_a', 'psi_b']
        """
        return ["psi_a", "psi_b"]
