import numpy as np
from abc import ABC, abstractmethod

from doubleml._utils_base import _var_est, _aggregate_thetas_and_ses


class DoubleMLBase(ABC):
    """Base Double Machine Learning Class for single estimate."""

    def __init__(
        self,
        psi_elements,
    ):
        # scores and parameters
        self._psi_elements = psi_elements
        self._score_type = None
        self._theta = None
        self._se = None
        self._n_obs = psi_elements['psi_a'].shape[0]
        self._n_rep = psi_elements['psi_a'].shape[1]

        # initalize arrays
        self._all_thetas = np.full(self._n_rep, np.nan)
        self._all_ses = np.full(self._n_rep, np.nan)
        self._var_scaling_factor = np.full(1, np.nan)
        self._psi = np.full((self._n_obs, self._n_rep), np.nan)
        self._psi_deriv = np.full((self._n_obs, self._n_rep), np.nan)

    @property
    def psi_elements(self):
        """
        Values of the score function components;
        For models (e.g., PLR, IRM, PLIV, IIVM) with linear score (in the parameter) a dictionary with entries ``psi_a``
        and ``psi_b`` for :math:`\\psi_a(W; \\eta)` and :math:`\\psi_b(W; \\eta)`.
        """
        return self._psi_elements

    @property
    def score_type(self):
        """
        Type of the score function. For models (e.g., PLR, IRM, PLIV, IIVM) with linear score (in the parameter) the
        type is ``linear``.
        """
        return self._score_type

    @property
    def theta(self):
        """
        Estimated target parameter.
        """
        return self._theta

    @property
    def all_thetas(self):
        """
        Estimated target parameters for each repetition.
        """
        return self._all_thetas

    @property
    def se(self):
        """
        Estimated standard error.
        """
        return self._se

    @property
    def all_ses(self):
        """
        Estimated standard error for each repetition.
        """
        return self._all_ses

    @property
    def n_rep(self):
        """
        Number of repetitions.
        """
        return self._n_rep

    @property
    def n_obs(self):
        """
        Number of observations.
        """
        return self._n_obs

    @property
    def psi(self):
        """
        Values of the score function.
        """
        return self._psi

    @property
    def psi_deriv(self):
        """
        Values of the score function derivative.
        """
        return self._psi_deriv

    @abstractmethod
    def _solve_score(self, psi_elements):
        pass

    @abstractmethod
    def _compute_score(self, psi_elements, coef):
        pass

    @abstractmethod
    def _compute_score_deriv(self, psi_elements, coef):
        pass

    def estimate_theta(self, aggregation_method='median'):
        for i_rep in range(self._n_rep):
            self._all_thetas[i_rep] = self._solve_score(self._psi_elements, i_rep)

            # compute score and derivative
            self._psi[:, i_rep] = self._compute_score(
                psi_elements=self._psi_elements,
                theta=self._all_thetas[i_rep],
                i_rep=i_rep
            )
            self._psi_deriv[:, i_rep] = self._compute_score_deriv(
                psi_elements=self._psi_elements,
                theta=self._all_thetas[i_rep],
                i_rep=i_rep
            )

            # variance estimation
            var_estimate, var_scaling_factor = _var_est(
                psi=self._psi[:, i_rep],
                psi_deriv=self._psi_deriv[:, i_rep]
            )
            self._var_scaling_factor = var_scaling_factor
            self._all_ses[i_rep] = np.sqrt(var_estimate)

        # aggregate estimates
        self._theta, self._se = _aggregate_thetas_and_ses(
            all_thetas=self._all_thetas,
            all_ses=self._all_ses,
            var_scaling_factor=self._var_scaling_factor,
            aggregation_method=aggregation_method
        )

        return self
