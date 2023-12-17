import numpy as np
from abc import ABC, abstractmethod

from doubleml._utils_base import _initialize_arrays, _var_est, _aggregate_thetas_and_ses


class DoubleMLBase(ABC):
    """Base Double Machine Learning Class."""

    def __init__(
        self,
        psi_elements,
        n_obs,
        n_thetas=1,
        n_rep=1,
    ):
        # scores and parameters
        self._psi_elements = psi_elements
        self._score_type = None
        self._n_obs = n_obs
        self._n_rep = n_rep
        self._n_thetas = n_thetas

        # initalize arrays
        self._thetas, self._ses, self._all_thetas, self._all_ses, self._var_scaling_factors, \
            self._psi, self._psi_deriv = _initialize_arrays(self._n_thetas, self._n_rep, self._n_obs)

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
    def thetas(self):
        """
        Estimated target parameters.
        """
        return self._thetas

    @property
    def all_thetas(self):
        """
        Estimated target parameters for each repetition.
        """
        return self._all_thetas

    @property
    def ses(self):
        """
        Estimated standard errors.
        """
        return self._ses

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
    def var_scaling_factors(self):
        """
        Scaling factors for the asymptotic variance.
        """
        return self._var_scaling_factors

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
    def _compute_score(self, psi_elements, thetas, i_rep):
        pass

    @abstractmethod
    def _compute_score_deriv(self, psi_elements, thetas, i_rep):
        pass

    def estimate_thetas(self, aggregation_method='median'):
        for i_rep in range(self._n_rep):
            self._all_thetas[:, i_rep] = self._solve_score(self._psi_elements, i_rep)

            # compute score and derivative
            self._psi[:, :, i_rep] = self._compute_score(
                psi_elements=self._psi_elements,
                thetas=self._all_thetas[:, i_rep],
                i_rep=i_rep
            )
            self._psi_deriv[:, :, i_rep] = self._compute_score_deriv(
                psi_elements=self._psi_elements,
                thetas=self._all_thetas[:, i_rep],
                i_rep=i_rep
            )

            # variance estimation
            var_estimates, var_scaling_factors = _var_est(
                psi=self._psi[:, :, i_rep],
                psi_deriv=self._psi_deriv[:, :, i_rep]
            )

            # TODO: check if var_scaling_factor is the same for all target parameters
            self._var_scaling_factors[:] = var_scaling_factors
            self._all_ses[:, i_rep] = np.sqrt(var_estimates)

        # aggregate estimates
        self._thetas, self._ses = _aggregate_thetas_and_ses(
            all_thetas=self._all_thetas,
            all_ses=self._all_ses,
            var_scaling_factors=self._var_scaling_factors,
            aggregation_method=aggregation_method
        )

        return self
