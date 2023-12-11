from abc import ABC, abstractmethod


class DoubleMLBasic(ABC):
    """Basic Double Machine Learning Class for single estimate."""

    def __init__(
        self,
        psi_elements,
    ):
        # intialize arrays
        self._psi_elements = psi_elements
        self._score_type = None

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

    @abstractmethod
    def estimate_theta(self, psi_elements):
        pass

    @abstractmethod
    def _compute_score(self, psi_elements, coef):
        pass

    @abstractmethod
    def _compute_score_deriv(self, psi_elements, coef):
        pass
