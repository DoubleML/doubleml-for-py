import numpy as np
from numpy.linalg import inv

from .double_ml_base import DoubleMLBase


class DoubleMLBaseLinear(DoubleMLBase):
    """Class implementing DML estimation for score functions being linear in the target parameter

    Notes
    -----
    The score functions of many DML models (PLR, PLIV, IRM, IIVM) are linear in the parameter :math:`\\theta`, i.e.,

    .. math::

        \\psi(W; \\theta, \\eta) = \\theta \\psi_a(W; \\eta) + \\psi_b(W; \\eta).

    The  class :class:`DoubleMLBasicLinear(` implements the empirical analog of the moment condition
    :math:`\\mathbb{E}(\\psi(W; \\theta, \\eta)) = 0`, the estimation of :math:`\\theta` by solving the moment condition
    and the estimation of the corresponding asymptotic variance. For details, see the chapters on
    `score functions <https://docs.doubleml.org/stable/guide/scores.html>`_ and on
    `variance estimation <https://docs.doubleml.org/stable/guide/se_confint.html>`_ in the DoubleML user guide.
    """
    def __init__(
        self,
        psi_elements,
        n_obs,
        n_thetas=1,
        n_rep=1,
    ):
        super().__init__(
            psi_elements,
            n_obs=n_obs,
            n_thetas=n_thetas,
            n_rep=n_rep,
        )
        self._score_type = 'linear'

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']

    def _compute_score(self, psi_elements, thetas, i_rep):
        psi_a = psi_elements['psi_a'][:, :, i_rep]
        psi_b = psi_elements['psi_b'][:, :, i_rep]
        psi = psi_a * thetas + psi_b
        return psi

    def _compute_score_deriv(self, psi_elements, thetas, i_rep):
        return psi_elements['psi_a'][:, :, i_rep]

    def _solve_score(self, psi_elements, i_rep):
        psi_a = psi_elements['psi_a'][:, :, i_rep]
        psi_b = psi_elements['psi_b'][:, :, i_rep]

        if self._n_thetas == 1:
            thetas = - np.mean(psi_b, axis=0) / np.mean(psi_a, axis=0)
        else:
            thetas = - np.matmul(inv(np.mean(psi_a, axis=0)), np.mean(psi_b, axis=0))

        return thetas
