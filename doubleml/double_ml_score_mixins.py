import copy

import numpy as np

import warnings

from scipy.optimize import fmin_l_bfgs_b, root_scalar
from .utils._estimation import _get_bracket_guess

from abc import abstractmethod


class LinearScoreMixin:
    """Mixin class implementing DML estimation for score functions being linear in the target parameter

    Notes
    -----
    The score functions of many DML models (PLR, PLIV, IRM, IIVM) are linear in the parameter :math:`\\theta`, i.e.,

    .. math::

        \\psi(W; \\theta, \\eta) = \\theta \\psi_a(W; \\eta) + \\psi_b(W; \\eta).

    The mixin class :class:`LinearScoreMixin` implements the empirical analog of the moment condition
    :math:`\\mathbb{E}(\\psi(W; \\theta, \\eta)) = 0`, the estimation of :math:`\\theta` by solving the moment condition
    and the estimation of the corresponding asymptotic variance. For details, see the chapters on
    `score functions <https://docs.doubleml.org/stable/guide/scores.html>`_ and on
    `variance estimation <https://docs.doubleml.org/stable/guide/se_confint.html>`_ in the DoubleML user guide.
    """
    _score_type = 'linear'

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']

    def _compute_score(self, psi_elements, coef):
        psi = psi_elements['psi_a'] * coef + psi_elements['psi_b']
        return psi

    def _compute_score_deriv(self, psi_elements, coef):
        return psi_elements['psi_a']

    def _est_coef(self, psi_elements, smpls=None, scaling_factor=None, inds=None):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        if inds is not None:
            psi_a = psi_a[inds]
            psi_b = psi_b[inds]

        if not self._is_cluster_data:
            coef = - np.mean(psi_b) / np.mean(psi_a)
        # for cluster we need the smpls and the scaling factors
        else:
            assert smpls is not None
            assert scaling_factor is not None
            assert inds is None
            # if we have clustered data and dml2 the solution is the root of a weighted sum
            psi_a_subsample_mean = 0.
            psi_b_subsample_mean = 0.
            for i_fold, (_, test_index) in enumerate(smpls):
                psi_a_subsample_mean += scaling_factor[i_fold] * np.sum(psi_a[test_index])
                psi_b_subsample_mean += scaling_factor[i_fold] * np.sum(psi_b[test_index])
            coef = -psi_b_subsample_mean / psi_a_subsample_mean

        return coef


class NonLinearScoreMixin:
    """Mixin class implementing DML estimation for score functions being nonlinear in the target parameter

    Notes
    -----
    The score functions of many DML models (PLR, PLIV, IRM, IIVM) are linear in the parameter. This mixin class
    :class:`NonLinearScoreMixin` allows to use the DML framework for models where the linearity in the parameter is not
    given. The mixin class implements the empirical analog of the moment condition
    :math:`\\mathbb{E}(\\psi(W; \\theta, \\eta)) = 0`, the estimation of :math:`\\theta` via numerical root search of
    the moment condition and the estimation of the corresponding asymptotic variance. For details, see the chapters on
    `score functions <https://docs.doubleml.org/stable/guide/scores.html>`_ and on
    `variance estimation <https://docs.doubleml.org/stable/guide/se_confint.html>`_ in the DoubleML user guide.

    To implement a DML model utilizing the :class:`NonLinearScoreMixin`  class, the abstract methods ``_compute_score``,
    which should implement the evaluation of the score function :math:`\\psi(W; \\theta, \\eta)`, and
    ``_compute_score_deriv``, which should implement the evaluation of the derivative of the score function
    :math:`\\frac{\\partial}{\\partial \\theta} \\psi(W; \\theta, \\eta)`, need to be added model-specifically.
    """
    _score_type = 'nonlinear'
    _coef_start_val = np.nan
    _coef_bounds = None

    @property
    @abstractmethod
    def _score_element_names(self):
        pass

    @abstractmethod
    def _compute_score(self, psi_elements, coef):
        pass

    @abstractmethod
    def _compute_score_deriv(self, psi_elements, coef):
        pass

    def _est_coef(self, psi_elements, smpls=None, scaling_factor=None, inds=None):
        # if the calculation is only done on a subset of observations
        if inds is not None:
            psi_elements = copy.deepcopy(psi_elements)
            for key, value in psi_elements.items():
                psi_elements[key] = value[inds]

        # for cluster we need the smpls and the scaling factors (only check once)
        if self._is_cluster_data:
            assert smpls is not None
            assert scaling_factor is not None
            assert inds is None

        # how to agregate the score and score derivative
        def _aggregate_obs(psi):
            # usually the solution is found as the root of the average score
            if not self._is_cluster_data:
                psi_mean = np.mean(psi)

            # if we have clustered data the solution is the root of a weighted sum
            else:
                psi_mean = 0.
                for i_fold, (_, test_index) in enumerate(smpls):
                    psi_mean += scaling_factor[i_fold] * np.sum(psi[test_index])

            return psi_mean

        # calculation of the score for a parameter theta
        def score(theta):
            psi = self._compute_score(psi_elements, theta)

            return _aggregate_obs(psi)

        # calculation of the score derivative for a parameter theta
        def score_deriv(theta):
            psi_deriv = self._compute_score_deriv(psi_elements, theta)

            return _aggregate_obs(psi_deriv)

        if self._coef_bounds is None:
            bounded = False
        else:
            bounded = (self._coef_bounds[0] > -np.inf) & (self._coef_bounds[1] < np.inf)

        if not bounded:
            root_res = root_scalar(score,
                                   x0=self._coef_start_val,
                                   fprime=score_deriv,
                                   method='newton')
            theta_hat = root_res.root
            if not root_res.converged:
                score_val = score(theta_hat)
                warnings.warn('Could not find a root of the score function.\n '
                              f'Flag: {root_res.flag}.\n'
                              f'Score value found is {score_val} '
                              f'for parameter theta equal to {theta_hat}.')
        else:
            signs_different, bracket_guess = _get_bracket_guess(score, self._coef_start_val, self._coef_bounds)

            if signs_different:
                root_res = root_scalar(score,
                                       bracket=bracket_guess,
                                       method='brentq')
                theta_hat = root_res.root
            else:
                # try to find an alternative start value
                def score_squared(theta):
                    res = np.power(np.mean(self._compute_score(psi_elements, theta)), 2)
                    return res
                # def score_squared_deriv(theta, inds):
                #     res = 2 * np.mean(self._compute_score(psi_elements, theta, inds)) * \
                #           np.mean(self._compute_score_deriv(psi_elements, theta, inds))
                #     return res
                alt_coef_start, _, _ = fmin_l_bfgs_b(score_squared,
                                                     self._coef_start_val,
                                                     approx_grad=True,
                                                     bounds=[self._coef_bounds])
                signs_different, bracket_guess = _get_bracket_guess(score, alt_coef_start, self._coef_bounds)

                if signs_different:
                    root_res = root_scalar(score,
                                           bracket=bracket_guess,
                                           method='brentq')
                    theta_hat = root_res.root
                else:
                    score_val_sign = np.sign(score(alt_coef_start))
                    if score_val_sign > 0:
                        theta_hat, score_val, _ = fmin_l_bfgs_b(score,
                                                                self._coef_start_val,
                                                                approx_grad=True,
                                                                bounds=[self._coef_bounds])
                        warnings.warn('Could not find a root of the score function.\n '
                                      f'Minimum score value found is {score_val} '
                                      f'for parameter theta equal to {theta_hat}.\n '
                                      'No theta found such that the score function evaluates to a negative value.')
                    else:
                        def neg_score(theta):
                            res = - np.mean(self._compute_score(psi_elements, theta))
                            return res
                        theta_hat, neg_score_val, _ = fmin_l_bfgs_b(neg_score,
                                                                    self._coef_start_val,
                                                                    approx_grad=True,
                                                                    bounds=[self._coef_bounds])
                        warnings.warn('Could not find a root of the score function. '
                                      f'Maximum score value found is {-1*neg_score_val} '
                                      f'for parameter theta equal to {theta_hat}. '
                                      'No theta found such that the score function evaluates to a positive value.')

        return theta_hat
