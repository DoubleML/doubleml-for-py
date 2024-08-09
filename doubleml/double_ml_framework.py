import numpy as np
import pandas as pd
import copy

from scipy.stats import norm
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import multipletests

from .utils._estimation import _draw_weights, _aggregate_coefs_and_ses, _var_est
from .utils._checks import _check_bootstrap, _check_framework_compatibility, _check_in_zero_one, \
    _check_float, _check_integer, _check_bool, _check_benchmarks
from .utils._descriptive import generate_summary
from .utils._plots import _sensitivity_contour_plot


class DoubleMLFramework():
    """Double Machine Learning Framework to combine DoubleML classes and compute confidendence intervals.

    Parameters
    ----------
    doubleml_dict : :dict
        A dictionary providing the estimated parameters and normalized scores. Keys have to be 'thetas', 'ses',
        'all_thetas', 'all_ses', 'var_scaling_factors' and 'scaled_psi'.
        Values have to be numpy arrays with the corresponding shapes.

    """

    def __init__(
            self,
            doubleml_dict=None,
    ):
        self._is_cluster_data = False

        # check input
        if not isinstance(doubleml_dict, dict):
            raise TypeError('doubleml_dict must be a dictionary.')
        expected_keys = ['thetas', 'ses', 'all_thetas', 'all_ses', 'var_scaling_factors', 'scaled_psi']
        if not all(key in doubleml_dict.keys() for key in expected_keys):
            raise ValueError('The dict must contain the following keys: ' + ', '.join(expected_keys))

        # set scores and parameters
        self._n_thetas = doubleml_dict['scaled_psi'].shape[1]
        self._n_rep = doubleml_dict['scaled_psi'].shape[2]
        self._n_obs = doubleml_dict['scaled_psi'].shape[0]

        self._thetas = doubleml_dict['thetas']
        self._ses = doubleml_dict['ses']
        self._all_thetas = doubleml_dict['all_thetas']
        self._all_ses = doubleml_dict['all_ses']
        self._var_scaling_factors = doubleml_dict['var_scaling_factors']
        self._scaled_psi = doubleml_dict['scaled_psi']

        # initialize cluster data
        self._check_and_set_cluster_data(doubleml_dict)

        # initialize sensitivity analysis
        self._check_and_set_sensitivity_elements(doubleml_dict)

        # check if all sizes match
        self._check_framework_shapes()

        self._treatment_names = None
        if 'treatment_names' in doubleml_dict.keys():
            self._check_treatment_names(doubleml_dict['treatment_names'])
            self._treatment_names = doubleml_dict['treatment_names']

        # initialize bootstrap distribution
        self._boot_t_stat = None
        self._boot_method = None
        self._n_rep_boot = None

    @property
    def n_thetas(self):
        """
        Number of target parameters.
        """
        return self._n_thetas

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
    def thetas(self):
        """
        Estimated target parameters (shape (``n_thetas``,)).
        """
        return self._thetas

    @property
    def all_thetas(self):
        """
        Estimated target parameters for each repetition (shape (``n_thetas``, ``n_rep``)).
        """
        return self._all_thetas

    @property
    def ses(self):
        """
        Estimated standard errors (shape (``n_thetas``,)).
        """
        return self._ses

    @property
    def all_ses(self):
        """
        Estimated standard errors for each repetition (shape (``n_thetas``, ``n_rep``)).
        """
        return self._all_ses

    @property
    def t_stats(self):
        """
        t-statistics for the causal parameter(s) (shape (``n_thetas``,)).
        """
        return self._thetas / self._ses

    @property
    def all_t_stats(self):
        """
        t-statistics for the causal parameter(s) for each repetition (shape (``n_thetas``, ``n_rep``)).
        """
        return self._all_thetas / self._all_ses

    @property
    def pvals(self):
        """
        p-values for the causal parameter(s) (shape (``n_thetas``,)).
        """
        # aggregate p-values according to Definition 4.2 https://arxiv.org/abs/1712.04802
        pvals = np.median(self.all_pvals, axis=1)
        return pvals

    @property
    def all_pvals(self):
        """
        p-values for the causal parameter(s) for each repetition (shape (``n_thetas``, ``n_rep``)).
        """
        all_pvals = 2 * (1 - norm.cdf(np.abs(self.all_t_stats)))
        return all_pvals

    @property
    def scaled_psi(self):
        """
        Normalized scores (shape (``n_obs``, ``n_thetas``, ``n_rep``)).
        """
        return self._scaled_psi

    @property
    def var_scaling_factors(self):
        """
        Variance scaling factors (shape (``n_thetas``,)).
        """
        return self._var_scaling_factors

    @property
    def n_rep_boot(self):
        """
        The number of bootstrap replications.
        """
        return self._n_rep_boot

    @property
    def boot_method(self):
        """
        The method to construct the bootstrap replications.
        """
        return self._boot_method

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`bootstrap`
         (shape (``n_rep_boot``, ``n_thetas``, ``n_rep``)).
        """
        return self._boot_t_stat

    @property
    def sensitivity_elements(self):
        """
        Values of the sensitivity components.
        If available (e.g., PLR, IRM) a dictionary with entries ``sigma2``, ``nu2``, ``psi_sigma2``, ``psi_nu2``
        and ``riesz_rep``.
        """
        return self._sensitivity_elements

    @property
    def sensitivity_params(self):
        """
        Values of the sensitivity parameters after calling :meth:`sesitivity_analysis`;
        If available (e.g., PLR, IRM) a dictionary with entries ``theta``, ``se``, ``ci``, ``rv``
        and ``rva``.
        """
        return self._sensitivity_params

    @property
    def treatment_names(self):
        """
        Names of the treatments.
        """
        return self._treatment_names

    @treatment_names.setter
    def treatment_names(self, value):
        self._check_treatment_names(value)
        self._treatment_names = value

    @property
    def summary(self):
        """
        A summary for the estimated causal parameters ``thetas``.
        """
        ci = self.confint()
        df_summary = generate_summary(self.thetas, self.ses, self.t_stats,
                                      self.pvals, ci, self._treatment_names)
        return df_summary

    @property
    def sensitivity_summary(self):
        """
        Returns a summary for the sensitivity analysis after calling :meth:`sensitivity_analysis`.

        Returns
        -------
        res : str
            Summary for the sensitivity analysis.
        """
        header = '================== Sensitivity Analysis ==================\n'
        if self.sensitivity_params is None:
            res = header + 'Apply sensitivity_analysis() to generate sensitivity_summary.'
        else:
            sig_level = f'Significance Level: level={self.sensitivity_params["input"]["level"]}\n'
            scenario_params = f'Sensitivity parameters: cf_y={self.sensitivity_params["input"]["cf_y"]}; ' \
                              f'cf_d={self.sensitivity_params["input"]["cf_d"]}, ' \
                              f'rho={self.sensitivity_params["input"]["rho"]}'

            theta_and_ci_col_names = ['CI lower', 'theta lower', ' theta', 'theta upper', 'CI upper']
            theta_and_ci = np.transpose(np.vstack((self.sensitivity_params['ci']['lower'],
                                                   self.sensitivity_params['theta']['lower'],
                                                   self.thetas,
                                                   self.sensitivity_params['theta']['upper'],
                                                   self.sensitivity_params['ci']['upper'])))
            df_theta_and_ci = pd.DataFrame(theta_and_ci,
                                           columns=theta_and_ci_col_names,
                                           index=self.treatment_names)
            theta_and_ci_summary = str(df_theta_and_ci)

            rvs_col_names = ['H_0', 'RV (%)', 'RVa (%)']
            rvs = np.transpose(np.vstack((self.sensitivity_params['rv'],
                                          self.sensitivity_params['rva']))) * 100

            df_rvs = pd.DataFrame(np.column_stack((self.sensitivity_params["input"]["null_hypothesis"], rvs)),
                                  columns=rvs_col_names,
                                  index=self.treatment_names)
            rvs_summary = str(df_rvs)

            res = header + \
                '\n------------------ Scenario          ------------------\n' + \
                sig_level + scenario_params + '\n' + \
                '\n------------------ Bounds with CI    ------------------\n' + \
                theta_and_ci_summary + '\n' + \
                '\n------------------ Robustness Values ------------------\n' + \
                rvs_summary

        return res

    def __add__(self, other):

        if isinstance(other, DoubleMLFramework):
            # internal consistency check
            self._check_framework_shapes()
            other._check_framework_shapes()
            _check_framework_compatibility(self, other, check_treatments=True)

            all_thetas = self._all_thetas + other._all_thetas
            scaled_psi = self._scaled_psi + other._scaled_psi

            # check if var_scaling_factors are the same
            assert np.allclose(self._var_scaling_factors, other._var_scaling_factors)
            var_scaling_factors = self._var_scaling_factors

            # compute standard errors
            sigma2_hat = np.divide(
                np.mean(np.square(scaled_psi), axis=0),
                var_scaling_factors.reshape(-1, 1))
            all_ses = np.sqrt(sigma2_hat)
            thetas, ses = _aggregate_coefs_and_ses(all_thetas, all_ses, var_scaling_factors)

            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': self._is_cluster_data,
                'cluster_dict': self._cluster_dict,
            }

            # sensitivity combination only available for same outcome and cond. expectation (e.g. IRM)
            if self._sensitivity_implemented and other._sensitivity_implemented:
                nu2_score_element = self._sensitivity_elements['psi_nu2'] + other._sensitivity_elements['psi_nu2'] - \
                     np.multiply(2.0, np.multiply(self._sensitivity_elements['riesz_rep'],
                                                  self._sensitivity_elements['riesz_rep']))
                nu2 = np.mean(nu2_score_element, axis=0, keepdims=True)
                psi_nu2 = nu2_score_element - nu2

                sensitivity_elements = {
                    'sigma2': self._sensitivity_elements['sigma2'],
                    'nu2': nu2,
                    'psi_sigma2': self._sensitivity_elements['psi_sigma2'],
                    'psi_nu2': psi_nu2,
                    'riesz_rep': self._sensitivity_elements['riesz_rep'] + other._sensitivity_elements['riesz_rep'],
                }
                doubleml_dict['sensitivity_elements'] = sensitivity_elements

            new_obj = DoubleMLFramework(doubleml_dict)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        if isinstance(other, DoubleMLFramework):
            # internal consistency check
            self._check_framework_shapes()
            other._check_framework_shapes()
            _check_framework_compatibility(self, other, check_treatments=True)

            all_thetas = self._all_thetas - other._all_thetas
            scaled_psi = self._scaled_psi - other._scaled_psi

            # check if var_scaling_factors are the same
            assert np.allclose(self._var_scaling_factors, other._var_scaling_factors)
            var_scaling_factors = self._var_scaling_factors

            # compute standard errors
            sigma2_hat = np.divide(
                np.mean(np.square(scaled_psi), axis=0),
                var_scaling_factors.reshape(-1, 1))
            all_ses = np.sqrt(sigma2_hat)
            thetas, ses = _aggregate_coefs_and_ses(all_thetas, all_ses, var_scaling_factors)

            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': self._is_cluster_data,
                'cluster_dict': self._cluster_dict,
            }

            # sensitivity combination only available for same outcome and cond. expectation (e.g. IRM)
            if self._sensitivity_implemented and other._sensitivity_implemented:
                nu2_score_element = self._sensitivity_elements['psi_nu2'] - other._sensitivity_elements['psi_nu2'] + \
                     np.multiply(2.0, np.multiply(self._sensitivity_elements['riesz_rep'],
                                                  self._sensitivity_elements['riesz_rep']))
                nu2 = np.mean(nu2_score_element, axis=0, keepdims=True)
                psi_nu2 = nu2_score_element - nu2

                sensitivity_elements = {
                    'sigma2': self._sensitivity_elements['sigma2'],
                    'nu2': nu2,
                    'psi_sigma2': self._sensitivity_elements['psi_sigma2'],
                    'psi_nu2': psi_nu2,
                    'riesz_rep': self._sensitivity_elements['riesz_rep'] - other._sensitivity_elements['riesz_rep'],
                }
                doubleml_dict['sensitivity_elements'] = sensitivity_elements

            new_obj = DoubleMLFramework(doubleml_dict)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __rsub__(self, other):
        return self.__sub__(other)

    # TODO: Restrict to linear?
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            thetas = np.multiply(other, self._thetas)
            all_thetas = np.multiply(other, self._all_thetas)

            var_scaling_factors = self._var_scaling_factors
            ses = np.multiply(other, self._ses)
            all_ses = np.multiply(other, self._all_ses)
            scaled_psi = np.multiply(other, self._scaled_psi)

            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': self._is_cluster_data,
                'cluster_dict': self._cluster_dict,
            }

            # sensitivity combination only available for linear models
            if self._sensitivity_implemented:
                nu2_score_element = np.multiply(np.square(other), self._sensitivity_elements['psi_nu2'])
                nu2 = np.mean(nu2_score_element, axis=0, keepdims=True)
                psi_nu2 = nu2_score_element - nu2

                sensitivity_elements = {
                    'sigma2': self._sensitivity_elements['sigma2'],
                    'nu2': nu2,
                    'psi_sigma2': self._sensitivity_elements['psi_sigma2'],
                    'psi_nu2': psi_nu2,
                    'riesz_rep': np.multiply(other, self._sensitivity_elements['riesz_rep']),
                }
                doubleml_dict['sensitivity_elements'] = sensitivity_elements

            new_obj = DoubleMLFramework(doubleml_dict)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __rmul__(self, other):
        return self.__mul__(other)

    def _calc_sensitivity_analysis(self, cf_y, cf_d, rho, level):
        if not self._sensitivity_implemented:
            raise NotImplementedError('Sensitivity analysis is not implemented for this model.')

        # input checks
        _check_in_zero_one(cf_y, 'cf_y', include_one=False)
        _check_in_zero_one(cf_d, 'cf_d', include_one=False)
        if not isinstance(rho, float):
            raise TypeError(f'rho must be of float type. '
                            f'{str(rho)} of type {str(type(rho))} was passed.')
        _check_in_zero_one(abs(rho), 'The absolute value of rho')
        _check_in_zero_one(level, 'The confidence level', include_zero=False, include_one=False)

        # set elements for readability
        sigma2 = self.sensitivity_elements['sigma2']
        nu2 = self.sensitivity_elements['nu2']
        psi_sigma = self.sensitivity_elements['psi_sigma2']
        psi_nu = self.sensitivity_elements['psi_nu2']
        psi_scaled = self._scaled_psi

        if (np.any(sigma2 < 0)) | (np.any(nu2 < 0)):
            raise ValueError('sensitivity_elements sigma2 and nu2 have to be positive. '
                             f"Got sigma2 {str(sigma2)} and nu2 {str(nu2)}. "
                             'Most likely this is due to low quality learners (especially propensity scores).')

        # elementwise operations
        confounding_strength = np.multiply(np.abs(rho), np.sqrt(np.multiply(cf_y, np.divide(cf_d, 1.0-cf_d))))
        sensitivity_scaling = np.sqrt(np.multiply(sigma2, nu2))

        # sigma2 and nu2 are of shape (1, n_thetas, n_rep), whereas the all_thetas is of shape (n_thetas, n_rep)
        all_theta_lower = self.all_thetas - np.multiply(np.squeeze(sensitivity_scaling, axis=0), confounding_strength)
        all_theta_upper = self.all_thetas + np.multiply(np.squeeze(sensitivity_scaling, axis=0), confounding_strength)

        psi_variances = np.multiply(sigma2, psi_nu) + np.multiply(nu2, psi_sigma)
        psi_bias = np.multiply(np.divide(confounding_strength, np.multiply(2.0, sensitivity_scaling)), psi_variances)
        psi_lower = psi_scaled - psi_bias
        psi_upper = psi_scaled + psi_bias

        # shape (n_thetas, n_reps); includes scaling with n^{-1/2}
        all_sigma_lower = np.full_like(all_theta_lower, fill_value=np.nan)
        all_sigma_upper = np.full_like(all_theta_upper, fill_value=np.nan)

        for i_rep in range(self.n_rep):
            for i_theta in range(self.n_thetas):

                if not self._is_cluster_data:
                    smpls = None
                    cluster_vars = None
                    smpls_cluster = None
                    n_folds_per_cluster = None
                else:
                    smpls = self._cluster_dict['smpls'][i_rep]
                    cluster_vars = self._cluster_dict['cluster_vars']
                    smpls_cluster = self._cluster_dict['smpls_cluster'][i_rep]
                    n_folds_per_cluster = self._cluster_dict['n_folds_per_cluster']

                sigma2_lower_hat, _ = _var_est(psi=psi_lower[:, i_theta, i_rep],
                                               psi_deriv=np.ones_like(psi_lower[:, i_theta, i_rep]),
                                               smpls=smpls,
                                               is_cluster_data=self._is_cluster_data,
                                               cluster_vars=cluster_vars,
                                               smpls_cluster=smpls_cluster,
                                               n_folds_per_cluster=n_folds_per_cluster)
                sigma2_upper_hat, _ = _var_est(psi=psi_upper[:, i_theta, i_rep],
                                               psi_deriv=np.ones_like(psi_upper[:, i_theta, i_rep]),
                                               smpls=smpls,
                                               is_cluster_data=self._is_cluster_data,
                                               cluster_vars=cluster_vars,
                                               smpls_cluster=smpls_cluster,
                                               n_folds_per_cluster=n_folds_per_cluster)

                all_sigma_lower[i_theta, i_rep] = np.sqrt(sigma2_lower_hat)
                all_sigma_upper[i_theta, i_rep] = np.sqrt(sigma2_upper_hat)

        # aggregate coefs and ses over n_rep
        theta_lower, sigma_lower = _aggregate_coefs_and_ses(all_theta_lower, all_sigma_lower, self._var_scaling_factors)
        theta_upper, sigma_upper = _aggregate_coefs_and_ses(all_theta_upper, all_sigma_upper, self._var_scaling_factors)

        # per repetition confidence intervals
        quant = norm.ppf(level)
        all_ci_lower = all_theta_lower - np.multiply(quant, all_sigma_lower)
        all_ci_upper = all_theta_upper + np.multiply(quant, all_sigma_upper)

        ci_lower = np.median(all_ci_lower, axis=1)
        ci_upper = np.median(all_ci_upper, axis=1)

        theta_dict = {'lower': theta_lower,
                      'upper': theta_upper}

        se_dict = {'lower': sigma_lower,
                   'upper': sigma_upper}

        ci_dict = {'lower': ci_lower,
                   'upper': ci_upper}

        res_dict = {'theta': theta_dict,
                    'se': se_dict,
                    'ci': ci_dict}

        return res_dict

    def _calc_robustness_value(self, null_hypothesis, level, rho, idx_treatment):
        _check_float(null_hypothesis, "null_hypothesis")
        _check_integer(idx_treatment, "idx_treatment", lower_bound=0, upper_bound=self._n_thetas-1)

        # check which side is relvant
        bound = 'upper' if (null_hypothesis > self.thetas[idx_treatment]) else 'lower'

        # minimize the square to find boundary solutions
        def rv_fct(value, param):
            res = self._calc_sensitivity_analysis(cf_y=value,
                                                  cf_d=value,
                                                  rho=rho,
                                                  level=level)[param][bound][idx_treatment] - null_hypothesis
            return np.square(res)

        rv = minimize_scalar(rv_fct, bounds=(0, 0.9999), method='bounded', args=('theta', )).x
        rva = minimize_scalar(rv_fct, bounds=(0, 0.9999), method='bounded', args=('ci', )).x

        return rv, rva

    def sensitivity_analysis(self, cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0):
        """
        Performs a sensitivity analysis to account for unobserved confounders.

        The evaluated scenario is stored as a dictionary in the property ``sensitivity_params``.

        Parameters
        ----------
        cf_y : float
            Percentage of the residual variation of the outcome explained by latent/confounding variables.
            Default is ``0.03``.

        cf_d : float
            Percentage gains in the variation of the Riesz representer generated by latent/confounding variables.
            Default is ``0.03``.

        rho : float
            The correlation between the differences in short and long representations in the main regression and
            Riesz representer. Has to be in [-1,1]. The absolute value determines the adversarial strength of the
            confounding (maximizes at 1.0).
            Default is ``1.0``.

        level : float
            The confidence level.
            Default is ``0.95``.

        null_hypothesis : float or numpy.ndarray
            Null hypothesis for the effect. Determines the robustness values.
            If it is a single float uses the same null hypothesis for all estimated parameters.
            Else the array has to be of shape (n_thetas,).
            Default is ``0.0``.

        Returns
        -------
        self : object
        """
        # check null_hypothesis
        if isinstance(null_hypothesis, float):
            null_hypothesis_vec = np.full(shape=self._n_thetas, fill_value=null_hypothesis)
        elif isinstance(null_hypothesis, np.ndarray):
            if null_hypothesis.shape == (self._n_thetas,):
                null_hypothesis_vec = null_hypothesis
            else:
                raise ValueError("null_hypothesis is numpy.ndarray but does not have the required "
                                 f"shape ({self._n_thetas},). "
                                 f'Array of shape {str(null_hypothesis.shape)} was passed.')
        else:
            raise TypeError("null_hypothesis has to be of type float or np.ndarry. "
                            f"{str(null_hypothesis)} of type {str(type(null_hypothesis))} was passed.")

        # compute sensitivity analysis
        sensitivity_dict = self._calc_sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level)

        # compute robustess values with respect to null_hypothesis
        rv = np.full(shape=self._n_thetas, fill_value=np.nan)
        rva = np.full(shape=self._n_thetas, fill_value=np.nan)

        for i_theta in range(self._n_thetas):
            rv[i_theta], rva[i_theta] = self._calc_robustness_value(
                null_hypothesis=null_hypothesis_vec[i_theta],
                level=level,
                rho=rho,
                idx_treatment=i_theta
            )

        sensitivity_dict['rv'] = rv
        sensitivity_dict['rva'] = rva

        # add all input parameters
        input_params = {'cf_y': cf_y,
                        'cf_d': cf_d,
                        'rho': rho,
                        'level': level,
                        'null_hypothesis': null_hypothesis_vec}
        sensitivity_dict['input'] = input_params

        self._sensitivity_params = sensitivity_dict
        return self

    def confint(self, joint=False, level=0.95):
        """
        Confidence intervals for DoubleML models.

        Parameters
        ----------
        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``False``

        level : float
            The confidence level.
            Default is ``0.95``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s).
        """

        if not isinstance(joint, bool):
            raise TypeError('joint must be True or False. '
                            f'Got {str(joint)}.')

        if not isinstance(level, float):
            raise TypeError('The confidence level must be of float type. '
                            f'{str(level)} of type {str(type(level))} was passed.')
        if (level <= 0) | (level >= 1):
            raise ValueError('The confidence level must be in (0,1). '
                             f'{str(level)} was passed.')

        # compute critical values
        alpha = 1 - level
        percentages = np.array([alpha / 2, 1. - alpha / 2])
        if joint:
            if self._boot_t_stat is None:
                raise ValueError('Apply bootstrap() before confint(joint=True).')

            max_abs_t_value_distribution = np.amax(np.abs(self._boot_t_stat), axis=1)
            critical_values = np.quantile(
                a=max_abs_t_value_distribution,
                q=level,
                axis=0)
        else:
            critical_values = np.repeat(norm.ppf(percentages[1]), self._n_rep)

        # compute all cis over repetitions (shape: n_thetas x 2 x n_rep)
        self._all_cis = np.stack(
            (self.all_thetas - self.all_ses * critical_values,
             self.all_thetas + self.all_ses * critical_values),
            axis=1)
        ci = np.median(self._all_cis, axis=2)
        df_ci = pd.DataFrame(ci, columns=['{:.1f} %'.format(i * 100) for i in percentages])

        if self._treatment_names is not None:
            df_ci.set_index(pd.Index(self._treatment_names), inplace=True)

        return df_ci

    def bootstrap(self, method='normal', n_rep_boot=500):
        """
        Multiplier bootstrap for DoubleMLFrameworks.

        Parameters
        ----------
        method : str
            A str (``'Bayes'``, ``'normal'`` or ``'wild'``) specifying the multiplier bootstrap method.
            Default is ``'normal'``

        n_rep_boot : int
            The number of bootstrap replications.

        Returns
        -------
        self : object
        """

        _check_bootstrap(method, n_rep_boot)
        if self._is_cluster_data:
            raise NotImplementedError('bootstrap not yet implemented with clustering.')

        self._n_rep_boot = n_rep_boot
        self._boot_method = method
        # initialize bootstrap distribution array
        self._boot_t_stat = np.full((n_rep_boot, self.n_thetas, self._n_rep), np.nan)
        var_scaling = self._var_scaling_factors.reshape(-1, 1) * self._all_ses
        for i_rep in range(self.n_rep):
            weights = _draw_weights(method, n_rep_boot, self._n_obs)
            bootstraped_scaled_psi = np.matmul(weights, np.divide(self._scaled_psi[:, :, i_rep], var_scaling[:, i_rep]))
            self._boot_t_stat[:, :, i_rep] = bootstraped_scaled_psi

        return self

    def p_adjust(self, method='romano-wolf'):
        """
        Multiple testing adjustment for DoubleML Frameworks.

        Parameters
        ----------
        method : str
            A str (``'romano-wolf''``, ``'bonferroni'``, ``'holm'``, etc) specifying the adjustment method.
            In addition to ``'romano-wolf''``, all methods implemented in
            :py:func:`statsmodels.stats.multitest.multipletests` can be applied.
            Default is ``'romano-wolf'``.

        Returns
        -------
        df_p_vals : pd.DataFrame
            A data frame with adjusted p-values.
        all_p_vals_corrected : np.ndarray
            A numpy array with all corrected p-values for each repetition.
        """
        if not isinstance(method, str):
            raise TypeError('The p_adjust method must be of str type. '
                            f'{str(method)} of type {str(type(method))} was passed.')

        all_p_vals_corrected = np.full_like(self.all_pvals, np.nan)

        for i_rep in range(self.n_rep):
            p_vals_tmp = self.all_pvals[:, i_rep]

            if method.lower() in ['rw', 'romano-wolf']:
                if self._boot_t_stat is None:
                    raise ValueError(f'Apply bootstrap() before p_adjust("{method}").')

                bootstrap_t_stats = self._boot_t_stat[:, :, i_rep]

                p_init = np.full_like(p_vals_tmp, np.nan)
                p_vals_corrected_tmp_sorted = np.full_like(p_vals_tmp, np.nan)

                abs_t_stats_tmp = abs(self.all_t_stats[:, i_rep])
                # sort in reverse order
                stepdown_ind = np.argsort(abs_t_stats_tmp)[::-1]
                # reversing the order of the sorted indices
                ro = np.argsort(stepdown_ind)

                for i_theta in range(self.n_thetas):
                    bootstrap_citical_value = np.max(
                        abs(np.delete(bootstrap_t_stats, stepdown_ind[:i_theta], axis=1)),
                        axis=1)
                    p_init[i_theta] = np.minimum(1, np.mean(bootstrap_citical_value >= abs_t_stats_tmp[stepdown_ind][i_theta]))

                for i_theta in range(self.n_thetas):
                    if i_theta == 0:
                        p_vals_corrected_tmp_sorted[i_theta] = p_init[i_theta]
                    else:
                        p_vals_corrected_tmp_sorted[i_theta] = np.maximum(
                            p_init[i_theta],
                            p_vals_corrected_tmp_sorted[i_theta - 1])

                # reorder p-values
                p_vals_corrected_tmp = p_vals_corrected_tmp_sorted[ro]
            else:
                _, p_vals_corrected_tmp, _, _ = multipletests(p_vals_tmp, method=method)

            all_p_vals_corrected[:, i_rep] = p_vals_corrected_tmp

        p_vals_corrected = np.median(all_p_vals_corrected, axis=1)
        df_p_vals = pd.DataFrame(
            np.vstack((self.thetas, p_vals_corrected)).T,
            columns=['thetas', 'pval'])

        return df_p_vals, all_p_vals_corrected

    def sensitivity_plot(self, idx_treatment=0, value='theta', rho=1.0, level=0.95, null_hypothesis=0.0,
                         include_scenario=True, benchmarks=None, fill=True, grid_bounds=(0.15, 0.15), grid_size=100):
        """
        Contour plot of the sensivity with respect to latent/confounding variables.

        Parameters
        ----------
        idx_treatment : int
            Index of the treatment to perform the sensitivity analysis.
            Default is ``0``.

        value : str
            Determines which contours to plot. Valid values are ``'theta'`` (refers to the bounds)
            and ``'ci'`` (refers to the bounds including statistical uncertainty).
            Default is ``'theta'``.

        rho: float
            The correlation between the differences in short and long representations in the main regression and
            Riesz representer. Has to be in [-1,1]. The absolute value determines the adversarial strength of the
            confounding (maximizes at 1.0).
            Default is ``1.0``.

        level : float
            The confidence level.
            Default is ``0.95``.

        null_hypothesis : float
            Null hypothesis for the effect. Determines the direction of the contour lines.

        include_scenario : bool
            Indicates whether to highlight the scenario from the call of :meth:`sensitivity_analysis`.
            Default is ``True``.

        benchmarks : dict or None
            Dictionary of benchmarks to be included in the plot. The keys are ``cf_y``, ``cf_d`` and ``name``.
            Default is ``None``.

        fill : bool
            Indicates whether to use a heatmap style or only contour lines.
            Default is ``True``.

        grid_bounds : tuple
            Determines the evaluation bounds of the grid for ``cf_d`` and ``cf_y``. Has to contain two floats in [0, 1).
            Default is ``(0.15, 0.15)``.

        grid_size : int
            Determines the number of evaluation points of the grid.
            Default is ``100``.

        Returns
        -------
        fig : object
            Plotly figure of the sensitivity contours.
        """
        _check_integer(idx_treatment, "idx_treatment", lower_bound=0, upper_bound=self.n_thetas-1)
        if not isinstance(value, str):
            raise TypeError('value must be a string. '
                            f'{str(value)} of type {type(value)} was passed.')
        valid_values = ['theta', 'ci']
        if value not in valid_values:
            raise ValueError('Invalid value ' + value + '. ' +
                             'Valid values ' + ' or '.join(valid_values) + '.')
        _check_float(null_hypothesis, "null_hypothesis")
        _check_bool(include_scenario, 'include_scenario')
        if include_scenario and self.sensitivity_params is None:
            raise ValueError('Apply sensitivity_analysis() to include senario in sensitivity_plot. ')
        _check_benchmarks(benchmarks)
        _check_bool(fill, 'fill')
        _check_in_zero_one(grid_bounds[0], "grid_bounds", include_zero=False, include_one=False)
        _check_in_zero_one(grid_bounds[1], "grid_bounds", include_zero=False, include_one=False)
        _check_integer(grid_size, "grid_size", lower_bound=10)

        null_hypothesis = self.sensitivity_params['input']['null_hypothesis'][idx_treatment]
        unadjusted_theta = self.thetas[idx_treatment]
        # check which side is relvant
        bound = 'upper' if (null_hypothesis > unadjusted_theta) else 'lower'

        # create evaluation grid
        cf_d_vec = np.linspace(0, grid_bounds[0], grid_size)
        cf_y_vec = np.linspace(0, grid_bounds[1], grid_size)

        # compute contour values
        contour_values = np.full(shape=(grid_size, grid_size), fill_value=np.nan)
        for i_cf_d_grid, cf_d_grid in enumerate(cf_d_vec):
            for i_cf_y_grid, cf_y_grid in enumerate(cf_y_vec):

                sens_dict = self._calc_sensitivity_analysis(
                    cf_y=cf_y_grid,
                    cf_d=cf_d_grid,
                    rho=rho,
                    level=level,
                )
                contour_values[i_cf_d_grid, i_cf_y_grid] = sens_dict[value][bound][idx_treatment]

        # get the correct unadjusted value for confidence bands
        if value == 'theta':
            unadjusted_value = unadjusted_theta
        else:
            assert value == 'ci'
            ci = self.confint(level=self.sensitivity_params['input']['level'])
            if bound == 'upper':
                unadjusted_value = ci.iloc[idx_treatment, 1]
            else:
                unadjusted_value = ci.iloc[idx_treatment, 0]

        # compute the values for the benchmarks
        benchmark_dict = copy.deepcopy(benchmarks)
        if benchmarks is not None:
            n_benchmarks = len(benchmarks['name'])
            benchmark_values = np.full(shape=(n_benchmarks,), fill_value=np.nan)
            for benchmark_idx in range(len(benchmarks['name'])):
                sens_dict_bench = self._calc_sensitivity_analysis(
                    cf_y=benchmarks['cf_y'][benchmark_idx],
                    cf_d=benchmarks['cf_d'][benchmark_idx],
                    rho=self.sensitivity_params['input']['rho'],
                    level=self.sensitivity_params['input']['level']
                )
                benchmark_values[benchmark_idx] = sens_dict_bench[value][bound][idx_treatment]
            benchmark_dict['value'] = benchmark_values
        fig = _sensitivity_contour_plot(x=cf_d_vec,
                                        y=cf_y_vec,
                                        contour_values=contour_values,
                                        unadjusted_value=unadjusted_value,
                                        scenario_x=self.sensitivity_params['input']['cf_d'],
                                        scenario_y=self.sensitivity_params['input']['cf_y'],
                                        scenario_value=self.sensitivity_params[value][bound][idx_treatment],
                                        include_scenario=include_scenario,
                                        benchmarks=benchmark_dict,
                                        fill=fill)
        return fig

    def _check_and_set_cluster_data(self, doubleml_dict):
        self._cluster_dict = None

        if "is_cluster_data" in doubleml_dict.keys():
            _check_bool(doubleml_dict['is_cluster_data'], 'is_cluster_data')
            self._is_cluster_data = doubleml_dict['is_cluster_data']

        if self._is_cluster_data:
            if not ("cluster_dict" in doubleml_dict.keys()):
                raise ValueError('If is_cluster_data is True, cluster_dict must be provided.')

            if not isinstance(doubleml_dict['cluster_dict'], dict):
                raise TypeError('cluster_dict must be a dictionary.')

            expected_keys_cluster = ['smpls', 'smpls_cluster', 'cluster_vars', 'n_folds_per_cluster']
            if not all(key in doubleml_dict['cluster_dict'].keys() for key in expected_keys_cluster):
                raise ValueError('The cluster_dict must contain the following keys: ' + ', '.join(expected_keys_cluster)
                                 + '. Got: ' + ', '.join(doubleml_dict['cluster_dict'].keys()) + '.')

            self._cluster_dict = doubleml_dict['cluster_dict']

        return

    def _check_and_set_sensitivity_elements(self, doubleml_dict):
        if not ("sensitivity_elements" in doubleml_dict.keys()):
            sensitivity_implemented = False
            sensitivity_elements = None

        else:
            if not isinstance(doubleml_dict['sensitivity_elements'], dict):
                raise TypeError('sensitivity_elements must be a dictionary.')
            expected_keys_sensitivity = ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2', 'riesz_rep']
            if not all(key in doubleml_dict['sensitivity_elements'].keys() for key in expected_keys_sensitivity):
                raise ValueError('The sensitivity_elements dict must contain the following '
                                 'keys: ' + ', '.join(expected_keys_sensitivity))

            for key in expected_keys_sensitivity:
                if not isinstance(doubleml_dict['sensitivity_elements'][key], np.ndarray):
                    raise TypeError(f'The sensitivity element {key} must be a numpy array.')

            # set sensitivity elements
            sensitivity_implemented = True
            sensitivity_elements = {
                'sigma2': doubleml_dict['sensitivity_elements']['sigma2'],
                'nu2': doubleml_dict['sensitivity_elements']['nu2'],
                'psi_sigma2': doubleml_dict['sensitivity_elements']['psi_sigma2'],
                'psi_nu2': doubleml_dict['sensitivity_elements']['psi_nu2'],
                'riesz_rep': doubleml_dict['sensitivity_elements']['riesz_rep'],
            }

        self._sensitivity_implemented = sensitivity_implemented
        self._sensitivity_elements = sensitivity_elements
        self._sensitivity_params = None

        return

    def _check_framework_shapes(self):
        score_dim = (self._n_obs, self._n_thetas, self.n_rep)
        # check if all sizes match
        if self._thetas.shape != (self._n_thetas,):
            raise ValueError(f'The shape of thetas does not match the expected shape ({self._n_thetas},).')
        if self._ses.shape != (self._n_thetas,):
            raise ValueError(f'The shape of ses does not match the expected shape ({self._n_thetas},).')
        if self._all_thetas.shape != (self._n_thetas, self._n_rep):
            raise ValueError(f'The shape of all_thetas does not match the expected shape ({self._n_thetas}, {self._n_rep}).')
        if self._all_ses.shape != (self._n_thetas, self._n_rep):
            raise ValueError(f'The shape of all_ses does not match the expected shape ({self._n_thetas}, {self._n_rep}).')
        if self._var_scaling_factors.shape != (self._n_thetas,):
            raise ValueError(f'The shape of var_scaling_factors does not match the expected shape ({self._n_thetas},).')
        # dimension of scaled_psi is n_obs x n_thetas x n_rep (per default)
        if self._scaled_psi.shape != score_dim:
            raise ValueError(('The shape of scaled_psi does not match the expected '
                              f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))

        if self._sensitivity_implemented:
            if self._sensitivity_elements['sigma2'].shape != (1, self._n_thetas, self.n_rep):
                raise ValueError('The shape of sigma2 does not match the expected shape '
                                 f'(1, {self._n_thetas}, {self._n_rep}).')
            if self._sensitivity_elements['nu2'].shape != (1, self._n_thetas, self.n_rep):
                raise ValueError(f'The shape of nu2 does not match the expected shape (1, {self._n_thetas}, {self._n_rep}).')
            if self._sensitivity_elements['psi_sigma2'].shape != score_dim:
                raise ValueError(('The shape of psi_sigma2 does not match the expected '
                                 f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))
            if self._sensitivity_elements['psi_nu2'].shape != score_dim:
                raise ValueError(('The shape of psi_nu2 does not match the expected '
                                 f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))
            if self._sensitivity_elements['riesz_rep'].shape != score_dim:
                raise ValueError(('The shape of riesz_rep does not match the expected '
                                 f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))

        return None

    def _check_treatment_names(self, treatment_names):
        if not isinstance(treatment_names, list):
            raise TypeError('treatment_names must be a list. '
                            f'Got {str(treatment_names)} of type {str(type(treatment_names))}.')
        is_str = [isinstance(name, str) for name in treatment_names]
        if not all(is_str):
            raise TypeError('treatment_names must be a list of strings. '
                            f'At least one element is not a string: {str(treatment_names)}.')
        if len(treatment_names) != self._n_thetas:
            raise ValueError('The length of treatment_names does not match the number of treatments. '
                             f'Got {self._n_thetas} treatments and {len(treatment_names)} treatment names.')
        return None


def concat(objs):
    """
    Concatenate DoubleMLFramework objects.
    """
    if len(objs) == 0:
        raise TypeError('Need at least one object to concatenate.')

    if not all(isinstance(obj, DoubleMLFramework) for obj in objs):
        raise TypeError('All objects must be of type DoubleMLFramework.')

    # check on internal consitency of objects
    _ = [obj._check_framework_shapes() for obj in objs]
    # check if all objects are compatible in n_obs and n_rep
    _ = [_check_framework_compatibility(objs[0], obj, check_treatments=False) for obj in objs[1:]]

    all_thetas = np.concatenate([obj.all_thetas for obj in objs], axis=0)
    all_ses = np.concatenate([obj.all_ses for obj in objs], axis=0)
    var_scaling_factors = np.concatenate([obj._var_scaling_factors for obj in objs], axis=0)
    scaled_psi = np.concatenate([obj._scaled_psi for obj in objs], axis=1)

    thetas = np.concatenate([obj.thetas for obj in objs], axis=0)
    ses = np.concatenate([obj.ses for obj in objs], axis=0)

    if any(obj._is_cluster_data for obj in objs):
        raise NotImplementedError('concat not yet implemented with clustering.')
    else:
        is_cluster_data = False

    doubleml_dict = {
        'thetas': thetas,
        'ses': ses,
        'all_thetas': all_thetas,
        'all_ses': all_ses,
        'var_scaling_factors': var_scaling_factors,
        'scaled_psi': scaled_psi,
        'is_cluster_data': is_cluster_data,
    }

    if all(obj._sensitivity_implemented for obj in objs):
        sensitivity_elements = {}
        for key in ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2', 'riesz_rep']:
            assert all(key in obj._sensitivity_elements.keys() for obj in objs)
            sensitivity_elements[key] = np.concatenate([obj._sensitivity_elements[key] for obj in objs], axis=1)

        doubleml_dict['sensitivity_elements'] = sensitivity_elements

    new_obj = DoubleMLFramework(doubleml_dict)

    # check internal consistency of new object
    new_obj._check_framework_shapes()

    return new_obj
