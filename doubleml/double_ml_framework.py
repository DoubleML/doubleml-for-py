import numpy as np
import pandas as pd

from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .utils._estimation import _draw_weights, _aggregate_coefs_and_ses
from .utils._checks import _check_bootstrap, _check_framework_compatibility


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
        assert isinstance(doubleml_dict, dict), "doubleml_dict must be a dictionary."
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

        if "is_cluster_data" in doubleml_dict.keys():
            self._is_cluster_data = doubleml_dict['is_cluster_data']

        # check if all sizes match
        _check_framework_shapes(self)
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

    def __add__(self, other):

        if isinstance(other, DoubleMLFramework):
            # internal consistency check
            _check_framework_shapes(self)
            _check_framework_shapes(other)
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

            is_cluster_data = self._is_cluster_data or other._is_cluster_data
            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': is_cluster_data,
            }
            new_obj = DoubleMLFramework(doubleml_dict)

        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        if isinstance(other, DoubleMLFramework):
            # internal consistency check
            _check_framework_shapes(self)
            _check_framework_shapes(other)
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

            is_cluster_data = self._is_cluster_data or other._is_cluster_data
            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': is_cluster_data,
            }
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

            is_cluster_data = self._is_cluster_data
            doubleml_dict = {
                'thetas': thetas,
                'ses': ses,
                'all_thetas': all_thetas,
                'all_ses': all_ses,
                'var_scaling_factors': var_scaling_factors,
                'scaled_psi': scaled_psi,
                'is_cluster_data': is_cluster_data,
            }
            new_obj = DoubleMLFramework(doubleml_dict)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __rmul__(self, other):
        return self.__mul__(other)

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
        # TODO: add treatment names
        df_ci = pd.DataFrame(ci, columns=['{:.1f} %'.format(i * 100) for i in percentages])
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


def concat(objs):
    """
    Concatenate DoubleMLFramework objects.
    """
    if len(objs) == 0:
        raise TypeError('Need at least one object to concatenate.')

    if not all(isinstance(obj, DoubleMLFramework) for obj in objs):
        raise TypeError('All objects must be of type DoubleMLFramework.')

    # check on internal consitency of objects
    _ = [_check_framework_shapes(obj) for obj in objs]
    # check if all objects are compatible in n_obs and n_rep
    _ = [_check_framework_compatibility(objs[0], obj, check_treatments=False) for obj in objs[1:]]

    all_thetas = np.concatenate([obj.all_thetas for obj in objs], axis=0)
    all_ses = np.concatenate([obj.all_ses for obj in objs], axis=0)
    var_scaling_factors = np.concatenate([obj._var_scaling_factors for obj in objs], axis=0)
    scaled_psi = np.concatenate([obj._scaled_psi for obj in objs], axis=1)

    thetas = np.concatenate([obj.thetas for obj in objs], axis=0)
    ses = np.concatenate([obj.ses for obj in objs], axis=0)

    is_cluster_data = any(obj._is_cluster_data for obj in objs)
    doubleml_dict = {
        'thetas': thetas,
        'ses': ses,
        'all_thetas': all_thetas,
        'all_ses': all_ses,
        'var_scaling_factors': var_scaling_factors,
        'scaled_psi': scaled_psi,
        'is_cluster_data': is_cluster_data,
    }
    new_obj = DoubleMLFramework(doubleml_dict)

    # check internal consistency of new object
    _check_framework_shapes(new_obj)

    return new_obj


def _check_framework_shapes(self):
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
    if self._scaled_psi.shape != (self._n_obs, self._n_thetas, self._n_rep):
        raise ValueError(('The shape of scaled_psi does not match the expected '
                          f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))

    return None
