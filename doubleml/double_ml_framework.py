import numpy as np
import pandas as pd
from scipy.stats import norm

from .double_ml import DoubleML
from .utils._estimation import _draw_weights, _aggregate_coefs_and_ses
from .utils._checks import _check_bootstrap


class DoubleMLFramework():
    """Double Machine Learning Framework to combine DoubleML classes and compute confidendence intervals.

    Parameters
    ----------
   doubleml_obj : :class:`DoubleML` object or dict
        The :class:`DoubleML` object providing the estimated parameters and normalized scores or a dict containing
        the corresponding keys and values. Keys have to be 'thetas', 'ses', 'all_thetas', 'all_ses', 'var_scaling_factors' and
         'scaled_psi'. Values have to be numpy arrays with the corresponding shapes.

    """

    def __init__(
            self,
            doubleml_obj=None,
    ):
        self._is_cluster_data = False
        if isinstance(doubleml_obj, dict):
            expected_keys = ['thetas', 'ses', 'all_thetas', 'all_ses', 'var_scaling_factors', 'scaled_psi']
            if not all(key in doubleml_obj.keys() for key in expected_keys):
                raise ValueError('The dict must contain the following keys: ' + ', '.join(expected_keys))

            # set scores and parameters
            self._n_thetas = doubleml_obj['scaled_psi'].shape[1]
            self._n_rep = doubleml_obj['scaled_psi'].shape[2]
            self._n_obs = doubleml_obj['scaled_psi'].shape[0]

            self._thetas = doubleml_obj['thetas']
            self._ses = doubleml_obj['ses']
            self._all_thetas = doubleml_obj['all_thetas']
            self._all_ses = doubleml_obj['all_ses']
            self._var_scaling_factors = doubleml_obj['var_scaling_factors']
            self._scaled_psi = doubleml_obj['scaled_psi']

            if "is_cluster_data" in doubleml_obj.keys():
                self._is_cluster_data = doubleml_obj['is_cluster_data']
        else:
            assert isinstance(doubleml_obj, DoubleML)
            if doubleml_obj._is_cluster_data:
                raise NotImplementedError('DoubleMLFramework does not support cluster data yet.')
            # set scores and parameters according to doubleml_obj
            self._n_thetas = doubleml_obj._dml_data.n_treat
            self._n_rep = doubleml_obj.n_rep
            self._n_obs = doubleml_obj._dml_data.n_obs

            self._thetas = doubleml_obj.coef
            self._ses = doubleml_obj.se
            self._all_thetas = doubleml_obj.all_coef
            self._all_ses = doubleml_obj.all_se
            self._var_scaling_factors = doubleml_obj._var_scaling_factors
            self._scaled_psi = np.divide(doubleml_obj.psi, np.mean(doubleml_obj.psi_deriv, axis=0))

            self._is_cluster_data = doubleml_obj._is_cluster_data

        # check if all sizes match
        _check_framework_shapes(self)
        # initialize bootstrap distribution
        self._bootstrap_distribution = None

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
        Estimated standard errors for each repetition.
        """
        return self._all_ses

    def __add__(self, other):

        if isinstance(other, DoubleMLFramework):

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
            if self._bootstrap_distribution is None:
                raise ValueError('Apply bootstrap() before confint().')
            critical_values = np.quantile(
                a=self._bootstrap_distribution,
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

        # initialize bootstrap distribution array
        self._bootstrap_distribution = np.full((n_rep_boot, self._n_rep), np.nan)
        var_scaling = self._var_scaling_factors.reshape(-1, 1) * self._all_ses
        for i_rep in range(self.n_rep):
            weights = _draw_weights(method, n_rep_boot, self._n_obs)
            bootstraped_scaled_psi = np.matmul(weights, np.divide(self._scaled_psi[:, :, i_rep], var_scaling[:, i_rep]))
            self._bootstrap_distribution[:, i_rep] = np.amax(np.abs(bootstraped_scaled_psi), axis=1)

        return self


def concat(objs):
    """
    Concatenate DoubleMLFramework objects.
    """
    if len(objs) == 0:
        raise ValueError('Need at least one object to concatenate.')

    if not all(isinstance(obj, DoubleMLFramework) for obj in objs):
        raise ValueError('All objects must be of type DoubleMLFramework.')

    # TODO: Add more Input checks
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
    if self._scaled_psi.shape != (self._n_obs, self._n_thetas, self._n_rep):
        raise ValueError(('The shape of scaled_psi does not match the expected '
                          f'shape ({self._n_obs}, {self._n_thetas}, {self._n_rep}).'))

    return None
