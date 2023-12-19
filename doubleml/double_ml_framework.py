import numpy as np
import pandas as pd
from scipy.stats import norm

from .double_ml_base import DoubleMLBase
from ._utils_base import _draw_weights, _initialize_arrays, _aggregate_thetas_and_ses
from ._utils_checks import _check_bootstrap


class DoubleMLFramework():
    """Double Machine Learning Framework to combine DoubleMLBase classes and compute confidendence intervals."""

    def __init__(
            self,
            dml_base_obj=None,
            n_thetas=1,
            n_rep=1,
            n_obs=1,
    ):
        # set dimensions
        if dml_base_obj is None:
            # set scores and parameters
            self._n_thetas = n_thetas
            self._n_rep = n_rep
            self._n_obs = n_obs
            self._var_scaling_factors = np.full(self._n_thetas, np.nan)

            # REMARK THIS IS NOT PSI BUT SCALED WITH J^-1
            self._thetas, self._ses, self._all_thetas, self._all_ses, self._var_scaling_factors, \
                self._scaled_psi, _ = _initialize_arrays(self._n_thetas, self._n_rep, self._n_obs)

        else:
            assert isinstance(dml_base_obj, DoubleMLBase)
            # set scores and parameters according to dml_base_obj
            self._n_thetas = dml_base_obj.n_thetas
            self._n_rep = dml_base_obj.n_rep
            self._n_obs = dml_base_obj.n_obs

            self._thetas = dml_base_obj.thetas
            self._ses = dml_base_obj.ses
            self._all_thetas = dml_base_obj.all_thetas
            self._all_ses = dml_base_obj.all_ses
            self._var_scaling_factors = dml_base_obj.var_scaling_factors
            self._scaled_psi = np.divide(dml_base_obj.psi, np.mean(dml_base_obj.psi_deriv, axis=0))

        # initialize bootstrap distribution
        self._bootstrap_distribution = None

    @property
    def dml_base_objs(self):
        """
        Sequence of DoubleMLBase objects.
        """
        return self._dml_base_objs

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
        new_obj = DoubleMLFramework(
            dml_base_obj=None,
            n_thetas=self._n_thetas,
            n_rep=self._n_rep,
            n_obs=self._n_obs,
        )
        if isinstance(other, DoubleMLFramework):
            new_obj._all_thetas = self._all_thetas + other._all_thetas

            # TODO: check if var_scaling_factors are the same
            assert np.allclose(self._var_scaling_factors, other._var_scaling_factors)
            new_obj._var_scaling_factors = self._var_scaling_factors
            new_obj._scaled_psi = self._scaled_psi + other._scaled_psi
            sigma2_hat = np.divide(
                np.mean(np.square(new_obj._scaled_psi), axis=0),
                new_obj._var_scaling_factors.reshape(-1, 1))
            new_obj._all_ses = np.sqrt(sigma2_hat)

            new_obj._thetas, new_obj._ses = _aggregate_thetas_and_ses(
                new_obj._all_thetas, new_obj._all_ses, new_obj._var_scaling_factors,
                aggregation_method='median')

        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new_obj = DoubleMLFramework(
            dml_base_obj=None,
            n_thetas=self._n_thetas,
            n_rep=self._n_rep,
            n_obs=self._n_obs,
        )
        if isinstance(other, DoubleMLFramework):
            new_obj._all_thetas = self._all_thetas - other._all_thetas

            # TODO: check if var_scaling_factors are the same
            assert np.allclose(self._var_scaling_factors, other._var_scaling_factors)
            new_obj._var_scaling_factors = self._var_scaling_factors
            new_obj._scaled_psi = self._scaled_psi - other._scaled_psi
            sigma2_hat = np.divide(
                np.mean(np.square(new_obj._scaled_psi), axis=0),
                new_obj._var_scaling_factors.reshape(-1, 1))
            new_obj._all_ses = np.sqrt(sigma2_hat)

            new_obj._thetas, new_obj._ses = _aggregate_thetas_and_ses(
                new_obj._all_thetas, new_obj._all_ses, new_obj._var_scaling_factors,
                aggregation_method='median')

        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __rsub__(self, other):
        return self.__sub__(other)

    # TODO: Restrict to linear?
    def __mul__(self, other):
        new_obj = DoubleMLFramework(
            dml_base_obj=None,
            n_thetas=self._n_thetas,
            n_rep=self._n_rep,
            n_obs=self._n_obs,
        )
        if isinstance(other, (int, float)):
            new_obj._thetas = np.multiply(other, self._thetas)
            new_obj._all_thetas = np.multiply(other, self._all_thetas)

            new_obj._var_scaling_factors = self._var_scaling_factors
            new_obj._ses = np.multiply(other, self._ses)
            new_obj._all_ses = np.multiply(other, self._all_ses)
            new_obj._scaled_psi = np.multiply(other, self._scaled_psi)
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
        # initialize bootstrap distribution array
        self._bootstrap_distribution = np.full((n_rep_boot, self._n_rep), np.nan)
        var_scaling = self._var_scaling_factors.reshape(-1, 1) * self._all_ses
        for i_rep in range(self.n_rep):
            weights = _draw_weights(method, n_rep_boot, self._n_obs)
            bootstraped_scaled_scores = np.matmul(weights, np.divide(self._scaled_psi[:, :, i_rep], var_scaling[:, i_rep]))
            self._bootstrap_distribution[:, i_rep] = np.amax(np.abs(bootstraped_scaled_scores), axis=1)

        return self


def concat(objs):
    """
    Concatenate DoubleMLFramework objects.
    """
    if len(objs) == 0:
        raise ValueError('Need at least one object to concatenate.')

    if not all(isinstance(obj, DoubleMLFramework) for obj in objs):
        raise ValueError('All objects must be of type DoubleMLFramework.')

    n_thetas = sum(obj.n_thetas for obj in objs)
    n_rep = objs[0].n_rep  # TODO: check if all n_rep are the same
    n_obs = objs[0].n_obs  # TODO: check if all n_obs are the same

    # TODO: Add more Input checks
    new_obj = DoubleMLFramework(
        n_thetas=n_thetas,
        n_rep=n_rep,
        n_obs=n_obs,
    )

    new_obj._all_thetas = np.concatenate([obj.all_thetas for obj in objs], axis=0)
    new_obj._all_ses = np.concatenate([obj.all_ses for obj in objs], axis=0)
    new_obj._var_scaling_factors = np.concatenate([obj._var_scaling_factors for obj in objs], axis=0)
    new_obj._scaled_psi = np.concatenate([obj._scaled_psi for obj in objs], axis=1)

    new_obj._thetas = np.concatenate([obj.thetas for obj in objs], axis=0)
    new_obj._ses = np.concatenate([obj.ses for obj in objs], axis=0)

    return new_obj
