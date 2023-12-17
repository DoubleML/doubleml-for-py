import numpy as np
import pandas as pd
from scipy.stats import norm

from .double_ml_base import DoubleMLBase
from ._utils_base import _draw_weights
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
            self._var_scaling_factor = np.full(self._n_thetas, np.nan)
        else:
            assert isinstance(dml_base_obj, DoubleMLBase)
            # set scores and parameters according to dml_base_obj
            self._n_thetas = 1
            self._n_rep = dml_base_obj.n_rep
            self._n_obs = dml_base_obj.n_obs
            self._var_scaling_factor = dml_base_obj.var_scaling_factor

        # initalize arrays
        self._thetas = np.full(self._n_thetas, np.nan)
        self._ses = np.full(self._n_thetas, np.nan)
        self._all_thetas = np.full((self._n_thetas, self._n_rep), np.nan)
        self._all_ses = np.full((self._n_thetas, self._n_rep), np.nan)
        self._psi = np.full((self._n_obs, self._n_thetas, self._n_rep), np.nan)
        self._psi_deriv = np.full((self._n_obs, self._n_thetas, self._n_rep), np.nan)
        self._bootstrap_distribution = None

        if dml_base_obj is not None:
            # initalize arrays from double_ml_base_obj
            self._thetas[0] = np.array([dml_base_obj.theta])
            self._ses[0] = np.array([dml_base_obj.se])
            self._all_thetas[0, :] = dml_base_obj.all_thetas
            self._all_ses[0, :] = dml_base_obj.all_ses
            self._psi[:, 0, :] = dml_base_obj.psi
            self._psi_deriv[:, 0, :] = dml_base_obj.psi_deriv

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
        if isinstance(other, (int, float)):
            new_obj._thetas = self._thetas + other
            new_obj._all_thetas = self._all_thetas + other

            new_obj._ses = self._ses
            new_obj._all_ses = self._all_ses
            new_obj._psi = self._psi
            new_obj._psi_deriv = self._psi_deriv

        elif isinstance(other, DoubleMLFramework):
            new_obj._all_thetas = self._all_thetas + other._all_thetas
            new_obj._psi = self._psi + other._psi
            new_obj._psi_deriv = self._psi_deriv + other._psi_deriv

            # TODO: check if var_scaling_factor is the same
            new_obj._var_scaling_factor = self._var_scaling_factor
            J_self = np.mean(self._psi_deriv, axis=0)
            J_other = np.mean(other._psi_deriv, axis=0)
            omega = self._psi / J_self + other._psi / J_other
            sigma2_hat = np.divide(np.mean(np.square(omega), axis=0), new_obj._var_scaling_factor)
            new_obj._all_ses = np.sqrt(sigma2_hat)

            # TODO: aggragate over repetitions
            new_obj._thetas = np.median(new_obj._all_thetas, axis=1)
            new_obj._ses = np.median(new_obj._all_ses, axis=1)

        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return new_obj

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            pass
        else:
            pass

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
        all_cis = np.stack(
            (self.all_thetas - self.all_ses * critical_values,
             self.all_thetas + self.all_ses * critical_values),
            axis=1)
        ci = np.median(all_cis, axis=2)
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
        score_scaling = self._n_obs * np.multiply(self._all_ses, np.mean(self._psi_deriv, axis=0))
        for i_rep in range(self.n_rep):
            weights = _draw_weights(method, n_rep_boot, self._n_obs)
            bootstraped_scores = np.matmul(weights, self._psi[:, :, i_rep])
            bootstraped_max_scores = np.amax(np.abs(bootstraped_scores), axis=1)
            self._bootstrap_distribution[:, i_rep] = np.divide(bootstraped_max_scores, score_scaling[:, i_rep])

        return self
