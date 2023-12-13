import numpy as np
import pandas as pd
from scipy.stats import norm


class DoubleMLFramework():
    """Double Machine Learning Framework to combine DoubleMLBase classes and compute confidendence intervals."""

    def __init__(
            self,
            dml_base_objs,
    ):

        # s
        self._dml_base_objs = dml_base_objs
        self._n_thetas = len(dml_base_objs)
        self._n_rep = dml_base_objs[0].n_rep
        self._n_obs = dml_base_objs[0].n_obs
        self._thetas = np.full(self._n_thetas, np.nan)
        self._ses = np.full(self._n_thetas, np.nan)

        # initalize arrays
        self._all_thetas = np.full((self._n_rep, self._n_thetas), np.nan)
        self._all_ses = np.full((self._n_rep, self._n_thetas), np.nan)
        self._psi = np.full((self._n_obs, self._n_rep, self._n_thetas), np.nan)
        self._psi_deriv = np.full((self._n_obs, self._n_rep, self._n_thetas), np.nan)

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

    def estimate_thetas(self, aggregation_method='median'):
        for i_theta, dml_base_obj in enumerate(self._dml_base_objs):
            dml_base_obj.estimate_theta(aggregation_method=aggregation_method)

            self._thetas[i_theta] = dml_base_obj.theta
            self._all_thetas[:, i_theta] = dml_base_obj.all_thetas

            self._ses[i_theta] = dml_base_obj.se
            self._all_ses[:, i_theta] = dml_base_obj.all_ses

            self._psi[:, :, i_theta] = dml_base_obj.psi
            self._psi_deriv[:, :, i_theta] = dml_base_obj.psi_deriv

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

        alpha = 1 - level
        ab = np.array([alpha / 2, 1. - alpha / 2])
        if joint:
            # TODO: add bootstraped critical values
            pass
        else:
            if np.isnan(self.thetas).any():
                raise ValueError('Apply estimate_thetas() before confint().')
            critical_value = norm.ppf(ab)

        ci = np.vstack((self.thetas + self.ses * critical_value[0],
                        self.thetas + self.ses * critical_value[1])).T
        # TODO: add treatment names
        df_ci = pd.DataFrame(
            ci,
            columns=['{:.1f} %'.format(i * 100) for i in ab])
        return df_ci
