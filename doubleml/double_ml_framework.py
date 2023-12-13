import numpy as np


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
