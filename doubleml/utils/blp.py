import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import sqrtm
from scipy.stats import norm

from ._estimation import _aggregate_coefs_and_ses


class DoubleMLBLP:
    """Best linear predictor (BLP) for DoubleML with orthogonal signals.
    Manily used for CATE and GATE estimation for IRM models.

    Parameters
    ----------
    orth_signal : :class:`numpy.array`
        The orthogonal signal to be predicted. Has to be of shape ``(n_obs,)`` or ``(n_obs, n_rep)``,
        where ``n_obs`` is the number of observations and ``n_rep`` is the number of repetitions.

    basis : :class:`pandas.DataFrame`
        The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
        where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

    is_gate : bool
        Indicates whether the basis is constructed for GATEs (dummy-basis).
        Default is ``False``.
    """

    def __init__(self, orth_signal, basis, is_gate=False):
        if not isinstance(orth_signal, np.ndarray):
            raise TypeError(f"The signal must be of np.ndarray type. Signal of type {str(type(orth_signal))} was passed.")

        if orth_signal.ndim not in [1, 2]:
            raise ValueError(
                f"The signal must be one- or two-dimensional. Signal of dimensions {str(orth_signal.ndim)} was passed."
            )

        if orth_signal.ndim == 1:
            self._orth_signal = orth_signal.reshape(-1, 1)
        else:
            self._orth_signal = orth_signal
        self._n_rep = self._orth_signal.shape[1]
        self._is_gate = is_gate

        if not isinstance(basis, pd.DataFrame):
            raise TypeError(f"The basis must be of DataFrame type. Basis of type {str(type(basis))} was passed.")
        if not basis.columns.is_unique:
            raise ValueError("Invalid pd.DataFrame: Contains duplicate column names.")
        if self._orth_signal.shape[0] != basis.shape[0]:
            raise ValueError(
                "The number of observations in signal and basis does not match. "
                f"Got {str(self._orth_signal.shape[0])} and {str(basis.shape[0])}."
            )
        self._basis = basis

        # initialize the score and the covariance
        self._blp_model = None
        self._blp_omega = None
        self._all_coef = None
        self._all_se = None
        self._coef = None
        self._se = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        fit_summary = str(self.summary)
        res = header + "\n------------------ Fit summary ------------------\n" + fit_summary
        return res

    @property
    def blp_model(self):
        """
        Best-Linear-Predictor models.
        This is a list with one model per repetition.
        """
        if self._blp_model is None:
            return None
        return self._blp_model

    @property
    def orth_signal(self):
        """
        Orthogonal signal.
        """
        return self._orth_signal

    @property
    def basis(self):
        """
        Basis.
        """
        return self._basis

    @property
    def n_rep(self):
        """
        Number of repetitions.
        """
        return self._n_rep

    @property
    def blp_omega(self):
        """
        Covariance matrix.
        For multiple repetitions this has shape ``(d, d, n_rep)``.
        """
        if self._blp_omega is None:
            return None
        return self._blp_omega

    @property
    def coef(self):
        """
        Aggregated coefficients over repetitions.
        """
        return self._coef

    @property
    def se(self):
        """
        Aggregated standard errors over repetitions.
        """
        return self._se

    @property
    def all_coef(self):
        """
        Coefficients for each repetition with shape ``(d, n_rep)``.
        """
        return self._all_coef

    @property
    def all_se(self):
        """
        Standard errors for each repetition with shape ``(d, n_rep)``.
        """
        return self._all_se

    @property
    def summary(self):
        """
        A summary for the best linear predictor effect after calling :meth:`fit`.
        """
        col_names = ["coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
        if self.blp_model is None:
            df_summary = pd.DataFrame(columns=col_names)
        else:
            conf_int_values = [self._blp_model[i].conf_int() for i in range(self.n_rep)]
            t_values = np.divide(self.coef, self.se)
            p_values = 2 * norm.cdf(-np.abs(t_values))
            summary_stats = {
                "coef": self.coef,
                "std err": self.se,
                "t": t_values,
                "P>|t|": p_values,
                "[0.025": np.median([conf_int_values[i][0] for i in range(self.n_rep)], axis=0),
                "0.975]": np.median([conf_int_values[i][1] for i in range(self.n_rep)], axis=0),
            }
            df_summary = pd.DataFrame(summary_stats, columns=col_names, index=self._basis.columns)
        return df_summary

    def fit(self, cov_type="HC0", **kwargs):
        """
        Estimate DoubleMLBLP models.

        Parameters
        ----------
        cov_type : str
            The covariance type to be used in the estimation. Default is ``'HC0'``.
            See :meth:`statsmodels.regression.linear_model.OLS.fit` for more information.

        **kwargs: dict
            Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit`.

        Returns
        -------
        self : object
        """

        # fit the best-linear-predictor of the orthogonal signal with respect to the grid
        n_basis = self._basis.shape[1]
        self._all_coef = np.full((n_basis, self.n_rep), np.nan)
        self._all_se = np.full((n_basis, self.n_rep), np.nan)
        self._blp_omega = np.full((n_basis, n_basis, self.n_rep), np.nan)
        self._blp_model = []

        for i_rep in range(self.n_rep):
            blp_model = sm.OLS(self._orth_signal[:, i_rep], self._basis).fit(cov_type=cov_type, **kwargs)
            self._blp_model.append(blp_model)
            self._all_coef[:, i_rep] = np.asarray(blp_model.params)
            self._all_se[:, i_rep] = np.asarray(blp_model.bse)
            self._blp_omega[:, :, i_rep] = blp_model.cov_params().to_numpy()

        self._coef, self._se = _aggregate_coefs_and_ses(self._all_coef, self._all_se)

        return self

    def confint(self, basis=None, joint=False, level=0.95, n_rep_boot=500):
        """
        Confidence intervals for the BLP model.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for constructing the confidence interval. Has to have the same form as the basis from
            the construction. If ``None`` is passed, if the basis is constructed for GATEs, the GATEs are returned.
            Else, the confidence intervals for the basis coefficients are returned (with pointwise cofidence intervals).
            Default is ``None``.

        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``False``.

        level : float
            The confidence level.
            Default is ``0.95``.

        n_rep_boot : int
            The number of bootstrap repetitions (only relevant for joint confidence intervals).
            Default is ``500``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s).
        """
        if not isinstance(joint, bool):
            raise TypeError(f"joint must be True or False. Got {str(joint)}.")

        if not isinstance(level, float):
            raise TypeError(f"The confidence level must be of float type. {str(level)} of type {str(type(level))} was passed.")
        if (level <= 0) | (level >= 1):
            raise ValueError(f"The confidence level must be in (0,1). {str(level)} was passed.")

        if not isinstance(n_rep_boot, int):
            raise TypeError(
                "The number of bootstrap replications must be of int type. "
                f"{str(n_rep_boot)} of type {str(type(n_rep_boot))} was passed."
            )
        if n_rep_boot < 1:
            raise ValueError(f"The number of bootstrap replications must be positive. {str(n_rep_boot)} was passed.")

        if self._blp_model is None:
            raise ValueError("Apply fit() before confint().")

        alpha = 1 - level
        gate_names = None
        # define basis if none is supplied
        if basis is None:
            if self._is_gate:
                # reduce to unique groups
                basis = pd.DataFrame(np.diag(v=np.full((self._basis.shape[1]), True)))
                gate_names = list(self._basis.columns.values)
            else:
                if joint:
                    warnings.warn("Returning pointwise confidence intervals for basis coefficients.", UserWarning)
                # return the confidence intervals for the basis coefficients
                conf_int_values = [self._blp_model[i].conf_int(alpha=alpha) for i in range(self.n_rep)]
                ci_lower = np.median([conf_int_values[i][0] for i in range(self.n_rep)], axis=0)
                ci_upper = np.median([conf_int_values[i][1] for i in range(self.n_rep)], axis=0)
                ci = np.vstack((ci_lower, self.coef, ci_upper)).T
                df_ci = pd.DataFrame(
                    ci,
                    columns=["{:.1f} %".format(alpha / 2 * 100), "effect", "{:.1f} %".format((1 - alpha / 2) * 100)],
                    index=self._basis.columns,
                )
                return df_ci

        elif not isinstance(basis, pd.DataFrame):
            raise TypeError(f"The basis must be of DataFrame type. Basis of type {str(type(basis))} was passed.")
        elif not (basis.shape[1] == self._basis.shape[1]):
            raise ValueError("Invalid basis: DataFrame has to have the exact same number and ordering of columns.")
        elif not list(basis.columns.values) == list(self._basis.columns.values):
            raise ValueError("Invalid basis: DataFrame has to have the exact same number and ordering of columns.")

        # blp of the orthogonal signal
        g_hat, _, all_g_hat, all_blp_se = self._predict_and_aggregate(basis)

        if joint:
            np_basis = basis.to_numpy()
            critical_values = np.full(self.n_rep, np.nan)

            for i_rep in range(self.n_rep):
                normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
                omega_sqrt = sqrtm(self._blp_omega[:, :, i_rep])
                bootstrap_samples = np.multiply(
                    np.dot(np_basis, np.dot(omega_sqrt, normal_samples)).T, (1.0 / all_blp_se[:, i_rep])
                )
                critical_values[i_rep] = np.quantile(np.max(np.abs(bootstrap_samples), axis=0), q=level)
        else:
            critical_values = np.repeat(norm.ppf(q=1 - alpha / 2), self.n_rep)

        all_g_hat_lower = all_g_hat - critical_values * all_blp_se
        all_g_hat_upper = all_g_hat + critical_values * all_blp_se

        g_hat_lower = np.median(all_g_hat_lower, axis=1)
        g_hat_upper = np.median(all_g_hat_upper, axis=1)

        ci = np.vstack((g_hat_lower, g_hat, g_hat_upper)).T
        df_ci = pd.DataFrame(
            ci,
            columns=["{:.1f} %".format(alpha / 2 * 100), "effect", "{:.1f} %".format((1 - alpha / 2) * 100)],
            index=basis.index,
        )

        if self._is_gate and gate_names is not None:
            df_ci.index = gate_names

        return df_ci

    def _predict_and_aggregate(self, basis):
        np_basis = basis.to_numpy()
        n_obs_basis = basis.shape[0]

        all_g_hat = np.full((n_obs_basis, self.n_rep), np.nan)
        all_blp_se = np.full((n_obs_basis, self.n_rep), np.nan)
        for i_rep in range(self.n_rep):
            all_g_hat[:, i_rep] = np.asarray(self._blp_model[i_rep].predict(basis))
            omega_rep = self._blp_omega[:, :, i_rep]
            all_blp_se[:, i_rep] = np.sqrt((np.dot(np_basis, omega_rep) * np_basis).sum(axis=1))

        g_hat, blp_se = _aggregate_coefs_and_ses(all_g_hat, all_blp_se)

        return g_hat, blp_se, all_g_hat, all_blp_se
