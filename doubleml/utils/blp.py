import statsmodels.api as sm
import numpy as np
import pandas as pd
import warnings

from scipy.stats import norm
from scipy.linalg import sqrtm


class DoubleMLBLP:
    """Best linear predictor (BLP) for DoubleML with orthogonal signals.
    Manily used for CATE and GATE estimation for IRM models.

    Parameters
    ----------
    orth_signal : :class:`numpy.array`
        The orthogonal signal to be predicted. Has to be of shape ``(n_obs,)``,
        where ``n_obs`` is the number of observations.

    basis : :class:`pandas.DataFrame`
        The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
        where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

    is_gate : bool
        Indicates whether the basis is constructed for GATEs (dummy-basis).
        Default is ``False``.
    """

    def __init__(self,
                 orth_signal,
                 basis,
                 is_gate=False):

        if not isinstance(orth_signal, np.ndarray):
            raise TypeError('The signal must be of np.ndarray type. '
                            f'Signal of type {str(type(orth_signal))} was passed.')

        if orth_signal.ndim != 1:
            raise ValueError('The signal must be of one dimensional. '
                             f'Signal of dimensions {str(orth_signal.ndim)} was passed.')

        if not isinstance(basis, pd.DataFrame):
            raise TypeError('The basis must be of DataFrame type. '
                            f'Basis of type {str(type(basis))} was passed.')

        if not basis.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')

        self._orth_signal = orth_signal
        self._basis = basis
        self._is_gate = is_gate

        # initialize the score and the covariance
        self._blp_model = None
        self._blp_omega = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Fit summary ------------------\n' + fit_summary
        return res

    @property
    def blp_model(self):
        """
        Best-Linear-Predictor model.
        """
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
    def blp_omega(self):
        """
        Covariance matrix.
        """
        return self._blp_omega

    @property
    def summary(self):
        """
        A summary for the best linear predictor effect after calling :meth:`fit`.
        """
        col_names = ['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
        if self.blp_model is None:
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = {'coef': self.blp_model.params,
                             'std err': self.blp_model.bse,
                             't': self.blp_model.tvalues,
                             'P>|t|': self.blp_model.pvalues,
                             '[0.025': self.blp_model.conf_int()[0],
                             '0.975]': self.blp_model.conf_int()[1]}
            df_summary = pd.DataFrame(summary_stats,
                                      columns=col_names)
        return df_summary

    def fit(self):
        """
        Estimate DoubleMLBLP models.

        Returns
        -------
        self : object
        """

        # fit the best-linear-predictor of the orthogonal signal with respect to the grid
        self._blp_model = sm.OLS(self._orth_signal, self._basis).fit()
        self._blp_omega = self._blp_model.cov_HC0

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
            raise TypeError('joint must be True or False. '
                            f'Got {str(joint)}.')

        if not isinstance(level, float):
            raise TypeError('The confidence level must be of float type. '
                            f'{str(level)} of type {str(type(level))} was passed.')
        if (level <= 0) | (level >= 1):
            raise ValueError('The confidence level must be in (0,1). '
                             f'{str(level)} was passed.')

        if not isinstance(n_rep_boot, int):
            raise TypeError('The number of bootstrap replications must be of int type. '
                            f'{str(n_rep_boot)} of type {str(type(n_rep_boot))} was passed.')
        if n_rep_boot < 1:
            raise ValueError('The number of bootstrap replications must be positive. '
                             f'{str(n_rep_boot)} was passed.')

        if self._blp_model is None:
            raise ValueError('Apply fit() before confint().')

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
                    warnings.warn('Returning pointwise confidence intervals for basis coefficients.', UserWarning)
                # return the confidence intervals for the basis coefficients
                ci = np.vstack((
                    self.blp_model.conf_int(alpha=alpha/2)[0],
                    self.blp_model.params,
                    self.blp_model.conf_int(alpha=alpha/2)[1])
                    ).T
                df_ci = pd.DataFrame(
                    ci,
                    columns=['{:.1f} %'.format(alpha/2 * 100), 'effect', '{:.1f} %'.format((1-alpha/2) * 100)],
                    index=self._basis.columns)
                return df_ci

        elif not (basis.shape[1] == self._basis.shape[1]):
            raise ValueError('Invalid basis: DataFrame has to have the exact same number and ordering of columns.')
        elif not list(basis.columns.values) == list(self._basis.columns.values):
            raise ValueError('Invalid basis: DataFrame has to have the exact same number and ordering of columns.')

        # blp of the orthogonal signal
        g_hat = self._blp_model.predict(basis)

        np_basis = basis.to_numpy()
        # calculate se for basis elements
        blp_se = np.sqrt((np.dot(np_basis, self._blp_omega) * np_basis).sum(axis=1))

        if joint:
            # calculate the maximum t-statistic with bootstrap
            normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
            bootstrap_samples = np.multiply(np.dot(np_basis, np.dot(sqrtm(self._blp_omega), normal_samples)).T,
                                            (1.0 / blp_se))

            max_t_stat = np.quantile(np.max(np.abs(bootstrap_samples), axis=0), q=level)

            # Lower simultaneous CI
            g_hat_lower = g_hat - max_t_stat * blp_se
            # Upper simultaneous CI
            g_hat_upper = g_hat + max_t_stat * blp_se

        else:
            # Lower point-wise CI
            g_hat_lower = g_hat + norm.ppf(q=alpha / 2) * blp_se
            # Upper point-wise CI
            g_hat_upper = g_hat + norm.ppf(q=1 - alpha / 2) * blp_se

        ci = np.vstack((g_hat_lower, g_hat, g_hat_upper)).T
        df_ci = pd.DataFrame(ci,
                             columns=['{:.1f} %'.format(alpha/2 * 100), 'effect', '{:.1f} %'.format((1-alpha/2) * 100)],
                             index=basis.index)

        if self._is_gate and gate_names is not None:
            df_ci.index = gate_names

        return df_ci
