import statsmodels.api as sm
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.linalg import sqrtm

class DoubleMLIRMBLP:
    """Best Linear Predictor for DoubleML IRM models

        Parameters
        ----------
        orth_signal : :class:`numpy.array`
            The orthogonal signal to be predicted. Has to be of shape (n,).

        basis : :class:`pandas.DataFrame`
            The basis for estimating the best linear predictor. Has to have the shape (n,d),
            where d is the number of predictors.
    """

    def __init__(self,
                 orth_signal,
                 basis):

        if not isinstance(orth_signal, np.ndarray):
            raise TypeError('The signal must be of np.ndarray type. '
                            f'{str(orth_signal)} of type {str(type(orth_signal))} was passed.')

        if orth_signal.ndim != 1:
            raise ValueError('The signal must be of one dimensional. '
                             f'{str(orth_signal)} of dimensions {str(orth_signal.ndim)} was passed.')

        if not isinstance(basis, pd.DataFrame):
            raise TypeError('The basis must be of DataFrame type. '
                            f'{str(basis)} of type {str(type(basis))} was passed.')

        if not basis.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')

        if orth_signal.shape[0] != basis.shape[0]:
            raise ValueError('Incompatible input dimensions.'
                             f'{str(orth_signal)} of shape {str(orth_signal.shape)} and'
                             f'{str(basis)} of shape {str(basis.shape)} were passed.')

        self._orth_signal = orth_signal
        self._basis = basis

        # initialize the score and the covariance
        self._blp_model = None
        self._blp_omega = None

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
    def blp_omega(self):
        """
        Covariance matrix.
        """
        return self._blp_omega

    def fit(self):
        """
        Estimate DoubleML models.

        Returns
        -------
        self : object
        """

        # fit the best-linear-predictor of the orthogonal signal with respect to the grid
        self._blp_model = sm.OLS(self._orth_signal, self._basis).fit()
        self._blp_omega = self._blp_model.cov_HC0

        return self

    def confint(self, basis, joint=False, level=0.95, n_rep_boot=500):
        """
        Confidence intervals for BLP for DoubleML IRM.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for constructing the confidence interval. Has to have the same form as the basis from
            the construction.

        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``False``

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
        alpha = 1 - level
        # blp of the orthogonal signal
        g_hat = self._blp_model.predict(basis)

        # calculate se for basis elements
        se_scaling = 1
        blp_se = np.sqrt((basis.to_numpy().dot(self._blp_omega) * basis.to_numpy()).sum(axis=1)) / se_scaling

        if joint:
            # calculate the maximum t-statistic with bootstrap
            normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
            bootstrap_samples = np.multiply(basis.to_numpy().dot(np.dot(sqrtm(self._blp_omega), normal_samples)).T,
                                            (blp_se * se_scaling))

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
        return df_ci