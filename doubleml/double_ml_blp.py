import statsmodels.api as sm
import numpy as np
import pandas as pd
import doubleml as dml

from scipy.stats import norm
from scipy.linalg import sqrtm

class DoubleMLIRMBLP:
    """Best Linear Predictor for DoubleML IRM models

        Parameters
        ----------
        obj_dml_irm : :class:`DoubleMLIRM` object
            The :class:`DoubleMLIRM` object providing the interactive regression model with fitted nuisance functions.

        basis : :class:`pandas.DataFrame`
        The basis for estimating the best linear predictor. Has to correspond to the observations of the IRM model.
    """

    def __init__(self,
                 obj_dml_irm,
                 basis):

        # check and pick up obj_dml_irm
        if not isinstance(obj_dml_irm, dml.DoubleMLIRM):
            raise TypeError('The model must be of DoubleMLIRM type. '
                            f'{str(obj_dml_irm)} of type {str(type(obj_dml_irm))} was passed.')

        if not isinstance(basis, pd.DataFrame):
            raise TypeError('The basis must be of DataFrame type. '
                            f'{str(basis)} of type {str(type(basis))} was passed.')

        self._dml_irm = obj_dml_irm
        self._basis = basis

        # initialize the orthogonal signal, the score and the covariance
        self._orth_signal = None
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
        # get the orthogonal signal from the IRM model
        self._orth_signal = self._dml_irm.psi_b.reshape(-1, 1)
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
        # check this again (calculation of HC0 should include scaling with sample size)
        # se_scaling = np.sqrt(self._dml_irm._dml_data.n_obs)
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


from doubleml.tests._utils_blp_manual import create_spline_basis, create_synthetic_data

# DGP constants
np.random.seed(123)
n = 2000
n_w = 10
support_size = 5
n_x = 1

# Create data
data, covariates = create_synthetic_data(n=n, n_w=n_w, support_size=support_size, n_x=n_x, constant=True)
data_dml_base = dml.DoubleMLData(data,
                                 y_col='y',
                                 d_cols='t',
                                 x_cols=covariates)

# First stage estimation
# Lasso regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
randomForest_reg = RandomForestRegressor(n_estimators=500)
randomForest_class = RandomForestClassifier(n_estimators=500)

np.random.seed(123)

dml_irm = dml.DoubleMLIRM(data_dml_base,
                          ml_g=randomForest_reg,
                          ml_m=randomForest_class,
                          trimming_threshold=0.01,
                          n_folds=5
                          )
print("Training first stage")
dml_irm.fit(store_predictions=True)

spline_basis = create_spline_basis(X=data["x"], knots=3, degree=2)

cate = DoubleMLIRMBLP(dml_irm, basis=spline_basis).fit()

print(cate.confint(spline_basis, joint=False))

groups = pd.DataFrame(np.vstack([data["x"] <= 0.2, (data["x"] >= 0.2) & (data["x"] <= 0.7), data["x"] >= 0.7]).T,
             columns=['Group 1', 'Group 2', 'Group 3'])

gate = DoubleMLIRMBLP(dml_irm, basis=groups).fit()
print(gate.confint(groups))
