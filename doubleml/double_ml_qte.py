import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold, train_test_split
import warnings

from .double_ml import DoubleML
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _draw_weights
from ._utils_resampling import DoubleMLResampling
from .double_ml_data import DoubleMLData, DoubleMLClusterData


class DoubleMLPQ(NonLinearScoreMixin, DoubleML):
    """Double machine learning for potential quantiles

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`g_0(X) = E[Y \le \theta|X, D=d]`.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=d|X]`.

    treatment : int
        Binary treatment indicator. Has to be either ``0`` or ``1``. Determines the potential outcome to be considered.
        Default is ``1``.

    n_folds : int
        Number of folds.
        Default is ``5``.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment,
                 tau=0.5,
                 n_folds=5,
                 n_rep=1,
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 score='PQ',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._tau = tau
        self._treatment = treatment
        self._h = None
        self._normalize = True

        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            self._is_cluster_data = True
        if self._is_cluster_data:
            raise NotImplementedError('Estimation with clustering not implemented.')

        # initialize starting values and bounds
        self._coef_bounds = (self._dml_data.y.min(), self._dml_data.y.max())
        self._coef_start_val = np.quantile(self._dml_data.y, self._tau)

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

        self._check_data(self._dml_data)
        _ = self._check_learner(ml_g, 'ml_g', regressor=False, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m),
                         'ml_m_prelim': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba',
                                'ml_m_prelim': 'predict_proba'}

        self._initialize_ml_nuisance_params()

    @property
    def _score_element_names(self):
        return ['Ind_d', 'g', 'm', 'y']

    def _compute_ipw_score(self, theta, d, y, prop):
        score = (d == self._treatment) * (y <= theta) / prop - self._tau
        return score

    def _compute_score(self, psi_elements, coef, inds=None):
        Ind_d = psi_elements['Ind_d']
        g = psi_elements['g']
        m = psi_elements['m']
        y = psi_elements['y']
        if inds is not None:
            Ind_d = psi_elements['Ind_d'][inds]
            g = psi_elements['g'][inds]
            m = psi_elements['m'][inds]
            y = psi_elements['y'][inds]

        score = Ind_d * ((y <= coef) - g) / m + g - self._tau
        return score


    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        Ind_d = psi_elements['Ind_d']
        m = psi_elements['m']
        y = psi_elements['y']
        if inds is not None:
            Ind_d = psi_elements['Ind_d'][inds]
            m = psi_elements['m'][inds]
            y = psi_elements['y'][inds]

        score_weights = Ind_d / m
        normalization = score_weights.mean()

        if self._normalize:
            score_weights /= normalization
        if self._h is None:
            self._h = np.power(self._dml_data.n_obs, -0.2)
        u = (y - coef).reshape(-1, 1) / self._h
        kernel_est = np.exp(-1. * np.power(u, 2) / 2) / np.sqrt(2 * np.pi)
        #u_tilde = u * (np.abs(u) <= 1)
        #kernel_est = 0.75 * (1 - np.square(u_tilde))

        deriv = np.multiply(score_weights, kernel_est.reshape(-1,)) / self._h

        return deriv

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in ['ml_g', 'ml_m', 'ml_m_prelim']}


    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                             force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                             force_all_finite=False)


        #initialize nuisance predictions
        g_hat = np.full(shape=(self._dml_data.n_obs), fill_value=np.nan)
        m_hat = np.full(shape=(self._dml_data.n_obs), fill_value=np.nan)


        #caculate nuisance functions over different folds
        for i_fold in range(self.n_folds):
            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5, random_state=42)
            smpls_prelim = [(train, test) for train, test in KFold(n_splits=self.n_folds).split(train_inds_1)]

            d_train_1 = d[train_inds_1]
            y_train_1 = y[train_inds_1]
            x_train_1 = x[train_inds_1, :]
            m_hat_prelim = _dml_cv_predict(self._learner['ml_m_prelim'], x_train_1, d_train_1,
                                               method='predict_proba', smpls=smpls_prelim)['preds']

            m_hat_prelim[m_hat_prelim < self.trimming_threshold] = self.trimming_threshold
            m_hat_prelim[m_hat_prelim > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

            if self._treatment == 0:
                m_hat_prelim = 1 - m_hat_prelim

            # preliminary ipw estimate
            #TODO improve the solver (add bracket_guess)
            def ipw_score(theta):
                res = np.mean(self._compute_ipw_score(theta, d_train_1, y_train_1,  m_hat_prelim))
                return res

            def get_bracket_guess(coef_start, coef_bounds):
                max_bracket_length = coef_bounds[1] - coef_bounds[0]
                b_guess = coef_bounds
                delta = 0.1
                s_different = False
                while (not s_different) & (delta <= 1.0):
                    a = np.maximum(coef_start - delta * max_bracket_length/2, coef_bounds[0])
                    b = np.minimum(coef_start + delta * max_bracket_length/2, coef_bounds[1])
                    b_guess = (a, b)
                    f_a = ipw_score(b_guess[0])
                    f_b = ipw_score(b_guess[1])
                    s_different = (np.sign(f_a) != np.sign(f_b))
                    delta += 0.1
                return s_different, b_guess

            _, bracket_guess = get_bracket_guess(self._coef_start_val, self._coef_bounds)

            root_res = root_scalar(ipw_score,
                                   bracket=bracket_guess,
                                   method='brentq')
            ipw_est = root_res.root

            # readjust start value for minimization
            self._coef_start_val = ipw_est

            # use the preliminary estimates to fit the nuisance parameters on train_2
            d_train_2 = d[train_inds_2]
            y_train_2 = y[train_inds_2]
            x_train_2 = x[train_inds_2, :]

            dx_treat_train_2 = np.column_stack((d_train_2[d_train_2 == self._treatment],
                                                x_train_2[d_train_2 == self._treatment, :]))
            y_treat_train_2 = y_train_2[d_train_2 == self._treatment]
            self._learner['ml_g'].fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

            # predict nuisance values on the test data
            if self._treatment == 0:
                dx_test = np.column_stack((np.zeros_like(d[test_inds]), x[test_inds, :]))
            elif self._treatment == 1:
                dx_test = np.column_stack((np.ones_like(d[test_inds]), x[test_inds, :]))

            g_hat[test_inds] = self._learner['ml_g'].predict_proba(dx_test)[:, 1]

            # refit the propensity score on the whole training set
            self._learner['ml_m'].fit(x[train_inds, :], d[train_inds])
            m_hat[test_inds] = self._learner['ml_m'].predict_proba(x[test_inds, :])[:, self._treatment]

            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        psi_elements = {'Ind_d': d == self._treatment,
                        'g': g_hat,
                        'm': m_hat,
                        'y': y}
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}
        return psi_elements, preds


    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        pass


    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). ')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not(one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an IRM model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return

class DoubleMLQTE:
    """Double machine learning for quantile treatment effects
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 quantiles=0.5,
                 n_folds=5,
                 n_rep=1,
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 draw_sample_splitting=True):

        self._dml_data = obj_dml_data
        self._quantiles = np.asarray(quantiles).reshape((-1, ))
        self._check_quantile()
        self._n_quantiles = len(self._quantiles)

        self._n_folds = n_folds
        self._n_rep = n_rep

        self._dml_procedure = dml_procedure

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

        self._check_data(self._dml_data)
        self._check_quantile()

        #todo add crossfitting = False
        self._apply_cross_fitting = True

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}

        # initialize arrays according to obj_dml_data and the resampling settings
        self._psi0, self._psi1, self._psi0_deriv, self._psi1_deriv,\
        self._coef, self._se, self._all_coef, self._all_se, self._all_dml1_coef = self._initialize_arrays()

        # also initialize bootstrap arrays with the default number of bootstrap replications
        self._n_rep_boot, self._boot_coef, self._boot_t_stat = self._initialize_boot_arrays(n_rep_boot=500)

    @property
    def n_folds(self):
        """
        Number of folds.
        """
        return self._n_folds

    @property
    def n_rep(self):
        """
        Number of repetitions for the sample splitting.
        """
        return self._n_rep

    @property
    def smpls(self):
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            err_msg = ('Sample splitting not specified. Draw samples via .draw_sample splitting().' +
                       'External samples not implemented yet.')
            raise ValueError(err_msg)
        return self._smpls

    @property
    def n_quantiles(self):
        """
        Number of Quantiles.
        """
        return self._n_quantiles

    @property
    def dml_procedure(self):
        """
        The double machine learning algorithm.
        """
        return self._dml_procedure

    @property
    def coef(self):
        """
        Estimates for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._coef

    @property
    def all_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`.
        """
        return self._all_coef

    @coef.setter
    def coef(self, value):
        self._coef = value

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._se

    @se.setter
    def se(self, value):
        self._se = value

    @property
    def apply_cross_fitting(self):
        """
        Indicates whether cross-fitting should be applied.
        """
        return self._apply_cross_fitting

    @property
    def n_rep_boot(self):
        """
        The number of bootstrap replications.
        """
        return self._n_rep_boot

    @property
    def boot_coef(self):
        """
        Bootstrapped coefficients for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`.
        """
        return self._boot_coef

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`.
        """
        return self._boot_t_stat

    @property
    def __psi0(self):
        return self._psi0[:, self._i_rep, self._i_quant]

    @property
    def __psi0_deriv(self):
        return self._psi0_deriv[:, self._i_rep, self._i_quant]

    @property
    def __psi1(self):
        return self._psi1[:, self._i_rep, self._i_quant]

    @property
    def __psi1_deriv(self):
        return self._psi1_deriv[:, self._i_rep, self._i_quant]

    @property
    def __all_se(self):
        return self._all_se[self._i_quant, self._i_rep]


    def fit(self):
        for i_quant in range(self._n_quantiles):
            self._i_quant = i_quant
            # initialize models for both potential quantiles
            model_PQ_0 = DoubleMLPQ(self._dml_data,
                                    self._learner['ml_g'],
                                    self._learner['ml_m'],
                                    tau=self._quantiles[i_quant],
                                    treatment=0,
                                    n_folds=self.n_folds,
                                    n_rep=self.n_rep,
                                    draw_sample_splitting=False,
                                    apply_cross_fitting=self._apply_cross_fitting)
            model_PQ_1 = DoubleMLPQ(self._dml_data,
                                    self._learner['ml_g'],
                                    self._learner['ml_m'],
                                    tau=self._quantiles[i_quant],
                                    treatment=1,
                                    n_folds=self.n_folds,
                                    n_rep=self.n_rep,
                                    draw_sample_splitting=False,
                                    apply_cross_fitting=self._apply_cross_fitting)

            #synchronize the sample splitting
            model_PQ_0.set_sample_splitting(all_smpls=self.smpls)
            model_PQ_1.set_sample_splitting(all_smpls=self.smpls)

            model_PQ_0.fit()
            model_PQ_1.fit()

            # Quantile Treatment Effects
            self._all_coef[self._i_quant, :] = model_PQ_1.all_coef - model_PQ_0.all_coef

            # save scores and derivatives
            self._psi0[:, :, self._i_quant] = np.squeeze(model_PQ_0.psi, 2)
            self._psi1[:, :, self._i_quant] = np.squeeze(model_PQ_1.psi, 2)

            self._psi0_deriv[:, :, self._i_quant] = np.squeeze(model_PQ_0.psi_deriv, 2)
            self._psi1_deriv[:, :, self._i_quant] = np.squeeze(model_PQ_1.psi_deriv, 2)

            # Estimate the variance
            for i_rep in range(self.n_rep):
                self._i_rep = i_rep

                self._all_se[self._i_quant, self._i_rep] = self._se_causal_pars()

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self._agg_cross_fit()

        return self

    def bootstrap(self, method='normal', n_rep_boot=500):
        """
        Multiplier bootstrap for DoubleML models.

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
        if np.isnan(self.coef).all():
            raise ValueError('Apply fit() before bootstrap().')

        if (not isinstance(method, str)) | (method not in ['Bayes', 'normal', 'wild']):
            raise ValueError('Method must be "Bayes", "normal" or "wild". '
                             f'Got {str(method)}.')

        if not isinstance(n_rep_boot, int):
            raise TypeError('The number of bootstrap replications must be of int type. '
                            f'{str(n_rep_boot)} of type {str(type(n_rep_boot))} was passed.')
        if n_rep_boot < 1:
            raise ValueError('The number of bootstrap replications must be positive. '
                             f'{str(n_rep_boot)} was passed.')

        self._n_rep_boot, self._boot_coef, self._boot_t_stat = self._initialize_boot_arrays(n_rep_boot)

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            # draw weights for the bootstrap
            if self.apply_cross_fitting:
                n_obs = self._dml_data.n_obs
            else:
                # be prepared for the case of test sets of different size in repeated no-cross-fitting
                smpls = self.__smpls
                test_index = smpls[0][1]
                n_obs = len(test_index)
            weights = _draw_weights(method, n_rep_boot, n_obs)

            for i_quant in range(self.n_quantiles):
                self._i_quant = i_quant
                i_start = self._i_rep * self.n_rep_boot
                i_end = (self._i_rep + 1) * self.n_rep_boot
                self._boot_coef[self._i_quant, i_start:i_end], self._boot_t_stat[self._i_quant, i_start:i_end] =\
                    self._compute_bootstrap(weights)

        return self

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds``, ``n_rep`` and ``apply_cross_fitting``.

        Returns
        -------
        self : object
        """
        obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                n_rep=self.n_rep,
                                                n_obs=self._dml_data.n_obs,
                                                apply_cross_fitting=self.apply_cross_fitting)
        self._smpls = obj_dml_resampling.split_samples()

        return self

    def _compute_bootstrap(self, weights):
        if self.apply_cross_fitting:
            J0 = np.mean(self.__psi1_deriv)
            J1 = np.mean(self.__psi1_deriv)
            scaled_score = self.__psi1 / J1 - self.__psi0 / J0

            boot_coef = np.matmul(weights, scaled_score) / self._dml_data.n_obs
            boot_t_stat = np.matmul(weights, scaled_score) / (self._dml_data.n_obs * self.__all_se)
        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            smpls = self.__smpls
            test_index = smpls[0][1]
            J0 = np.mean(self.__psi1_deriv[test_index])
            J1 = np.mean(self.__psi1_deriv[test_index])
            scaled_score = self.__psi1[test_index] / J1 - self.__psi0[test_index] / J0

            boot_coef = np.matmul(weights, scaled_score) / len(test_index)
            boot_t_stat = np.matmul(weights, scaled_score) / (len(test_index) * self.__all_se)
        return boot_coef, boot_t_stat

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

        a = (1 - level)
        ab = np.array([a / 2, 1. - a / 2])
        if joint:
            if np.isnan(self.boot_coef).all():
                raise ValueError('Apply fit() & bootstrap() before confint(joint=True).')
            sim = np.amax(np.abs(self.boot_t_stat), 0)
            hatc = np.quantile(sim, 1 - a)
            ci = np.vstack((self.coef - self.se * hatc, self.coef + self.se * hatc)).T
        else:
            if np.isnan(self.coef).all():
                raise ValueError('Apply fit() before confint().')
            fac = norm.ppf(ab)
            ci = np.vstack((self.coef + self.se * fac[0], self.coef + self.se * fac[1])).T

        df_ci = pd.DataFrame(ci,
                             columns=['{:.1f} %'.format(i * 100) for i in ab],
                             index=self._quantiles)
        return df_ci


    def _agg_cross_fit(self):
        # aggregate parameters from the repeated cross-fitting
        # don't use the getter (always for one treatment variable and one sample), but the private variable
        self.coef = np.median(self._all_coef, 1)

        # TODO: In the documentation of standard errors we need to cleary state what we return here, i.e.,
        #  the asymptotic variance sigma_hat/N and not sigma_hat (which sometimes is also called the asympt var)!
        # TODO: In the edge case of repeated no-cross-fitting, the test sets might have different size and therefore
        #  it would note be valid to always use the same self._var_scaling_factor
        xx = np.tile(self.coef.reshape(-1, 1), self.n_rep)
        self.se = np.sqrt(np.divide(np.median(np.multiply(np.power(self._all_se, 2), self._var_scaling_factor) +
                                              np.power(self._all_coef - xx, 2), 1), self._var_scaling_factor))


    def _var_est(self):
        """
        Estimate the standard errors of the structural parameter
        """
        J0 = self._psi0_deriv[:, self._i_rep, self._i_rep].mean()
        J1 = self._psi1_deriv[:, self._i_rep, self._i_rep].mean()
        score0 = self._psi0[:, self._i_rep, self._i_rep]
        score1 = self._psi0[:, self._i_rep, self._i_rep]
        omega = score1 / J1 - score0 / J0

        if self.apply_cross_fitting:
            self._var_scaling_factor = self._dml_data.n_obs

        sigma2_hat = 1 / self._var_scaling_factor * np.mean(np.power(omega, 2))

        return sigma2_hat

    def _se_causal_pars(self):
        se = np.sqrt(self._var_est())
        return se


    def _initialize_arrays(self):
        psi0 = np.full((self._dml_data.n_obs, self.n_rep, self.n_quantiles), np.nan)
        psi0_deriv = np.full((self._dml_data.n_obs, self.n_rep, self.n_quantiles), np.nan)

        psi1 = np.full((self._dml_data.n_obs, self.n_rep, self.n_quantiles), np.nan)
        psi1_deriv = np.full((self._dml_data.n_obs, self.n_rep, self.n_quantiles), np.nan)

        coef = np.full(self.n_quantiles, np.nan)
        se = np.full(self.n_quantiles, np.nan)

        all_coef = np.full((self.n_quantiles, self.n_rep), np.nan)
        all_se = np.full((self.n_quantiles, self.n_rep), np.nan)

        if self.dml_procedure == 'dml1':
            if self.apply_cross_fitting:
                all_dml1_coef = np.full((self.n_quantiles, self.n_rep, self.n_folds), np.nan)
            else:
                all_dml1_coef = np.full((self.n_quantiles, self.n_rep, 1), np.nan)
        else:
            all_dml1_coef = None

        return psi0, psi1, psi0_deriv, psi1_deriv, coef, se, all_coef, all_se, all_dml1_coef

    def _initialize_boot_arrays(self, n_rep_boot):
        boot_coef = np.full((self.n_quantiles, n_rep_boot * self.n_rep), np.nan)
        boot_t_stat = np.full((self.n_quantiles, n_rep_boot * self.n_rep), np.nan)
        return n_rep_boot, boot_coef, boot_t_stat


    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). ')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not(one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an QTE model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return

    def _check_quantile(self):
        if self._quantiles.ndim > 1:
            raise ValueError(f'Quantile have to be of dimension 0 or 1.' +
                             f'Object of dimension {str(self._quantiles.ndim)} passed.')

        if np.any(self._quantiles <= 0) | np.any(self._quantiles >= 1):
            raise ValueError(f'Quantiles have be between 0 or 1.' +
                             f'Quantiles {str(self._quantiles)} passed.')
