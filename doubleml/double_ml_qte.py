import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.base import clone

from ._utils import _draw_weights, _check_zero_one_treatment, _check_score, _check_trimming
from ._utils_resampling import DoubleMLResampling
from .double_ml_data import DoubleMLData, DoubleMLClusterData
from .double_ml_pq import DoubleMLPQ
from .double_ml_lpq import DoubleMLLPQ


class DoubleMLQTE:
    """Double machine learning for quantile treatment effects

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function
         :math:`g_0(X) = E[Y <= \theta|X, D=d]`.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=d|X]`.

    quantiles : float or array_like
        Quantiles for treatment effect estimation. Entries have to be between ``0`` and ``1``.
        Default is ``0.5``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

    h : float or None
        The bandwidth to be used for the kernel density estimation of the score derivative.
        If ``None`` the bandwidth will be set to ``np.power(n_obs, -0.2)``, where ``n_obs`` is
        the number of observations in the sample.
        Default is ``1e-12``.

    normalize : bool
        Indicates whether to normalize weights in the estimation of the score derivative.
        Default is ``True``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m=None,
                 quantiles=0.5,
                 n_folds=5,
                 n_rep=1,
                 dml_procedure='dml2',
                 score='PQ',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 h=None,
                 normalize=True,
                 draw_sample_splitting=True):

        self._dml_data = obj_dml_data
        self._quantiles = np.asarray(quantiles).reshape((-1, ))
        self._check_quantile()
        self._n_quantiles = len(self._quantiles)
        self._h = h
        self._normalize = normalize
        self._n_folds = n_folds
        self._n_rep = n_rep
        self._dml_procedure = dml_procedure

        # check score
        self._score = score
        valid_scores = ['PQ', 'LPQ']
        _check_score(self.score, valid_scores)

        # check data
        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            self._is_cluster_data = True
        if self._is_cluster_data:
            raise NotImplementedError('Estimation with clustering not implemented.')
        self._check_data(self._dml_data)

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._check_quantile()

        # todo add crossfitting = False
        self._apply_cross_fitting = True

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()
        if self.score == 'PQ':
            self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
            self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
        elif self.score == 'LPQ':
            self._learner = {'ml_g': clone(ml_g)}
            self._predict_method = {'ml_g': 'predict_proba'}

        # initialize arrays according to obj_dml_data and the resampling settings
        self._psi0, self._psi1, self._psi0_deriv, self._psi1_deriv,\
            self._coef, self._se, self._all_coef, self._all_se, self._all_dml1_coef = self._initialize_arrays()

        # also initialize bootstrap arrays with the default number of bootstrap replications
        self._n_rep_boot, self._boot_coef, self._boot_t_stat = self._initialize_boot_arrays(n_rep_boot=500)

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Fit summary       ------------------\n' + fit_summary
        return res

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
            err_msg = ('Sample splitting not specified. Draw samples via .draw_sample splitting(). ' +
                       'External samples not implemented yet.')
            raise ValueError(err_msg)
        return self._smpls

    @property
    def quantiles(self):
        """
        Number of Quantiles.
        """
        return self._quantiles

    @property
    def n_quantiles(self):
        """
        Number of Quantiles.
        """
        return self._n_quantiles

    @property
    def score(self):
        """
        Number of Quantiles.
        """
        return self._score

    @property
    def dml_procedure(self):
        """
        The double machine learning algorithm.
        """
        return self._dml_procedure

    @property
    def h(self):
        """
        The bandwidth the kernel density estimation of the derivative.
        """
        return self._h

    @property
    def normalize(self):
        """
        Indicates of the weights in the derivative estimation should be normalized.
        """
        return self._normalize

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

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
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit`.
        """
        t_stat = self.coef / self.se
        return t_stat

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) after calling :meth:`fit`.
        """
        pval = 2 * norm.cdf(-np.abs(self.t_stat))
        return pval

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
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        col_names = ['coef', 'std err', 't', 'P>|t|']
        if np.isnan(self.coef).all():
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = np.transpose(np.vstack(
                [self.coef, self.se,
                 self.t_stat, self.pval]))
            df_summary = pd.DataFrame(summary_stats,
                                      columns=col_names,
                                      index=self.quantiles)
            ci = self.confint()
            df_summary = df_summary.join(ci)
        return df_summary

    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection.
    # The slicing is based on the two properties self._i_quant, the index of the quantile, and
    # self._i_rep, the index of the cross-fitting sample.

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

    def fit(self, n_jobs_cv=None):
        for i_quant in range(self._n_quantiles):
            self._i_quant = i_quant
            # initialize models for both potential quantiles
            if self.score == 'PQ':
                model_PQ_0 = DoubleMLPQ(self._dml_data,
                                        self._learner['ml_g'],
                                        self._learner['ml_m'],
                                        quantile=self._quantiles[i_quant],
                                        treatment=0,
                                        n_folds=self.n_folds,
                                        n_rep=self.n_rep,
                                        dml_procedure=self.dml_procedure,
                                        trimming_rule=self.trimming_rule,
                                        trimming_threshold=self.trimming_threshold,
                                        h=self.h,
                                        normalize=self.normalize,
                                        draw_sample_splitting=False,
                                        apply_cross_fitting=self._apply_cross_fitting)
                model_PQ_1 = DoubleMLPQ(self._dml_data,
                                        self._learner['ml_g'],
                                        self._learner['ml_m'],
                                        quantile=self._quantiles[i_quant],
                                        treatment=1,
                                        n_folds=self.n_folds,
                                        n_rep=self.n_rep,
                                        dml_procedure=self.dml_procedure,
                                        trimming_rule=self.trimming_rule,
                                        trimming_threshold=self.trimming_threshold,
                                        h=self.h,
                                        normalize=self.normalize,
                                        draw_sample_splitting=False,
                                        apply_cross_fitting=self._apply_cross_fitting)
            elif self.score == 'LPQ':
                model_PQ_0 = DoubleMLLPQ(self._dml_data,
                                         self._learner['ml_g'],
                                         quantile=self._quantiles[i_quant],
                                         treatment=0,
                                         n_folds=self.n_folds,
                                         n_rep=self.n_rep,
                                         dml_procedure=self.dml_procedure,
                                         trimming_rule=self.trimming_rule,
                                         trimming_threshold=self.trimming_threshold,
                                         h=self.h,
                                         normalize=self.normalize,
                                         draw_sample_splitting=False,
                                         apply_cross_fitting=self._apply_cross_fitting)
                model_PQ_1 = DoubleMLLPQ(self._dml_data,
                                         self._learner['ml_g'],
                                         quantile=self._quantiles[i_quant],
                                         treatment=1,
                                         n_folds=self.n_folds,
                                         n_rep=self.n_rep,
                                         dml_procedure=self.dml_procedure,
                                         trimming_rule=self.trimming_rule,
                                         trimming_threshold=self.trimming_threshold,
                                         h=self.h,
                                         normalize=self.normalize,
                                         draw_sample_splitting=False,
                                         apply_cross_fitting=self._apply_cross_fitting)

            # synchronize the sample splitting
            model_PQ_0.set_sample_splitting(all_smpls=self.smpls)
            model_PQ_1.set_sample_splitting(all_smpls=self.smpls)

            model_PQ_0.fit(n_jobs_cv=n_jobs_cv)
            model_PQ_1.fit(n_jobs_cv=n_jobs_cv)

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
                                                apply_cross_fitting=self.apply_cross_fitting,
                                                groups=self._dml_data.d)
        self._smpls = obj_dml_resampling.split_samples()

        return self

    def _compute_bootstrap(self, weights):
        if self.apply_cross_fitting:
            J0 = np.mean(self.__psi0_deriv)
            J1 = np.mean(self.__psi1_deriv)
            scaled_score = self.__psi1 / J1 - self.__psi0 / J0

            boot_coef = np.matmul(weights, scaled_score) / self._dml_data.n_obs
            boot_t_stat = np.matmul(weights, scaled_score) / (self._dml_data.n_obs * self.__all_se)
        else:
            # be prepared for the case of test sets of different size in repeated no-cross-fitting
            smpls = self.__smpls
            test_index = smpls[0][1]
            J0 = np.mean(self.__psi0_deriv[test_index])
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
                            f'Object of type {str(type(level))} was passed.')
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
        J0 = self._psi0_deriv[:, self._i_rep, self._i_quant].mean()
        J1 = self._psi1_deriv[:, self._i_rep, self._i_quant].mean()
        score0 = self._psi0[:, self._i_rep, self._i_quant]
        score1 = self._psi1[:, self._i_rep, self._i_quant]
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
        _check_zero_one_treatment(self)
        return

    def _check_quantile(self):
        if np.any(self.quantiles <= 0) | np.any(self.quantiles >= 1):
            raise ValueError('Quantiles have be between 0 or 1. ' +
                             f'Quantiles {str(self.quantiles)} passed.')
