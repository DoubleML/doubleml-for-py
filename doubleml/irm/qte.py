import numpy as np
import pandas as pd

from sklearn.base import clone

from joblib import Parallel, delayed

from ..double_ml_data import DoubleMLData, DoubleMLClusterData
from .pq import DoubleMLPQ
from .lpq import DoubleMLLPQ
from .cvar import DoubleMLCVAR
from ..double_ml_framework import concat

from ..utils._estimation import _default_kde
from ..utils.resampling import DoubleMLResampling
from ..utils._checks import _check_score, _check_trimming, _check_zero_one_treatment, _check_sample_splitting

from ..utils._descriptive import generate_summary


class DoubleMLQTE:
    """Double machine learning for quantile treatment effects

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance elements which depend on priliminary estimation.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the propensity nuisance functions.

    quantiles : float or array_like
        Quantiles for treatment effect estimation. Entries have to be between ``0`` and ``1``.
        Default is ``0.5``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'PQ'``, ``'LPQ'`` or ``'CVaR'``) specifying the score function.
        Default is ``'PQ'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    kde : callable or None
        A callable object / function with signature ``deriv = kde(u, weights)`` for weighted kernel density estimation.
        Here ``deriv`` should evaluate the density in ``0``.
        Default is ``'None'``, which uses :py:class:`statsmodels.nonparametric.kde.KDEUnivariate` with a
        gaussian kernel and silverman for bandwidth determination.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_qte_obj = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, quantiles=[0.25, 0.5, 0.75])
    >>> dml_qte_obj.fit().summary
              coef   std err         t     P>|t|     2.5 %    97.5 %
    0.25  0.274825  0.347310  0.791297  0.428771 -0.405890  0.955541
    0.50  0.449150  0.192539  2.332782  0.019660  0.071782  0.826519
    0.75  0.709606  0.193308  3.670867  0.000242  0.330731  1.088482
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m=None,
                 quantiles=0.5,
                 n_folds=5,
                 n_rep=1,
                 score='PQ',
                 normalize_ipw=True,
                 kde=None,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):

        self._dml_data = obj_dml_data
        self._quantiles = np.asarray(quantiles).reshape((-1, ))
        self._check_quantile()
        self._n_quantiles = len(self._quantiles)

        if kde is None:
            self._kde = _default_kde
        else:
            if not callable(kde):
                raise TypeError('kde should be either a callable or None. '
                                '%r was passed.' % kde)
            self._kde = kde

        self._normalize_ipw = normalize_ipw
        self._n_folds = n_folds
        self._n_rep = n_rep

        # check score
        self._score = score
        valid_scores = ['PQ', 'LPQ', 'CVaR']
        _check_score(self.score, valid_scores, allow_callable=False)

        # check data
        self._is_cluster_data = False
        if isinstance(obj_dml_data, DoubleMLClusterData):
            self._is_cluster_data = True
        self._check_data(self._dml_data)

        # initialize framework which is constructed after the fit method is called
        self._framework = None

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()
            # initialize all models
            self._modellist_0, self._modellist_1 = self._initialize_models()

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
    def n_rep_boot(self):
        """
        The number of bootstrap replications.
        """
        if self._framework is None:
            n_rep_boot = None
        else:
            n_rep_boot = self._framework.n_rep_boot
        return n_rep_boot

    @property
    def boot_method(self):
        """
        The method to construct the bootstrap replications.
        """
        if self._framework is None:
            method = None
        else:
            method = self._framework.boot_method
        return method

    @property
    def smpls(self):
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            err_msg = ('Sample splitting not specified. Either draw samples via .draw_sample splitting() ' +
                       'or set external samples via .set_sample_splitting().')
            raise ValueError(err_msg)
        return self._smpls

    @property
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

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
        The score function.
        """
        return self._score

    @property
    def kde(self):
        """
        The kernel density estimation of the derivative.
        """
        return self._kde

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

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
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape (``n_quantiles``,)).
        """
        if self._framework is None:
            coef = None
        else:
            coef = self.framework.thetas
        return coef

    @property
    def all_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`
         (shape (``n_quantiles``, ``n_rep``)).
        """
        if self._framework is None:
            all_coef = None
        else:
            all_coef = self.framework.all_thetas
        return all_coef

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (``n_quantiles``,)).
        """
        if self._framework is None:
            se = None
        else:
            se = self.framework.ses
        return se

    @property
    def all_se(self):
        """
        Standard errors of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`
         (shape (``n_quantiles``, ``n_rep``)).
        """
        if self._framework is None:
            all_se = None
        else:
            all_se = self.framework.all_ses
        return all_se

    @property
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit` (shape (``n_quantiles``,)).
        """
        t_stat = self.coef / self.se
        return t_stat

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) (shape (``n_quantiles``,)).
        """
        return self.framework.pvals

    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`
         (shape (``n_rep_boot``, ``n_quantiles``, ``n_rep``)).
        """
        if self._framework is None:
            boot_t_stat = None
        else:
            boot_t_stat = self._framework.boot_t_stat
        return boot_t_stat

    @property
    def modellist_0(self):
        """
        List of the models for the control group (``treatment==0``).
        """
        return self._modellist_0

    @property
    def modellist_1(self):
        """
        List of the models for the treatment group (``treatment==1``).
        """
        return self._modellist_1

    @property
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        if self.framework is None:
            col_names = ['coef', 'std err', 't', 'P>|t|']
            df_summary = pd.DataFrame(columns=col_names)
        else:
            ci = self.confint()
            df_summary = generate_summary(self.coef, self.se, self.t_stat,
                                          self.pval, ci, self.quantiles)
        return df_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        """
        Estimate DoubleMLQTE models.

        Parameters
        ----------
        n_jobs_models : None or int
            The number of CPUs to use to fit the quantiles. ``None`` means ``1``.
            Default is ``None``.

        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Does not speed up computation for quantile models.
            Default is ``None``.

        store_predictions : bool
            Indicates whether the predictions for the nuisance functions should be stored in ``predictions``.
            Default is ``True``.

        store_models : bool
            Indicates whether the fitted models for the nuisance functions should be stored in ``models``. This allows
            to analyze the fitted models or extract information like variable importance.
            Default is ``False``.

        Returns
        -------
        self : object
        """

        if external_predictions is not None:
            raise NotImplementedError(f"External predictions not implemented for {self.__class__.__name__}.")

        # parallel estimation of the quantiles
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch='2*n_jobs')
        fitted_models = parallel(delayed(self._fit_quantile)(i_quant, n_jobs_cv, store_predictions, store_models)
                                 for i_quant in range(self.n_quantiles))

        # combine the estimates and scores
        framework_list = [None] * self.n_quantiles

        for i_quant in range(self.n_quantiles):
            # save the parallel fitted models in the right list
            self._modellist_0[i_quant] = fitted_models[i_quant][0]
            self._modellist_1[i_quant] = fitted_models[i_quant][1]

            # set up the framework
            framework_list[i_quant] = self._modellist_1[i_quant].framework - \
                self._modellist_0[i_quant].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

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
        if self._framework is None:
            raise ValueError('Apply fit() before bootstrap().')
        self._framework.bootstrap(method=method, n_rep_boot=n_rep_boot)

        return self

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """
        obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                n_rep=self.n_rep,
                                                n_obs=self._dml_data.n_obs,
                                                stratify=self._dml_data.d)
        self._smpls = obj_dml_resampling.split_samples()
        # initialize all models
        self._modellist_0, self._modellist_1 = self._initialize_models()

        return self

    def set_sample_splitting(self, all_smpls, all_smpls_cluster=None):
        """
        Set the sample splitting for DoubleML models.

        The  attributes ``n_folds`` and ``n_rep`` are derived from the provided partition.

        Parameters
        ----------
        all_smpls : list or tuple
            If nested list of lists of tuples:
                The outer list needs to provide an entry per repeated sample splitting (length of list is set as
                ``n_rep``).
                The inner list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as
                ``n_folds``). test_ind must form a partition for each inner list.
            If list of tuples:
                The list needs to provide a tuple (train_ind, test_ind) per fold (length of list is set as
                ``n_folds``). test_ind must form a partition. ``n_rep=1`` is always set.
            If tuple:
                Must be a tuple with two elements train_ind and test_ind. Only viable option is to set
                train_ind and test_ind to np.arange(n_obs), which corresponds to no sample splitting.
                ``n_folds=1`` and ``n_rep=1`` is always set.

        all_smpls_cluster : list or None
            Nested list or ``None``. The first level of nesting corresponds to the number of repetitions. The second level
            of nesting corresponds to the number of folds. The third level of nesting contains a tuple of training and
            testing lists. Both training and testing contain an array for each cluster variable, which form a partition of
            the clusters.
            Default is ``None``.

        Returns
        -------
        self : object

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from doubleml.datasets import make_plr_CCDDHNR2018
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.base import clone
        >>> np.random.seed(3141)
        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_g = learner
        >>> ml_m = learner
        >>> obj_dml_data = make_plr_CCDDHNR2018(n_obs=10, alpha=0.5)
        >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        >>> dml_plr_obj.set_sample_splitting(smpls)
        >>> # sample splitting with two folds and cross-fitting
        >>> smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        >>>          ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])]
        >>> dml_plr_obj.set_sample_splitting(smpls)
        >>> # sample splitting with two folds and repeated cross-fitting with n_rep = 2
        >>> smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        >>>           ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
        >>>          [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
        >>>           ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
        >>> dml_plr_obj.set_sample_splitting(smpls)
        """
        self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
            all_smpls, all_smpls_cluster, self._dml_data, self._is_cluster_data)

        # initialize all models
        self._modellist_0, self._modellist_1 = self._initialize_models()

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

        if self.framework is None:
            raise ValueError('Apply fit() before confint().')

        df_ci = self.framework.confint(joint=joint, level=level)
        df_ci.set_index(pd.Index(self._quantiles), inplace=True)

        return df_ci

    def p_adjust(self, method='romano-wolf'):
        """
        Multiple testing adjustment for DoubleML models.

        Parameters
        ----------
        method : str
            A str (``'romano-wolf''``, ``'bonferroni'``, ``'holm'``, etc) specifying the adjustment method.
            In addition to ``'romano-wolf''``, all methods implemented in
            :py:func:`statsmodels.stats.multitest.multipletests` can be applied.
            Default is ``'romano-wolf'``.

        Returns
        -------
        p_val : pd.DataFrame
            A data frame with adjusted p-values.
        """

        if self.framework is None:
            raise ValueError('Apply fit() before p_adjust().')

        p_val, _ = self.framework.p_adjust(method=method)
        p_val.set_index(pd.Index(self._quantiles), inplace=True)

        return p_val

    def _fit_quantile(self, i_quant, n_jobs_cv=None, store_predictions=True, store_models=False):

        model_0 = self.modellist_0[i_quant]
        model_1 = self.modellist_1[i_quant]

        model_0.fit(n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models)
        model_1.fit(n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models)

        return model_0, model_1

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

    def _initialize_models(self):
        modellist_0 = [None] * self.n_quantiles
        modellist_1 = [None] * self.n_quantiles
        kwargs = {
            'obj_dml_data': self._dml_data,
            'ml_g': self._learner['ml_g'],
            'ml_m': self._learner['ml_m'],
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'trimming_rule': self.trimming_rule,
            'trimming_threshold': self.trimming_threshold,
            'normalize_ipw': self.normalize_ipw,
            'draw_sample_splitting': False
        }
        for i_quant in range(self.n_quantiles):

            # initialize models for both potential quantiles
            if self.score == 'PQ':
                model_0 = DoubleMLPQ(quantile=self._quantiles[i_quant],
                                     treatment=0,
                                     kde=self.kde,
                                     **kwargs)
                model_1 = DoubleMLPQ(quantile=self._quantiles[i_quant],
                                     treatment=1,
                                     kde=self.kde,
                                     **kwargs)
            elif self.score == 'LPQ':
                model_0 = DoubleMLLPQ(quantile=self._quantiles[i_quant],
                                      treatment=0,
                                      kde=self.kde,
                                      **kwargs)
                model_1 = DoubleMLLPQ(quantile=self._quantiles[i_quant],
                                      treatment=1,
                                      kde=self.kde,
                                      **kwargs)

            elif self.score == 'CVaR':
                model_0 = DoubleMLCVAR(quantile=self._quantiles[i_quant],
                                       treatment=0,
                                       **kwargs)
                model_1 = DoubleMLCVAR(quantile=self._quantiles[i_quant],
                                       treatment=1,
                                       **kwargs)

            # synchronize the sample splitting
            model_0.set_sample_splitting(all_smpls=self.smpls)
            model_1.set_sample_splitting(all_smpls=self.smpls)

            modellist_0[i_quant] = model_0
            modellist_1[i_quant] = model_1

        return modellist_0, modellist_1
