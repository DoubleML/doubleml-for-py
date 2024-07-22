import numpy as np
import pandas as pd

from sklearn.base import clone

from joblib import Parallel, delayed

from ..double_ml_data import DoubleMLData, DoubleMLClusterData
from .apo import DoubleMLAPO
from ..double_ml_framework import concat

from ..utils.resampling import DoubleMLResampling
from ..utils._descriptive import generate_summary
from ..utils._checks import _check_score, _check_trimming, _check_weights, _check_sample_splitting


class DoubleMLAPOS:
    """Double machine learning for interactive regression models with multiple discrete treatments.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment_levels,
                 n_folds=5,
                 n_rep=1,
                 score='APO',
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):

        self._dml_data = obj_dml_data
        self._is_cluster_data = isinstance(obj_dml_data, DoubleMLClusterData)
        self._check_data(self._dml_data)

        self._treatment_levels = np.asarray(treatment_levels).reshape((-1, ))
        self._check_treatment_levels()
        self._n_treatment_levels = len(self._treatment_levels)

        self._normalize_ipw = normalize_ipw
        self._n_folds = n_folds
        self._n_rep = n_rep

        # check score
        self._score = score
        valid_scores = ['APO']
        _check_score(self.score, valid_scores, allow_callable=False)

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
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}

        # APO weights
        _check_weights(weights, score="ATE", n_obs=obj_dml_data.n_obs, n_rep=self.n_rep)
        self._initialize_weights(weights)

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

            # initialize all models if splits are known
            self._modellist = self._initialize_models()

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Fit summary       ------------------\n' + fit_summary
        return res

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def n_treatment_levels(self):
        """
        The number of treatment levels.
        """
        return self._n_treatment_levels

    @property
    def treatment_levels(self):
        """
        The evaluated treatment levels.
        """
        return self._treatment_levels

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
    def weights(self):
        """
        Specifies the weights for a weighted average potential outcome.
        """
        return self._weights

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
    def coef(self):
        """
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
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
         (shape (``n_treatment_levels``, ``n_rep``)).
        """
        if self._framework is None:
            all_coef = None
        else:
            all_coef = self.framework.all_thetas
        return all_coef

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
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
         (shape (``n_treatment_levels``, ``n_rep``)).
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
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

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
    def modellist(self):
        """
        The list of models for each level.
        """
        return self._modellist

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
                                          self.pval, ci, self._treatment_levels)
        return df_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        """
        Estimate DoubleMLAPOS models.

        Parameters
        ----------
        n_jobs_models : None or int
            The number of CPUs to use to fit the treatment_levels. ``None`` means ``1``.
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

        external_predictions : None
            Not implemented for DoubleMLAPOS.

        Returns
        -------
        self : object
        """

        if external_predictions is not None:
            raise NotImplementedError(f"External predictions not implemented for {self.__class__.__name__}.")

        # parallel estimation of the models
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch='2*n_jobs')
        fitted_models = parallel(delayed(self._fit_model)(i_level, n_jobs_cv, store_predictions, store_models)
                                 for i_level in range(self.n_treatment_levels))

        # combine the estimates and scores
        framework_list = [None] * self.n_treatment_levels

        for i_level in range(self.n_treatment_levels):
            self._modellist[i_level] = fitted_models[i_level]
            framework_list[i_level] = self._modellist[i_level].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

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
        df_ci.set_index(pd.Index(self._treatment_levels), inplace=True)

        return df_ci

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

        self._modellist = self._initialize_models()

        return self

    def _fit_model(self, i_level, n_jobs_cv=None, store_predictions=True, store_models=False):

        model = self.modellist[i_level]
        model.fit(n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models)
        return model

    def _check_treatment_levels(self):
        if not np.all(np.isin(self._treatment_levels, np.unique(self._dml_data.d))):
            raise ValueError('The treatment levels have to be a subset of the unique treatment levels in the data.')

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData or DoubleMLClusterData type.')
        if obj_dml_data.z is not None:
            raise ValueError('The data must not contain instrumental variables.')
        return

    def _initialize_weights(self, weights):
        if weights is None:
            weights = np.ones(self._dml_data.n_obs)
        if isinstance(weights, np.ndarray):
            self._weights = weights
        else:
            assert isinstance(weights, dict)
            self._weights = weights

    def _initialize_models(self):
        modellist = [None] * self.n_treatment_levels
        kwargs = {
            'obj_dml_data': self._dml_data,
            'ml_g': self._learner['ml_g'],
            'ml_m': self._learner['ml_m'],
            'score': self.score,
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'weights': self.weights,
            'trimming_rule': self.trimming_rule,
            'trimming_threshold': self.trimming_threshold,
            'normalize_ipw': self.normalize_ipw,
            'draw_sample_splitting': False
        }
        for i_level in range(self.n_treatment_levels):
            # initialize models for all levels
            model = DoubleMLAPO(
                treatment_level=self._treatment_levels[i_level],
                **kwargs
            )

            # synchronize the sample splitting
            model.set_sample_splitting(all_smpls=self.smpls)
            modellist[i_level] = model

        return modellist
