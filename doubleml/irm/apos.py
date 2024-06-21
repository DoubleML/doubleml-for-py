import numpy as np
import pandas as pd

from sklearn.base import clone

from joblib import Parallel, delayed

from ..double_ml_data import DoubleMLData, DoubleMLClusterData
from .apo import DoubleMLAPO
from ..double_ml_framework import concat

from ..utils.resampling import DoubleMLResampling
from ..utils._checks import _check_score, _check_trimming


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
        self._n_levels = len(self._treatment_levels)

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

        # perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

        self._learner = {'ml_g': clone(ml_g), 'ml_m': clone(ml_m)}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}

        # initialize all models
        self._modellist = self._initialize_models()

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def n_levels(self):
        """
        The number of treatment levels.
        """
        return self._n_levels

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
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

    @property
    def modellist(self):
        """
        The list of models for each level.
        """
        return self._modellist

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

        Returns
        -------
        self : object
        """

        if external_predictions is not None:
            raise NotImplementedError(f"External predictions not implemented for {self.__class__.__name__}.")

        # parallel estimation of the quantiles
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch='2*n_jobs')
        fitted_models = parallel(delayed(self._fit_model)(i_level, n_jobs_cv, store_predictions, store_models)
                                 for i_level in range(self.n_treatment_levels))

        # combine the estimates and scores
        framework_list = [None] * self.n_levels

        for i_level in range(self.n_levels):
            self._modellist[i_level] = fitted_models[i_level][0]
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

    def _fit_model(self, i_level, n_jobs_cv=None, store_predictions=True, store_models=False):

        model = self.modellist_0[i_level]
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

    def _initialize_models(self):
        modellist = [None] * self.n_levels
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
        for i_level in range(self.n_levels):
            # initialize models for all levels
            model = DoubleMLAPO(
                treatment_level=self._treatment_levels[i_level],
                **kwargs
            )

            # synchronize the sample splitting
            model.set_sample_splitting(all_smpls=self.smpls)
            modellist[i_level] = model

        return modellist
