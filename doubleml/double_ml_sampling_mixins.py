from abc import abstractmethod

from doubleml.utils._checks import _check_sample_splitting
from doubleml.utils.resampling import DoubleMLClusterResampling, DoubleMLDoubleResampling, DoubleMLResampling


class SampleSplittingMixin:
    """
    Mixin class implementing sample splitting for DoubleML models.

    Notes
    -----
    The mixin class :class:`SampleSplittingMixin` implements the sample splitting procedure for DoubleML models.
    The sample splitting is drawn according to the attributes ``n_folds`` and ``n_rep``.
    If the data is clustered, the sample splitting is drawn such that clusters are not split across folds.
    For details, see the chapter on
    `sample splitting <https://docs.doubleml.org/stable/guide/resampling.html>`_ in the DoubleML user guide.
    """

    _double_sample_splitting = False

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """
        if self._is_cluster_data:
            if self._double_sample_splitting:
                raise ValueError("Cluster data not supported for double sample splitting.")
            obj_dml_resampling = DoubleMLClusterResampling(
                n_folds=self._n_folds_per_cluster,
                n_rep=self.n_rep,
                n_obs=self._n_obs_sample_splitting,
                n_cluster_vars=self._dml_data.n_cluster_vars,
                cluster_vars=self._dml_data.cluster_vars,
            )
            self._smpls, self._smpls_cluster = obj_dml_resampling.split_samples()
        else:
            if self._double_sample_splitting:
                obj_dml_resampling = DoubleMLDoubleResampling(
                    n_folds=self.n_folds,
                    n_folds_inner=self.n_folds_inner,
                    n_rep=self.n_rep,
                    n_obs=self._dml_data.n_obs,
                    stratify=self._strata,
                )
                self._smpls, self._smpls_inner = obj_dml_resampling.split_samples()
            else:
                obj_dml_resampling = DoubleMLResampling(
                    n_folds=self.n_folds, n_rep=self.n_rep, n_obs=self._n_obs_sample_splitting, stratify=self._strata
                )
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
        >>> from doubleml.plm.datasets import make_plr_CCDDHNR2018
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
        ...          ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])]
        >>> dml_plr_obj.set_sample_splitting(smpls) # doctest: +ELLIPSIS
        <doubleml.plm.plr.DoubleMLPLR object at 0x...>
        >>> # sample splitting with two folds and repeated cross-fitting with n_rep = 2
        >>> smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        ...           ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
        ...          [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
        ...           ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
        >>> dml_plr_obj.set_sample_splitting(smpls) # doctest: +ELLIPSIS
        <doubleml.plm.plr.DoubleMLPLR object at 0x...>
        """
        if self._double_sample_splitting:
            raise ValueError("set_sample_splitting not supported for double sample splitting.")

        self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
            all_smpls, all_smpls_cluster, self._dml_data, self._is_cluster_data, n_obs=self._n_obs_sample_splitting
        )

        self._initialize_dml_model()

        return self

    @abstractmethod
    def _initialize_dml_model(self):
        """
        Set sample splitting for DoubleML models. Can update the number of repetitions.
        Updates model dimensions to (n_folds, n_rep).
        This method needs to be implemented in the child class.
        """
