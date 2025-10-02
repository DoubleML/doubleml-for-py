from doubleml.utils.resampling import DoubleMLClusterResampling, DoubleMLResampling


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

    def __init__(self):
        self.n_folds = 5
        self.n_rep = 1
        self._smpls = None
        self._smpls_cluster = None
        self._is_cluster_data = False
        self._n_folds_per_cluster = None
        self._n_obs_sample_splitting = None
        self._strata = None

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
            obj_dml_resampling = DoubleMLClusterResampling(
                n_folds=self._n_folds_per_cluster,
                n_rep=self.n_rep,
                n_obs=self._n_obs_sample_splitting,
                n_cluster_vars=self._dml_data.n_cluster_vars,
                cluster_vars=self._dml_data.cluster_vars,
            )
            self._smpls, self._smpls_cluster = obj_dml_resampling.split_samples()
        else:
            obj_dml_resampling = DoubleMLResampling(
                n_folds=self.n_folds, n_rep=self.n_rep, n_obs=self._n_obs_sample_splitting, stratify=self._strata
            )
            self._smpls = obj_dml_resampling.split_samples()

        return self
