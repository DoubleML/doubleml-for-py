import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from typing import Optional, List, Tuple, Callable


class DoubleMLResampling:
    """
    A class for flexible repeated cross-validation or custom splitting,
    especially for panel/time-series data.

    Parameters
    ----------
    n_folds : int
        Number of folds (splits) per repetition.
    n_rep : int
        Number of repetitions.
    n_obs : int
        Total number of observations in `data`.
    data : np.ndarray or pd.DataFrame
        The dataset from which splits are generated. Must support indexing via
        `data[col_name]` or similar if using custom splits.
    t_col : str, optional
        Column name for time dimension (used in time-based splits).
    u_col : str, optional
        Column name for 'unit' dimension (used in unit-based splits).
    stratify : np.ndarray, optional
        Labels for stratification. If not None, uses `RepeatedStratifiedKFold`
        when `split_strategy='random'`.
    split_strategy : {'random', 'by_unit', 'by_time', 'time_adjacent', 'neighbors_left_out'}, default='random'
        The splitting strategy to use.
    random_state : int, optional
        If provided, used to seed the RNG for reproducible random splits.

    Attributes
    ----------
    resampling : generator or List[List[Tuple[np.ndarray, np.ndarray]]]
        The scikit-learn cross-validator (if `random` strategy)
        or a nested list-of-lists of splits (custom strategies).
    """

    def __init__(
        self,
        n_folds: int,
        n_rep: int,
        n_obs: int,
        data,
        t_col: Optional[str] = None,
        u_col: Optional[str] = None,
        stratify=None,
        split_strategy: str = "random",
        random_state: Optional[int] = None,
    ):
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs
        self.data = data
        self.t_col = t_col
        self.u_col = u_col
        self.stratify = stratify
        self.split_strategy = split_strategy
        self.random_state = random_state

        # Basic input check
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

        # Set up a NumPy random generator if needed
        self.rng = np.random.default_rng(random_state)

        # Built-in scikit-learn strategies
        if self.split_strategy == "random":
            self._init_random_resampling()

        # Custom strategies
        elif self.split_strategy in [
            "by_unit",
            "by_time",
            "time_adjacent",
            "neighbors_left_out",
        ]:
            self.resampling = self._generate_repeated_splits(
                getattr(self, f"_split_{self.split_strategy}")
            )
        else:
            raise ValueError(f"Unsupported split strategy: {self.split_strategy}")

    def _init_random_resampling(self):
        """
        Internal helper to initialize scikit-learn's cross-validation
        (RepeatedKFold or RepeatedStratifiedKFold).
        """
        if self.stratify is None:
            # Note: RepeatedKFold doesn't directly take random_state in older
            # scikit-learn versions, but recent versions do.
            self.resampling = RepeatedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_rep,
                random_state=self.random_state
            )
        else:
            # RepeatedStratifiedKFold requires discrete/categorical `stratify`.
            self.resampling = RepeatedStratifiedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_rep,
                random_state=self.random_state
            )

    def split_samples(self) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Return the generated splits for each repetition and fold.

        Returns
        -------
        smpls : list of list of tuples
            For each repetition, a list of (train_idx, test_idx) pairs.
            So, `smpls[rep][fold] = (train_idx, test_idx)`.
        """
        if self.split_strategy == "random":
            # scikit-learn's RepeatedKFold or RepeatedStratifiedKFold
            all_smpls = [
                (train, test)
                for train, test in self.resampling.split(
                    X=np.zeros(self.n_obs), y=self.stratify
                )
            ]
            # `all_smpls` is a flat list of length (n_rep * n_folds).
            return [
                all_smpls[i_repeat * self.n_folds : (i_repeat + 1) * self.n_folds]
                for i_repeat in range(self.n_rep)
            ]
        else:
            # Already constructed as a list-of-lists by _generate_repeated_splits
            return self.resampling

    def _generate_repeated_splits(
        self, split_fn: Callable[[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Repeatedly generate splits using the specified single-repetition function.

        Parameters
        ----------
        split_fn : callable
            A function that takes a 1D array of row indices and returns
            a list of (train_idx, test_idx) for each fold.

        Returns
        -------
        splits : list of list of tuples
            A nested list, where `splits[rep][fold] = (train_idx, test_idx)`.
        """
        indices = np.arange(self.n_obs)
        all_reps = []
        for _ in range(self.n_rep):
            rep_splits = split_fn(indices)
            all_reps.append(rep_splits)
        return all_reps

    # -------------------------------------------------------------------------
    # Custom single-repetition split functions
    # -------------------------------------------------------------------------

    def _split_by_unit(self, indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split based on unique units in `u_col` for a single repetition.

        - Shuffles the unique units, then partitions them into folds.
        - All observations with a given unit are exclusively in train or test.

        Parameters
        ----------
        indices : np.ndarray
            Array of indices [0..n_obs-1].

        Returns
        -------
        rep_splits : list of tuples
            A list of (train_idx, test_idx) for each fold.
        """
        if self.u_col is None:
            raise ValueError("`u_col` must be provided for 'by_unit' split strategy.")

        unique_units = np.unique(self.data[self.u_col])
        self.rng.shuffle(unique_units)  # shuffle in-place

        # Partition the shuffled units into n_folds
        folds = np.array_split(unique_units, self.n_folds)
        rep_splits = []
        for fold_units in folds:
            test_mask = np.isin(self.data[self.u_col], fold_units)
            train_mask = ~test_mask
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            rep_splits.append((train_idx, test_idx))

        return rep_splits

    def _split_by_time(self, indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split based on unique time periods in `t_col`, for a single repetition.
        - Shuffles the unique time periods, then partitions them into folds.
        - This is a 'block-based' approach, but with random ordering of time blocks.

        Parameters
        ----------
        indices : np.ndarray
            Array of indices [0..n_obs-1].

        Returns
        -------
        rep_splits : list of tuples
            A list of (train_idx, test_idx) for each fold.
        """
        if self.t_col is None:
            raise ValueError("`t_col` must be provided for 'by_time' split strategy.")

        unique_times = np.unique(self.data[self.t_col])
        self.rng.shuffle(unique_times)  # shuffle in-place

        folds = np.array_split(unique_times, self.n_folds)
        rep_splits = []
        for fold_times in folds:
            test_idx = np.where(np.isin(self.data[self.t_col], fold_times))[0]
            train_idx = np.where(~np.isin(self.data[self.t_col], fold_times))[0]
            rep_splits.append((train_idx, test_idx))

        return rep_splits

    def _split_time_adjacent(
        self, indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create 'adjacent' time-based splits for a single repetition.

        - Sorts unique times, then for fold i, picks every nth time period
          starting at `i` (i.e., i, i+n_folds, i+2*n_folds, ...).
        - Non-random; each repetition returns the same set of folds.

        Parameters
        ----------
        indices : np.ndarray
            Array of indices [0..n_obs-1].

        Returns
        -------
        rep_splits : list of tuples
            A list of (train_idx, test_idx) for each fold.
        """
        if self.t_col is None:
            raise ValueError("`t_col` must be provided for 'time_adjacent' strategy.")

        unique_times = np.unique(self.data[self.t_col])
        unique_times.sort()

        rep_splits = []
        for i in range(self.n_folds):
            test_periods = unique_times[i :: self.n_folds]
            test_idx = np.where(np.isin(self.data[self.t_col], test_periods))[0]
            train_idx = np.where(~np.isin(self.data[self.t_col], test_periods))[0]
            rep_splits.append((train_idx, test_idx))

        return rep_splits

    def _split_neighbors_left_out(
        self, indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Leave out neighboring time intervals for each fold (single repetition).
        - Sequentially partition times into folds in sorted order.
        - For each fold, exclude `neighbors` intervals on each side
          from the training set.

        Parameters
        ----------
        indices : np.ndarray
            Array of indices [0..n_obs-1].

        Returns
        -------
        rep_splits : list of tuples
            A list of (train_idx, test_idx) for each fold.
        """
        if self.t_col is None:
            raise ValueError("`t_col` must be provided for 'neighbors_left_out' strategy.")

        neighbors = 5  # <-- You may expose this as a user parameter if desired.

        unique_times = np.unique(self.data[self.t_col])
        n_times = len(unique_times)

        # Fold sizes
        fold_sizes = np.full(self.n_folds, n_times // self.n_folds, dtype=int)
        fold_sizes[: n_times % self.n_folds] += 1

        rep_splits = []
        current = 0
        for fold_size in fold_sizes:
            test_start = current
            test_stop = current + fold_size

            test_times = unique_times[test_start:test_stop]
            test_idx = np.where(np.isin(self.data[self.t_col], test_times))[0]

            # Exclude neighbors from training
            excluded_times = unique_times[
                max(0, test_start - neighbors) : min(n_times, test_stop + neighbors)
            ]
            train_idx = np.where(~np.isin(self.data[self.t_col], excluded_times))[0]

            rep_splits.append((train_idx, test_idx))
            current = test_stop

        return rep_splits


class DoubleMLClusterResampling:
    def __init__(self,
                 n_folds,
                 n_rep,
                 n_obs,
                 n_cluster_vars,
                 cluster_vars):

        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs

        assert cluster_vars.shape[0] == n_obs
        assert cluster_vars.shape[1] == n_cluster_vars
        self.n_cluster_vars = n_cluster_vars
        self.cluster_vars = cluster_vars
        self.resampling = KFold(n_splits=n_folds, shuffle=True)

    def split_samples(self):
        all_smpls = []
        all_smpls_cluster = []
        for _ in range(self.n_rep):
            smpls_cluster_vars = []
            for i_var in range(self.n_cluster_vars):
                this_cluster_var = self.cluster_vars[:, i_var]
                clusters = np.unique(this_cluster_var)
                n_clusters = len(clusters)
                smpls_cluster_vars.append([(clusters[train], clusters[test])
                                           for train, test in self.resampling.split(np.zeros(n_clusters))])

            smpls = []
            smpls_cluster = []
            # build the cartesian product
            cart = np.array(np.meshgrid(*[np.arange(self.n_folds)
                                          for i in range(self.n_cluster_vars)])).T.reshape(-1, self.n_cluster_vars)
            for i_smpl in range(cart.shape[0]):
                ind_train = np.full(self.n_obs, True)
                ind_test = np.full(self.n_obs, True)
                this_cluster_smpl_train = []
                this_cluster_smpl_test = []
                for i_var in range(self.n_cluster_vars):
                    i_fold = cart[i_smpl, i_var]
                    train_clusters = smpls_cluster_vars[i_var][i_fold][0]
                    test_clusters = smpls_cluster_vars[i_var][i_fold][1]
                    this_cluster_smpl_train.append(train_clusters)
                    this_cluster_smpl_test.append(test_clusters)
                    ind_train = ind_train & np.in1d(self.cluster_vars[:, i_var], train_clusters)
                    ind_test = ind_test & np.in1d(self.cluster_vars[:, i_var], test_clusters)
                train_set = np.arange(self.n_obs)[ind_train]
                test_set = np.arange(self.n_obs)[ind_test]
                smpls.append((train_set, test_set))
                smpls_cluster.append((this_cluster_smpl_train, this_cluster_smpl_test))
            all_smpls.append(smpls)
            all_smpls_cluster.append(smpls_cluster)

        return all_smpls, all_smpls_cluster
