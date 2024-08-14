from rdrobust import rdrobust, rdbwselect
import numpy as np
import scipy.stats as stats
from doubleml.utils.resampling import DoubleMLResampling
from sklearn.base import clone


class RDFlex():
    """Flexible adjustment with double machine learning for regression discontinuity designs

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()`` or None
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        Or None, in case of a non-fuzzy design.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    cutoff : float or int
        A float or intspecifying the cutoff in the score.
        Default is ``0``.

    h_fs : float or int or None
        Initial bandwidth in the first stage estimation. If ``None``, then the optimal bandwidth without
        covariates will be used.
        Default is ``None``.

    fs_kernel : str
        Kernel for the first stage estimation. ``uniform``, ``triangular`` and ``epanechnikov``are supported.
        Default is ``uniform``.

    **kwargs : kwargs
        Key-worded arguments that are not used within RDFlex but directly handed to rdrobust.

    Examples
    --------

    Notes
    -----

    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m=None,
                 cutoff=0,
                 n_folds=5,
                 n_rep=1,
                 h_fs=None,
                 fs_kernel="uniform",
                 **kwargs):

        if obj_dml_data.d is not None and ml_m is None:
            raise ValueError("If D is specified (Fuzzy Design), a classifier 'ml_m' must be provided.")

        # TODO: Add further input checks

        self._dml_data = obj_dml_data

        self._dml_data._s -= cutoff
        self.T = (0.5*(np.sign(obj_dml_data.s)+1)).astype(bool)

        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.h_fs = h_fs
        self.kwargs = kwargs
        self.fs_kernel = fs_kernel

        if h_fs is None:
            self.h_fs = rdbwselect(y=obj_dml_data.y,
                                   x=obj_dml_data.s,
                                   fuzzy=obj_dml_data.d).bws.values.flatten().max()
        else:
            self.h_fs = h_fs

        self.w, self.w_mask = self._calc_weights(fs_kernel=fs_kernel, bw=self.h_fs)

        self.smpls = DoubleMLResampling(n_folds=n_folds, n_rep=n_rep, n_obs=self.w_mask.sum(),
                                        stratify=obj_dml_data.d[self.w_mask]).split_samples()

        self._initialize_reps(n_obs=self.w_mask.sum(), n_rep=n_rep)

    def __str__(self):
        # TODO: Adjust __str__ to other DoubleML classes (see doubleml.py)
        ci_conventional = [round(ci, 3) for ci in self.ci[0, :]]
        ci_robust = [round(ci, 3) for ci in self.ci[2, :]]
        col_format = "{:<20} {:>8} {:>8} {:>8} {:>8} to {:<8}"

        header = (
            "Method                  Coef.     S.E.    P>|t|            95% CI\n"
            "-----------------------------------------------------------------"
        )

        conventional_row = col_format.format(
            "Conventional",
            round(self.coef[0], 3),
            round(self.se[0], 3),
            round(self.p_value[0], 3),
            ci_conventional[0],
            ci_conventional[1]
        )

        robust_row = col_format.format(
            "Robust",
            "-",
            "-",
            round(self.p_value[2], 3),
            ci_robust[0],
            ci_robust[1]
        )

        result = f"{header}\n{conventional_row}\n{robust_row}"

        return result

    def fit(self, iterative=True, n_jobs_cv=-1, external_predictions=None):
        """
        Estimate RDFlex model.

        Parameters
        ----------
        iterative : bool
            Indicates whether the first stage bandwidth should be fitted iteratively.
            Defaule is ``True``

        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        external_predictions : None or dict
            If `None` all models for the learners are fitted and evaluated. If a dictionary containing predictions
            for a specific learner is supplied, the model will use the supplied nuisance predictions instead. Has to
            be a nested dictionary where the keys refer to the treatment and the keys of the nested dictionarys refer to the
            corresponding learners.
            Default is `None`.

        Returns
        -------
        self : object
        """

        # TODO: Implement external predictions
        if external_predictions is not None:
            raise NotImplementedError("Currently argument only included for compatibility.")

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            eta_Y, eta_D = self._fit_nuisance_models(n_jobs_cv, weights=self.w, w_mask=self.w_mask)
            self.M_Y[i_rep] = self._dml_data.y[self.w_mask] - eta_Y
            if self._dml_data.d is not None:
                self.M_D[i_rep] = self._dml_data.d[self.w_mask] - eta_D
            initial_h = self._fit_rdd(h=None, w_mask=self.w_mask)

            if iterative:
                adj_w, adj_w_mask = self._calc_weights(fs_kernel=self.fs_kernel, bw=initial_h)
                # created new smpls for smaller mask
                self.smpls[i_rep] = DoubleMLResampling(n_folds=self.n_folds, n_rep=1, n_obs=adj_w_mask.sum(),
                                                       stratify=self._dml_data.d[adj_w_mask]).split_samples()[0]
                eta_Y, eta_D = self._fit_nuisance_models(n_jobs_cv, weights=adj_w, w_mask=adj_w_mask)
                self.M_Y[i_rep] = self._dml_data.y[adj_w_mask] - eta_Y
                if self._dml_data.d is not None:
                    self.M_D[i_rep] = self._dml_data.d[adj_w_mask] - eta_D
                self._fit_rdd(h=initial_h, w_mask=adj_w_mask)

        self.aggregate_over_splits()

        return self

    def _fit_nuisance_models(self, n_jobs_cv, weights, w_mask):
        T = self.T[w_mask]
        TX = np.c_[T, self._dml_data.x[w_mask]]
        Y = self._dml_data.y[w_mask]
        X = self._dml_data.x[w_mask]
        D = self._dml_data.d[w_mask]
        weights = weights[w_mask]

        # TODO: Hard coded rule ok? Min 2 * n_folds fuzzy required per side.
        _check_left, _check_right = self._check_fuzzyness(w=w_mask, min_smpls=2*self.n_folds)

        pred_y, pred_d = np.zeros(Y.shape), np.zeros(D.shape)

        # TODO: Add parallelization for loop (n_jobs_cv)
        for train_index, test_index in self.smpls[self._i_rep]:
            ml_g, ml_m = clone(self.ml_g), clone(self.ml_m)
            ml_g.fit(TX[train_index], Y[train_index], sample_weight=weights[train_index])

            X_test_pos = np.c_[np.ones_like(T[test_index]), X[test_index]]
            X_test_neg = np.c_[np.zeros_like(T[test_index]), X[test_index]]

            pred_y[test_index] += ml_g.predict(X_test_pos)
            pred_y[test_index] += ml_g.predict(X_test_neg)

            if (_check_left | _check_right):
                ml_m.fit(TX[train_index], D[train_index], sample_weight=weights[train_index])

            if _check_left:
                pred_d[test_index] += ml_m.predict_proba(X_test_neg)[:, 1]
            if _check_right:
                pred_d[test_index] += ml_m.predict_proba(X_test_pos)[:, 1]

        if ~(_check_left):
            pred_d += np.average(D[~T], weights=weights[~T])
        if ~(_check_right):
            pred_d += np.average(D[T], weights=weights[T])

        return pred_y/2, pred_d/2

    def _fit_rdd(self, w_mask, h=None):
        _rdd_res = rdrobust(y=self.M_Y[self._i_rep], x=self._dml_data.s[w_mask],
                            fuzzy=self.M_D[self._i_rep], h=h, **self.kwargs)
        self.coefs[:, self._i_rep] = _rdd_res.coef.values.flatten()
        self.ses[:, self._i_rep] = _rdd_res.se.values.flatten()
        self.cis[:, :, self._i_rep] = _rdd_res.ci.values
        self.rdd_res.append(_rdd_res)
        # TODO: "h" features "left" and "right" - what do we do if it is non-symmetric?
        return _rdd_res.bws.loc["h"].max()

    def _calc_weights(self, bw, fs_kernel="uniform"):
        if fs_kernel == "uniform":
            weights = (np.abs(self._dml_data.s) < bw)
        if fs_kernel == "triangular":
            weights = np.maximum(0, (bw - np.abs(self._dml_data.s)) / bw)
        if fs_kernel == "epanechnikov":
            weights = np.where(np.abs(self._dml_data.s) < bw, .75*(1-self._dml_data.s/bw)**2, 0)
        return weights, weights.astype(bool)

    def _check_fuzzyness(self, w, min_smpls):
        return ((self._dml_data.d[w][self._dml_data.s[w] < 0].sum() > min_smpls),
                ((self._dml_data.d[w][self._dml_data.s[w] > 0] - 1).sum() < -(min_smpls)))

    def _initialize_reps(self, n_obs, n_rep):
        self.M_Y = [None] * n_rep
        self.M_D = [None] * n_rep
        self.rdd_res = []
        self.coefs = np.empty(shape=(3, n_rep))
        self.ses = np.empty(shape=(3, n_rep))
        self.cis = np.empty(shape=(3, 2, n_rep))
        return

    def aggregate_over_splits(self):
        self.coef = np.median(self.coefs, axis=1)
        self.ci = np.median(self.cis, axis=2)
        med_se = np.median(self.ses, axis=1)
        self.se = [np.sqrt(np.median(med_se[i]**2 + (self.coefs[i, :] - self.coef[i])**2)) for i in range(3)]

        z = self.coef / self.se
        self.p_value = 2*stats.norm.cdf(-np.abs(z))
