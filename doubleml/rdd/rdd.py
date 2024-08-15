import numpy as np
from scipy.stats import norm
import warnings
from collections.abc import Callable

from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target

from rdrobust import rdrobust, rdbwselect

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.utils.resampling import DoubleMLResampling
from doubleml.utils._checks import _check_resampling_specification


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

    h_fs : float or None
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

        self._check_data(obj_dml_data, cutoff)
        self._dml_data = obj_dml_data

        self._score = self._dml_data.s - cutoff
        self._cutoff = cutoff
        self._intendend_treatment = (self._score >= 0).astype(bool)
        self._fuzzy = any(self._dml_data.d != self._intendend_treatment)

        self._check_and_set_learner(ml_g, ml_m)

        _check_resampling_specification(n_folds, n_rep)
        self._n_folds = n_folds
        self._n_rep = n_rep

        if h_fs is None:
            fuzzy = self._dml_data.d if self._fuzzy else None
            self._h_fs = rdbwselect(y=obj_dml_data.y,
                                    x=self._score,
                                    fuzzy=fuzzy).bws.values.flatten().max()
        else:
            if not isinstance(h_fs, (float)):
                raise TypeError("Initial bandwidth 'h_fs' has to be a float. "
                                f'Object of type {str(type(h_fs))} passed.')
            self._h_fs = h_fs

        self._fs_kernel_function, self._fs_kernel_name = self._check_and_set_kernel(fs_kernel)
        self._w, self._w_mask = self._calc_weights(kernel=self._fs_kernel_function, h=self.h_fs)

        # TODO: Add further input checks
        self.kwargs = kwargs

        self._smpls = DoubleMLResampling(n_folds=self.n_folds, n_rep=self.n_rep, n_obs=self.w_mask.sum(),
                                         stratify=obj_dml_data.d[self.w_mask]).split_samples()

        self._initialize_reps(n_rep=self.n_rep)

    def __str__(self):
        # TODO: Adjust __str__ to other DoubleML classes (see doubleml.py)
        if self._M_Y[0] is not None:
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
                round(self.pval[0], 3),
                ci_conventional[0],
                ci_conventional[1]
            )

            robust_row = col_format.format(
                "Robust",
                "-",
                "-",
                round(self.pval[2], 3),
                ci_robust[0],
                ci_robust[1]
            )

            result = f"{header}\n{conventional_row}\n{robust_row}"

            return result
        else:
            return "DoubleML RDFlex Object. Run `.fit()` for estimation."

    @property
    def fuzzy(self):
        """
        Indicates whether the design is fuzzy or not.
        """
        return self._fuzzy

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
    def h_fs(self):
        """
        Initial bandwidth in the first stage estimation.
        """
        return self._h_fs

    @property
    def fs_kernel(self):
        """
        Kernel for the first stage estimation.
        """
        return self._fs_kernel_name

    @property
    def w(self):
        """
        Weights for the first stage estimation.
        """
        return self._w

    @property
    def w_mask(self):
        """
        Mask for the weights of the first stage estimation.
        """
        return self._w_mask

    @property
    def cutoff(self):
        """
        Cutoff at which the treatment effect is estimated.
        """
        return self._cutoff

    @property
    def coef(self):
        """
        Estimates for the causal parameter after calling :meth:`fit`.
        """
        return self._coef

    @property
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._se

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
    def ci(self):
        """
        Confidence intervals for the causal parameter(s) after calling :meth:`fit`.
        """
        return self._ci

    @property
    def all_coef(self):
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`.
        """
        return self._all_coef

    @property
    def all_se(self):
        """
        Standard errors of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth:`fit`.
        """
        return self._all_se

    def fit(self, n_iterations=2):
        """
        Estimate RDFlex model.

        Parameters
        ----------

        n_iterations : int
            Number of iterations for the iterative bandwidth fitting.
            Default is ``2``.

        Returns
        -------
        self : object
        """

        if not isinstance(n_iterations, int):
            raise TypeError('The number of iterations for the iterative bandwidth fitting must be of int type. '
                            f'{str(n_iterations)} of type {str(type(n_iterations))} was passed.')
        if n_iterations < 1:
            raise ValueError('The number of iterations for the iterative bandwidth fitting has to be positive. '
                             f'{str(n_iterations)} was passed.')

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            # reset weights, smpls and bandwidth
            h = None
            weights = self.w
            weights_mask = self.w_mask
            tmp_smpls = self._smpls[i_rep]

            for _ in range(n_iterations):
                y_masked = self._dml_data.y[weights_mask]
                eta_Y = self._fit_nuisance_model(outcome=y_masked, estimator_name="ml_g",
                                                 weights=weights, w_mask=weights_mask, smpls=tmp_smpls)
                self._M_Y[i_rep] = y_masked - eta_Y

                if self.fuzzy:
                    d_masked = self._dml_data.d[weights_mask]
                    eta_D = self._fit_nuisance_model(outcome=d_masked, estimator_name="ml_m",
                                                     weights=weights, w_mask=weights_mask, smpls=tmp_smpls)
                    self._M_D[i_rep] = d_masked - eta_D

                # update weights, smpls and bandwidth
                h = self._fit_rdd(h=h, w_mask=weights_mask)
                weights, weights_mask = self._calc_weights(kernel=self._fs_kernel_function, h=h)
                # using new masked d for stratification
                # TODO: Add seed to resampling
                tmp_smpls = DoubleMLResampling(n_folds=self.n_folds, n_rep=1, n_obs=weights_mask.sum(),
                                               stratify=self._dml_data.d[weights_mask]).split_samples()[0]
        self.aggregate_over_splits()

        return self

    def _fit_nuisance_model(self, outcome, estimator_name, weights, w_mask, smpls):
        Z = self._intendend_treatment[w_mask]  # instrument for treatment
        X = self._dml_data.x[w_mask]
        weights = weights[w_mask]
        ZX = np.column_stack((Z, X))

        pred_left, pred_right = np.zeros_like(outcome), np.zeros_like(outcome)

        for train_index, test_index in smpls:
            estimator = clone(self._learner[estimator_name])
            estimator.fit(ZX[train_index], outcome[train_index], sample_weight=weights[train_index])

            X_test_pos = np.column_stack((np.ones_like(Z[test_index]), X[test_index]))
            X_test_neg = np.column_stack((np.zeros_like(Z[test_index]), X[test_index]))

            if self._predict_method[estimator_name] == "predict":
                pred_left[test_index] = estimator.predict(X_test_neg)
                pred_right[test_index] = estimator.predict(X_test_pos)
            else:
                assert self._predict_method[estimator_name] == "predict_proba"
                pred_left[test_index] = estimator.predict_proba(X_test_neg)[:, 1]
                pred_right[test_index] = estimator.predict_proba(X_test_pos)[:, 1]

        return (pred_left + pred_right)/2

    def _fit_rdd(self, w_mask, h=None):
        _rdd_res = rdrobust(y=self._M_Y[self._i_rep], x=self._dml_data.s[w_mask],
                            fuzzy=self._M_D[self._i_rep], h=h, **self.kwargs)
        self._all_coef[:, self._i_rep] = _rdd_res.coef.values.flatten()
        self._all_se[:, self._i_rep] = _rdd_res.se.values.flatten()
        self._all_ci[:, :, self._i_rep] = _rdd_res.ci.values
        self._rdd_obj[self._i_rep] = _rdd_res
        # TODO: "h" features "left" and "right" - what do we do if it is non-symmetric?
        return _rdd_res.bws.loc["h"].max()

    def _calc_weights(self, kernel, h):
        weights = kernel(self._score, h)
        return weights, weights.astype(bool)

    def _initialize_reps(self, n_rep):
        self._M_Y = [None] * n_rep
        self._M_D = [None] * n_rep
        self._rdd_obj = [None] * n_rep
        self._all_coef = np.empty(shape=(3, n_rep))
        self._all_se = np.empty(shape=(3, n_rep))
        self._all_ci = np.empty(shape=(3, 2, n_rep))
        return

    def _check_data(self, obj_dml_data, cutoff):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')

        # score checks
        if obj_dml_data.s_col is None:
            raise ValueError('Incompatible data. ' +
                             'Score variable has not been set. ')
        is_continuous = (type_of_target(obj_dml_data.s) == 'continuous')
        if not is_continuous:
            raise ValueError('Incompatible data. ' +
                             'Score variable has to be continuous. ')

        if not isinstance(cutoff, (int, float)):
            raise TypeError('Cutoff value has to be a float or int. '
                            f'Object of type {str(type(cutoff))} passed.')
        if not (obj_dml_data.s.min() <= cutoff <= obj_dml_data.s.max()):
            raise ValueError('Cutoff value is not within the range of the score variable. ')

        # treatment checks
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an RDFlex model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')

        # instrument checks
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). ')
        return

    def _check_and_set_learner(self, ml_g, ml_m):
        # check ml_g
        ml_g_is_classifier = DoubleML._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        self._learner = {'ml_g': ml_g}
        if ml_g_is_classifier:
            if self._dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict'}

        # check ml_m
        if self._fuzzy:
            if ml_m is not None:
                _ = DoubleML._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)

                self._learner['ml_m'] = ml_m
                self._predict_method['ml_m'] = 'predict_proba'
            else:
                raise ValueError('Fuzzy design requires a classifier ml_m for treatment assignment.')

        else:
            if ml_m is not None:
                warnings.warn(('A learner ml_m has been provided for for a sharp design but will be ignored. '
                               'A learner ml_m is not required for estimation.'))
        return

    def _check_and_set_kernel(self, fs_kernel):
        if not isinstance(fs_kernel, (str, Callable)):
            raise TypeError('fs_kernel must be either a string or a callable. '
                            f'{str(fs_kernel)} of type {str(type(fs_kernel))} was passed.')

        kernel_functions = {
            "uniform": lambda x, h: np.array(np.abs(x) <= h, dtype=float),
            "triangular": lambda x, h: np.array(np.maximum(0, (h - np.abs(x)) / h), dtype=float),
            "epanechnikov": lambda x, h: np.array(np.where(np.abs(x) < h, .75 * (1 - np.square(x / h)), 0), dtype=float)
        }

        if isinstance(fs_kernel, str):
            fs_kernel = fs_kernel.casefold()
            if fs_kernel not in kernel_functions:
                raise ValueError(f"Invalid kernel '{fs_kernel}'. Valid kernels are {list(kernel_functions.keys())}.")

            kernel_function = kernel_functions[fs_kernel]
            kernel_name = fs_kernel

        elif callable(fs_kernel):
            kernel_function = fs_kernel
            kernel_name = 'custom_kernel'

        return kernel_function, kernel_name

    def aggregate_over_splits(self):
        self._coef = np.median(self.all_coef, axis=1)
        self._ci = np.median(self._all_ci, axis=2)
        med_se = np.median(self.all_se, axis=1)
        self._se = [np.sqrt(np.median(med_se[i]**2 + (self.all_coef[i, :] - self._coef[i])**2)) for i in range(3)]
