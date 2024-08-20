import warnings
import numpy as np
import pandas as pd
from collections.abc import Callable

from scipy.stats import norm
from rdrobust import rdrobust, rdbwselect

from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.utils.resampling import DoubleMLResampling
from doubleml.utils._checks import _check_resampling_specification, _check_supports_sample_weights


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

    fuzzy: bool
        Indicates whether to fit a fuzzy or a sharp design.
        Default is ``True``.
    
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
                 fuzzy=True,
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
        self._fuzzy = fuzzy

        if not fuzzy and any(self._dml_data.d != self._intendend_treatment):
            warnings.warn('Treatment assignment does not match treatment intended. \n \
                           Did you mean `fuzzy = True`?')

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
        self._w = self._calc_weights(kernel=self._fs_kernel_function, h=self.h_fs)

        # TODO: Add further input checks
        self.kwargs = kwargs

        self._smpls = DoubleMLResampling(n_folds=self.n_folds, n_rep=self.n_rep, n_obs=obj_dml_data.n_obs,
                                         stratify=obj_dml_data.d).split_samples()

        self._initialize_reps(n_rep=self.n_rep)

    def __str__(self):
        if np.any(~np.isnan(self._M_Y[:, 0])):
            method_names = ["Conventional", "Robust"]
            lines = [
                "Method             Coef.     S.E.     t-stat       P>|t|           95% CI",
                "-------------------------------------------------------------------------",
            ]

            for i, name in enumerate(method_names):
                if i == 0:
                    line = (
                        f"{name:<18}"
                        f"{self.coef[i]:<10.3f}"
                        f"{self.se[i]:<10.3f}"
                        f"{self.t_stat[i]:<9.3f}"
                        f"{self.pval[i]:<11.3e}"
                        f"[{self.ci[i, 0]:.3f}, {self.ci[i, 1]:.3f}]"
                    )
                elif i == 1:
                    # Access robust values from index 2 as specified
                    line = (
                        f"{name:<17}"
                        "      -        -     "
                        f"{self.t_stat[2]:<9.3f}"
                        f"{self.pval[2]:<11.3e}"
                        f"[{self.ci[2, 0]:.3f}, {self.ci[2, 1]:.3f}]"
                    )

                lines.append(line)
            result = "\n".join(lines)
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
    def h(self):
        """
        Array of final bandwidths in the last stage estimation (shape (``n_rep``,)).
        """
        return self._h

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

        self._check_iterations(n_iterations)

        # set variables for readablitity
        Y = self._dml_data.y
        D = self._dml_data.d
        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            # set initial weights smpls
            weights = self.w

            for iteration in range(n_iterations):
                eta_Y = self._fit_nuisance_model(outcome=Y, estimator_name="ml_g",
                                                 weights=weights, smpls=self._smpls[i_rep])
                self._M_Y[:, i_rep] = Y - eta_Y

                if self.fuzzy:
                    eta_D = self._fit_nuisance_model(outcome=D, estimator_name="ml_m",
                                                     weights=weights, smpls=self._smpls[i_rep])
                    self._M_D[:, i_rep] = D - eta_D

                # update weights via iterative bandwidth fitting
                if iteration < (n_iterations - 1):
                    h, weights = self._update_weights()
                else:
                    if n_iterations == 1:
                        h = None

                    rdd_res = self._fit_rdd(h=h)
                    self._set_coefs(rdd_res, h)

        self.aggregate_over_splits()

        return self

    def confint(self, level=0.95):
        """
        Confidence intervals for RDFlex models.

        Parameters
        ----------
        level : float
            The confidence level.
            Default is ``0.95``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s).
        """
        if not isinstance(level, float):
            raise TypeError('The confidence level must be of float type. '
                            f'{str(level)} of type {str(type(level))} was passed.')
        if (level <= 0) | (level >= 1):
            raise ValueError('The confidence level must be in (0,1). '
                             f'{str(level)} was passed.')

        # compute critical values
        alpha = 1 - level
        percentages = np.array([alpha / 2, 1. - alpha / 2])

        critical_values = np.repeat(norm.ppf(percentages[1]), self._n_rep)

        # compute all cis over repetitions (shape: n_coef x 2 x n_rep)
        self._all_cis = np.stack(
            (self.all_coef - self.all_se * critical_values,
             self.all_coef + self.all_se * critical_values),
            axis=1)
        ci = np.median(self._all_cis, axis=2)
        df_ci = pd.DataFrame(ci, columns=['{:.1f} %'.format(i * 100) for i in percentages],
                             index=['Conventional', 'Bias-Corrected', 'Robust'])

        return df_ci

    def _fit_nuisance_model(self, outcome, estimator_name, weights, smpls):
        Z = self._intendend_treatment  # instrument for treatment
        X = self._dml_data.x
        ZX = np.column_stack((Z, X))

        mu_left, mu_right = np.full_like(outcome, fill_value=np.nan), np.full_like(outcome, fill_value=np.nan)

        for train_index, test_index in smpls:
            estimator = clone(self._learner[estimator_name])
            estimator.fit(ZX[train_index], outcome[train_index], sample_weight=weights[train_index])

            X_test_left = np.column_stack((np.zeros_like(Z[test_index]), X[test_index]))
            X_test_right = np.column_stack((np.ones_like(Z[test_index]), X[test_index]))

            if self._predict_method[estimator_name] == "predict":
                mu_left[test_index] = estimator.predict(X_test_left)
                mu_right[test_index] = estimator.predict(X_test_right)
            else:
                assert self._predict_method[estimator_name] == "predict_proba"
                mu_left[test_index] = estimator.predict_proba(X_test_left)[:, 1]
                mu_right[test_index] = estimator.predict_proba(X_test_right)[:, 1]

        return (mu_left + mu_right)/2

    def _update_weights(self):
        rdd_res = self._fit_rdd()
        # TODO: "h" features "left" and "right" - what do we do if it is non-symmetric?
        h = rdd_res.bws.loc["h"].max()
        weights = self._calc_weights(kernel=self._fs_kernel_function, h=h)

        return h, weights

    def _fit_rdd(self, h=None):
        if self.fuzzy:
            rdd_res = rdrobust(y=self._M_Y[:, self._i_rep], x=self._score,
                               fuzzy=self._M_D[:, self._i_rep], h=h, **self.kwargs)
        else:
            rdd_res = rdrobust(y=self._M_Y[:, self._i_rep], x=self._score,
                               h=h, **self.kwargs)
        return rdd_res

    def _set_coefs(self, rdd_res, h):
        self._h[self._i_rep] = h
        self._all_coef[:, self._i_rep] = rdd_res.coef.values.flatten()
        self._all_se[:, self._i_rep] = rdd_res.se.values.flatten()
        self._all_ci[:, :, self._i_rep] = rdd_res.ci.values
        self._rdd_obj[self._i_rep] = rdd_res

    def _calc_weights(self, kernel, h):
        weights = kernel(self._score, h)
        return weights

    def _initialize_reps(self, n_rep):
        self._M_Y = np.full(shape=(self._dml_data.n_obs, n_rep), fill_value=np.nan)
        self._M_D = np.full(shape=(self._dml_data.n_obs, n_rep), fill_value=np.nan)
        self._h = np.full(shape=n_rep, fill_value=np.nan)
        self._rdd_obj = [None] * n_rep
        self._all_coef = np.full(shape=(3, n_rep), fill_value=np.nan)
        self._all_se = np.full(shape=(3, n_rep), fill_value=np.nan)
        self._all_ci = np.full(shape=(3, 2, n_rep), fill_value=np.nan)
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
        _check_supports_sample_weights(ml_g, 'ml_g')
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
                _check_supports_sample_weights(ml_m, 'ml_m')

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

    def _check_iterations(self, n_iterations):
        """Validate the number of iterations."""
        if not isinstance(n_iterations, int):
            raise TypeError('The number of iterations for the iterative bandwidth fitting must be of int type. '
                            f'{str(n_iterations)} of type {str(type(n_iterations))} was passed.')
        if n_iterations < 1:
            raise ValueError('The number of iterations for the iterative bandwidth fitting has to be positive. '
                             f'{str(n_iterations)} was passed.')

    def aggregate_over_splits(self):
        self._coef = np.median(self.all_coef, axis=1)
        self._ci = np.median(self._all_ci, axis=2)
        med_se = np.median(self.all_se, axis=1)
        self._se = [np.sqrt(np.median(med_se[i]**2 + (self.all_coef[i, :] - self._coef[i])**2)) for i in range(3)]
