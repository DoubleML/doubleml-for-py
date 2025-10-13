import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.kde import KDEUnivariate

from ._checks import _check_is_partition


def _assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        raise ValueError("Only one- or two-dimensional arrays are allowed")
    return x


def _get_cond_smpls(smpls, bin_var):
    smpls_0 = [(np.intersect1d(np.where(bin_var == 0)[0], train), test) for train, test in smpls]
    smpls_1 = [(np.intersect1d(np.where(bin_var == 1)[0], train), test) for train, test in smpls]
    return smpls_0, smpls_1


def _get_cond_smpls_2d(smpls, bin_var1, bin_var2):
    subset_00 = (bin_var1 == 0) & (bin_var2 == 0)
    smpls_00 = [(np.intersect1d(np.where(subset_00)[0], train), test) for train, test in smpls]
    subset_01 = (bin_var1 == 0) & (bin_var2 == 1)
    smpls_01 = [(np.intersect1d(np.where(subset_01)[0], train), test) for train, test in smpls]
    subset_10 = (bin_var1 == 1) & (bin_var2 == 0)
    smpls_10 = [(np.intersect1d(np.where(subset_10)[0], train), test) for train, test in smpls]
    subset_11 = (bin_var1 == 1) & (bin_var2 == 1)
    smpls_11 = [(np.intersect1d(np.where(subset_11)[0], train), test) for train, test in smpls]
    return smpls_00, smpls_01, smpls_10, smpls_11


def _fit(estimator, x, y, train_index, idx=None):
    estimator.fit(x[train_index, :], y[train_index])
    return estimator, idx


def _dml_cv_predict(
    estimator, x, y, smpls=None, n_jobs=None, est_params=None, method="predict", return_train_preds=False, return_models=False
):
    n_obs = x.shape[0]

    smpls_is_partition = _check_is_partition(smpls, n_obs)
    fold_specific_params = (est_params is not None) & (not isinstance(est_params, dict))
    fold_specific_target = isinstance(y, list)
    manual_cv_predict = (
        (not smpls_is_partition) | return_train_preds | fold_specific_params | fold_specific_target | return_models
    )

    res = {"models": None}
    if not manual_cv_predict:
        if est_params is None:
            # if there are no parameters set we redirect to the standard method
            preds = cross_val_predict(clone(estimator), x, y, cv=smpls, n_jobs=n_jobs, method=method)
        else:
            assert isinstance(est_params, dict)
            # if no fold-specific parameters we redirect to the standard method
            # warnings.warn("Using the same (hyper-)parameters for all folds")
            preds = cross_val_predict(clone(estimator).set_params(**est_params), x, y, cv=smpls, n_jobs=n_jobs, method=method)
        if method == "predict_proba":
            res["preds"] = preds[:, 1]
        else:
            res["preds"] = preds
        res["targets"] = np.copy(y)
    else:
        if not smpls_is_partition:
            assert not fold_specific_target, "combination of fold-specific y and no cross-fitting not implemented yet"
            assert len(smpls) == 1

        if method == "predict_proba":
            assert not fold_specific_target  # fold_specific_target only needed for PLIV.partialXZ
            y = np.asarray(y)
            le = LabelEncoder()
            y = le.fit_transform(y)

        parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")

        if fold_specific_target:
            y_list = list()
            for idx, (train_index, _) in enumerate(smpls):
                xx = np.full(n_obs, np.nan)
                xx[train_index] = y[idx]
                y_list.append(xx)
        else:
            # just replicate the y in a list
            y_list = [y] * len(smpls)

        if est_params is None:
            fitted_models = parallel(
                delayed(_fit)(clone(estimator), x, y_list[idx], train_index, idx)
                for idx, (train_index, test_index) in enumerate(smpls)
            )
        elif isinstance(est_params, dict):
            # warnings.warn("Using the same (hyper-)parameters for all folds")
            fitted_models = parallel(
                delayed(_fit)(clone(estimator).set_params(**est_params), x, y_list[idx], train_index, idx)
                for idx, (train_index, test_index) in enumerate(smpls)
            )
        else:
            assert len(est_params) == len(smpls), "provide one parameter setting per fold"
            fitted_models = parallel(
                delayed(_fit)(clone(estimator).set_params(**est_params[idx]), x, y_list[idx], train_index, idx)
                for idx, (train_index, test_index) in enumerate(smpls)
            )

        preds = np.full(n_obs, np.nan)
        targets = np.full(n_obs, np.nan)
        train_preds = list()
        train_targets = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            assert idx == fitted_models[idx][1]
            pred_fun = getattr(fitted_models[idx][0], method)
            if method == "predict_proba":
                preds[test_index] = pred_fun(x[test_index, :])[:, 1]
            else:
                preds[test_index] = pred_fun(x[test_index, :])

            if fold_specific_target:
                # targets not available for fold specific target
                targets = None
            else:
                targets[test_index] = y[test_index]

            if return_train_preds:
                train_preds.append(pred_fun(x[train_index, :]))
                train_targets.append(y[train_index])

        res["preds"] = preds
        res["targets"] = targets
        if return_train_preds:
            res["train_preds"] = train_preds
            res["train_targets"] = train_targets
        if return_models:
            fold_ids = [xx[1] for xx in fitted_models]
            if not np.all(fold_ids == np.arange(len(smpls))):
                raise RuntimeError("export of fitted models failed")
            res["models"] = [xx[0] for xx in fitted_models]

    return res


class _OptunaSearchResult:
    """Lightweight container mimicking selected GridSearchCV attributes."""

    def __init__(self, estimator, best_params, best_score, study, trials_dataframe):
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.study_ = study
        self.trials_dataframe_ = trials_dataframe

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("The wrapped estimator does not support predict_proba().")
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


def _dml_tune(
    y,
    x,
    train_inds,
    learner,
    param_grid,
    scoring_method,
    n_folds_tune,
    n_jobs_cv,
    search_mode,
    n_iter_randomized_search,
    optuna_settings,
):
    if search_mode == "optuna":
        return _dml_tune_optuna(
            y,
            x,
            train_inds,
            learner,
            param_grid,
            scoring_method,
            n_folds_tune,
            n_jobs_cv,
            optuna_settings,
        )

    tune_res = list()
    for train_index in train_inds:
        tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        if search_mode == "grid_search":
            g_grid_search = GridSearchCV(learner, param_grid, scoring=scoring_method, cv=tune_resampling, n_jobs=n_jobs_cv)
        else:
            assert search_mode == "randomized_search"
            g_grid_search = RandomizedSearchCV(
                learner,
                param_grid,
                scoring=scoring_method,
                cv=tune_resampling,
                n_jobs=n_jobs_cv,
                n_iter=n_iter_randomized_search,
            )
        tune_res.append(g_grid_search.fit(x[train_index, :], y[train_index]))

    return tune_res


def _resolve_optuna_settings(optuna_settings):
    default_settings = {
        "n_trials": 100,
        "timeout": None,
        "direction": "maximize",
        "study_kwargs": {},
        "optimize_kwargs": {},
        "sampler": None,
        "pruner": None,
        "callbacks": None,
        "catch": (),
        "show_progress_bar": False,
        "gc_after_trial": False,
        "search_space": None,
        "study_factory": None,
        "study": None,
        "n_jobs_optuna": None,  # Parallel trial execution
        "verbosity": None,  # Optuna logging verbosity level
    }

    if optuna_settings is None:
        return default_settings

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    resolved = default_settings.copy()
    resolved.update(optuna_settings)
    if not isinstance(resolved["study_kwargs"], dict):
        raise TypeError("optuna_settings['study_kwargs'] must be a dict.")
    if not isinstance(resolved["optimize_kwargs"], dict):
        raise TypeError("optuna_settings['optimize_kwargs'] must be a dict.")
    if resolved["callbacks"] is not None and not isinstance(resolved["callbacks"], (list, tuple)):
        raise TypeError("optuna_settings['callbacks'] must be a sequence of callables or None.")
    if resolved["study"] is not None and resolved["study_factory"] is not None:
        raise ValueError("Provide only one of 'study' or 'study_factory' in optuna_settings.")
    if resolved["search_space"] is not None and not isinstance(resolved["search_space"], dict):
        if not callable(resolved["search_space"]):
            raise TypeError("optuna_settings['search_space'] must be callable, a dict, or None.")
    return resolved


def _select_optuna_settings(optuna_settings, learner_name):
    if optuna_settings is None:
        return _resolve_optuna_settings(None)

    if not isinstance(optuna_settings, dict):
        raise TypeError("optuna_settings must be a dict or None.")

    base_keys = {
        "n_trials",
        "timeout",
        "direction",
        "study_kwargs",
        "optimize_kwargs",
        "sampler",
        "pruner",
        "callbacks",
        "catch",
        "show_progress_bar",
        "gc_after_trial",
        "search_space",
        "study_factory",
        "study",
        "n_jobs_optuna",
        "verbosity",
    }

    base_settings = {key: value for key, value in optuna_settings.items() if key in base_keys}

    learner_specific = optuna_settings.get(learner_name)
    if learner_specific is None:
        return _resolve_optuna_settings(base_settings)

    if not isinstance(learner_specific, dict):
        raise TypeError(f"optuna_settings for learner '{learner_name}' must be a dict or None.")

    merged = base_settings.copy()
    merged.update(learner_specific)
    return _resolve_optuna_settings(merged)


def _suggest_from_grid(trial, param_name, param_spec, search_space_config, optuna_module):
    """
    Suggest a parameter value from a grid specification.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object.
    param_name : str
        The name of the parameter.
    param_spec : various
        The parameter specification (list, dict, distribution, etc.).
    search_space_config : callable, dict, or None
        Optional search space configuration override.
    optuna_module : module
        The optuna module.

    Returns
    -------
    value
        The suggested parameter value.
    """
    # Handle search_space overrides first
    if search_space_config is not None:
        if callable(search_space_config):
            return search_space_config(trial, param_name, param_spec)
        if isinstance(search_space_config, dict) and param_name in search_space_config:
            override = search_space_config[param_name]
            if callable(override):
                return override(trial, param_spec)
            if hasattr(optuna_module, "distributions") and isinstance(override, optuna_module.distributions.BaseDistribution):
                return trial._suggest(param_name, override)
            if isinstance(override, (list, tuple)):
                return _suggest_from_grid(trial, param_name, override, None, optuna_module)
            raise TypeError(f"Unsupported search_space override type for parameter '{param_name}'. "
                           f"Expected callable, Optuna distribution, or list/tuple, got {type(override)}.")

    # Handle Optuna distributions directly
    if hasattr(optuna_module, "distributions") and isinstance(param_spec, optuna_module.distributions.BaseDistribution):
        return trial._suggest(param_name, param_spec)

    # Handle dict with 'suggest' callable
    if isinstance(param_spec, dict) and "suggest" in param_spec:
        suggest_func = param_spec["suggest"]
        if not callable(suggest_func):
            raise TypeError(f"The 'suggest' entry for parameter '{param_name}' must be callable, got {type(suggest_func)}.")
        return suggest_func(trial)

    # Handle list/tuple specifications
    if isinstance(param_spec, (list, tuple)):
        if len(param_spec) == 0:
            raise ValueError(f"Parameter grid for '{param_name}' is empty.")

        # Check for numeric range: [low, high] or [low, high, step]
        if len(param_spec) in (2, 3) and all(isinstance(v, (int, float)) for v in param_spec):
            low, high = param_spec[0], param_spec[1]
            step = param_spec[2] if len(param_spec) == 3 else None

            if low >= high:
                raise ValueError(f"Parameter '{param_name}': low ({low}) must be less than high ({high}).")

            if step is not None and step <= 0:
                raise ValueError(f"Step must be positive for parameter '{param_name}', got {step}.")

            # Use int if all values are integers, otherwise float
            if all(isinstance(v, int) for v in param_spec):
                if step is not None:
                    return trial.suggest_int(param_name, int(low), int(high), step=int(step))
                return trial.suggest_int(param_name, int(low), int(high))
            else:
                if step is not None:
                    return trial.suggest_float(param_name, float(low), float(high), step=float(step))
                return trial.suggest_float(param_name, float(low), float(high))

        # Categorical choice
        return trial.suggest_categorical(param_name, list(param_spec))

    raise TypeError(
        f"Unsupported parameter specification for '{param_name}' in optuna tuning. "
        f"Provide a list/tuple, optuna distribution, or a dict with a 'suggest' callable. "
        f"Got {type(param_spec)}."
    )


def _dml_tune_optuna(y, x, train_inds, learner, param_grid, scoring_method, n_folds_tune, n_jobs_cv, optuna_settings):
    try:
        import optuna  # pylint: disable=import-error
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is not installed. Please install Optuna (e.g., pip install optuna) to use search_mode='optuna'."
        ) from exc

    if isinstance(param_grid, list):
        raise ValueError("Param grids provided as a list of dicts are not supported for optuna tuning.")
    if not isinstance(param_grid, dict):
        raise TypeError("Param grid for optuna tuning must be a dict.")

    # Validate param_grid before starting optimization
    if not param_grid:
        raise ValueError("param_grid cannot be empty for optuna tuning.")

    tune_res = list()

    for train_index in train_inds:
        learner_key = learner.__class__.__name__ if hasattr(learner, "__class__") else ""
        settings = _select_optuna_settings(optuna_settings, learner_key)

        # Set Optuna logging verbosity if specified
        if settings.get("verbosity") is not None:
            optuna.logging.set_verbosity(settings["verbosity"])

        X_train = x[train_index, :]
        y_train = y[train_index]

        # Pre-create KFold object outside objective to ensure consistent splitting with fixed random state
        cv = KFold(n_splits=n_folds_tune, shuffle=True, random_state=42)

        def objective(trial):
            params = {}
            for param_name, param_spec in param_grid.items():
                params[param_name] = _suggest_from_grid(
                    trial,
                    param_name,
                    param_spec,
                    settings.get("search_space"),
                    optuna,
                )

            estimator = clone(learner).set_params(**params)
            scores = cross_validate(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring_method,
                n_jobs=n_jobs_cv,
                return_train_score=False,
                error_score="raise",
            )
            test_scores = scores["test_score"]
            return np.nanmean(test_scores)

        study_kwargs = settings.get("study_kwargs", {}).copy()
        direction = settings.get("direction", "maximize")
        if "direction" not in study_kwargs:
            study_kwargs["direction"] = direction

        sampler = settings.get("sampler")
        if sampler is not None:
            study_kwargs["sampler"] = sampler
        pruner = settings.get("pruner")
        if pruner is not None:
            study_kwargs["pruner"] = pruner

        optimize_kwargs = {
            "n_trials": settings.get("n_trials"),
            "timeout": settings.get("timeout"),
            "callbacks": settings.get("callbacks"),
            "catch": settings.get("catch"),
            "show_progress_bar": settings.get("show_progress_bar", False),
            "gc_after_trial": settings.get("gc_after_trial", False),
        }

        # Add n_jobs support for parallel trial execution if available in Optuna version
        n_jobs_optuna = settings.get("n_jobs_optuna")
        if n_jobs_optuna is not None:
            optimize_kwargs["n_jobs"] = n_jobs_optuna

        optimize_kwargs.update(settings.get("optimize_kwargs", {}))
        optimize_kwargs = {
            key: value
            for key, value in optimize_kwargs.items()
            if value is not None or key in ["show_progress_bar", "gc_after_trial"]
        }

        study_instance = settings.get("study")
        if study_instance is not None:
            study = study_instance
        else:
            factory = settings.get("study_factory")
            if callable(factory):
                try:
                    maybe_study = factory(study_kwargs)
                except TypeError:
                    maybe_study = factory()
                if maybe_study is None:
                    study = optuna.create_study(**study_kwargs)
                elif isinstance(maybe_study, optuna.study.Study):
                    study = maybe_study
                else:
                    raise TypeError("study_factory must return an optuna.study.Study or None.")
            else:
                study = optuna.create_study(**study_kwargs)

        study.optimize(objective, **optimize_kwargs)

        # Check if optimization found any successful trials
        if study.best_trial is None:
            raise RuntimeError(
                f"Optuna optimization failed to find any successful trials. "
                f"Total trials: {len(study.trials)}, "
                f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
            )

        best_params = study.best_trial.params
        best_estimator = clone(learner).set_params(**best_params)
        best_estimator.fit(X_train, y_train)

        tune_res.append(
            _OptunaSearchResult(
                estimator=best_estimator,
                best_params=best_params,
                best_score=study.best_value,
                study=study,
                trials_dataframe=study.trials_dataframe(attrs=("number", "value", "params", "state")),
            )
        )

    return tune_res


def _draw_weights(method, n_rep_boot, n_obs):
    if method == "Bayes":
        weights = np.random.exponential(scale=1.0, size=(n_rep_boot, n_obs)) - 1.0
    elif method == "normal":
        weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
    elif method == "wild":
        xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
    else:
        raise ValueError("invalid boot method")

    return weights


def _rmse(y_true, y_pred):
    subset = np.logical_not(np.isnan(y_true))
    rmse = root_mean_squared_error(y_true[subset], y_pred[subset])
    return rmse


def _logloss(y_true, y_pred):
    subset = np.logical_not(np.isnan(y_true))
    logloss = log_loss(y_true[subset], y_pred[subset])
    return logloss


def _predict_zero_one_propensity(learner, X):
    pred_proba = learner.predict_proba(X)
    if pred_proba.shape[1] == 2:
        res = pred_proba[:, 1]
    else:
        warnings.warn("Subsample has not common support. Results are based on adjusted propensities.")
        res = learner.predict(X)
    return res


def _get_bracket_guess(score, coef_start, coef_bounds):
    max_bracket_length = coef_bounds[1] - coef_bounds[0]
    b_guess = coef_bounds
    delta = 0.1
    s_different = False
    while (not s_different) & (delta <= 1.0):
        a = np.maximum(coef_start - delta * max_bracket_length / 2, coef_bounds[0])
        b = np.minimum(coef_start + delta * max_bracket_length / 2, coef_bounds[1])
        b_guess = (a, b)
        f_a = score(b_guess[0])
        f_b = score(b_guess[1])
        s_different = np.sign(f_a) != np.sign(f_b)
        delta += 0.1
    return s_different, b_guess


def _default_kde(u, weights):
    dens = KDEUnivariate(u)
    dens.fit(kernel="gau", bw="silverman", weights=weights, fft=False)

    return dens.evaluate(0)


def _solve_ipw_score(ipw_score, bracket_guess):
    def abs_ipw_score(theta):
        return abs(ipw_score(theta))

    res = minimize_scalar(abs_ipw_score, bracket=bracket_guess, method="brent")
    ipw_est = res.x
    return ipw_est


def _aggregate_coefs_and_ses(all_coefs, all_ses):
    # already expects equally scaled variances over all repetitions
    # aggregation is done over dimension 1, such that the coefs and ses have to be of shape (n_coefs, n_rep)
    coefs = np.median(all_coefs, 1)

    # construct the upper bounds & aggregate
    critical_value = 1.96
    all_upper_bounds = all_coefs + critical_value * all_ses
    agg_upper_bounds = np.median(all_upper_bounds, axis=1)
    # reverse to calculate the standard errors
    ses = (agg_upper_bounds - coefs) / critical_value
    return coefs, ses


def _var_est(psi, psi_deriv, smpls, is_cluster_data, cluster_vars=None, smpls_cluster=None, n_folds_per_cluster=None):
    if not is_cluster_data:
        # psi and psi_deriv should be of shape (n_obs, ...)
        var_scaling_factor = psi.shape[0]

        J = np.mean(psi_deriv)
        gamma_hat = np.mean(np.square(psi))

    else:
        assert cluster_vars is not None
        assert smpls_cluster is not None
        assert n_folds_per_cluster is not None
        n_folds = len(smpls)

        # one cluster
        if cluster_vars.shape[1] == 1:
            first_cluster_var = cluster_vars[:, 0]
            clusters = np.unique(first_cluster_var)
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(n_folds):
                test_inds = smpls[i_fold][1]
                test_cluster_inds = smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                const = 1 / len(I_k)
                for cluster_value in I_k:
                    ind_cluster = first_cluster_var == cluster_value
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_deriv[test_inds]) / len(I_k)

            var_scaling_factor = len(clusters)
            J = np.divide(j_hat, n_folds_per_cluster)
            gamma_hat = np.divide(gamma_hat, n_folds_per_cluster)

        else:
            assert cluster_vars.shape[1] == 2
            first_cluster_var = cluster_vars[:, 0]
            second_cluster_var = cluster_vars[:, 1]
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(n_folds):
                test_inds = smpls[i_fold][1]
                test_cluster_inds = smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                J_l = test_cluster_inds[1]
                const = np.divide(min(len(I_k), len(J_l)), (np.square(len(I_k) * len(J_l))))
                for cluster_value in I_k:
                    ind_cluster = (first_cluster_var == cluster_value) & np.isin(second_cluster_var, J_l)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                for cluster_value in J_l:
                    ind_cluster = (second_cluster_var == cluster_value) & np.isin(first_cluster_var, I_k)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_deriv[test_inds]) / (len(I_k) * len(J_l))

            n_first_clusters = len(np.unique(first_cluster_var))
            n_second_clusters = len(np.unique(second_cluster_var))
            var_scaling_factor = min(n_first_clusters, n_second_clusters)
            J = np.divide(j_hat, np.square(n_folds_per_cluster))
            gamma_hat = np.divide(gamma_hat, np.square(n_folds_per_cluster))

    scaling = np.divide(1.0, np.multiply(var_scaling_factor, np.square(J)))
    sigma2_hat = np.multiply(scaling, gamma_hat)

    return sigma2_hat, var_scaling_factor


def _cond_targets(target, cond_sample):
    cond_target = target.astype(float)
    cond_target[np.invert(cond_sample)] = np.nan
    return cond_target


def _set_external_predictions(external_predictions, learners, treatment, i_rep):
    ext_prediction_dict = {}
    for learner in learners:
        if external_predictions is None:
            ext_prediction_dict[learner] = None
        elif learner in external_predictions[treatment].keys():
            if isinstance(external_predictions[treatment][learner], np.ndarray):
                ext_prediction_dict[learner] = external_predictions[treatment][learner][:, i_rep]
            else:
                ext_prediction_dict[learner] = None
        else:
            ext_prediction_dict[learner] = None
    return ext_prediction_dict


def _solve_quadratic_inequality(a: float, b: float, c: float):
    """
    Solves the quadratic inequation a*x^2 + b*x + c <= 0 and returns the intervals.

    Parameters
    ----------
    a : float
        Coefficient of x^2.
    b : float
        Coefficient of x.
    c : float
        Constant term.

    Returns
    -------
    List[Tuple[float, float]]
        A list of intervals where the inequation holds.
    """

    # Handle special cases
    if abs(a) < 1e-12:  # a is effectively zero
        if abs(b) < 1e-12:  # constant case
            return [(-np.inf, np.inf)] if c <= 0 else []
        # Linear case:
        else:
            root = -c / b
            return [(-np.inf, root)] if b > 0 else [(root, np.inf)]

    # Standard case: quadratic equation
    roots = np.polynomial.polynomial.polyroots([c, b, a])
    real_roots = np.sort(roots[np.isreal(roots)].real)

    if len(real_roots) == 0:  # No real roots
        if a > 0:  # parabola opens upwards, no real roots
            return []
        else:  # parabola opens downwards, always <= 0
            return [(-np.inf, np.inf)]
    elif len(real_roots) == 1 or np.allclose(real_roots[0], real_roots[1]):  # One real root
        if a > 0:
            return [(real_roots[0], real_roots[0])]  # parabola touches x-axis at one point
        else:
            return [(-np.inf, np.inf)]  # parabola is always <= 0
    else:
        assert len(real_roots) == 2
        if a > 0:  # happy quadratic (parabola opens upwards)
            return [(real_roots[0], real_roots[1])]
        else:  # sad quadratic (parabola opens downwards)
            return [(-np.inf, real_roots[0]), (real_roots[1], np.inf)]
