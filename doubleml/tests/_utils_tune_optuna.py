import numpy as np
import optuna


def _basic_optuna_settings(additional=None):
    base_settings = {
        "n_trials": 10,
        "sampler": optuna.samplers.TPESampler(seed=3141),
        "verbosity": optuna.logging.WARNING,
        "show_progress_bar": False,
    }
    if additional is not None:
        base_settings.update(additional)
    return base_settings


_SAMPLER_CASES = [
    ("random", optuna.samplers.RandomSampler(seed=3141)),
    ("tpe", optuna.samplers.TPESampler(seed=3141)),
]


def _small_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def _assert_tree_params(param_dict, depth_range=(1, 20), leaf_range=(1, 10), split_range=(2, 20)):
    assert set(param_dict.keys()) == {"max_depth", "min_samples_leaf", "min_samples_split"}
    assert depth_range[0] <= param_dict["max_depth"] <= depth_range[1]
    assert leaf_range[0] <= param_dict["min_samples_leaf"] <= leaf_range[1]
    assert split_range[0] <= param_dict["min_samples_split"] <= split_range[1]


def _build_param_space(dml_obj, param_fn):
    """Build parameter grid using the actual params_names from the DML object."""
    param_grid = {learner_name: param_fn for learner_name in dml_obj.params_names}
    return param_grid


def _select_binary_periods(panel_data):
    t_values = np.sort(panel_data.t_values)
    finite_g = sorted(val for val in panel_data.g_values if np.isfinite(val))
    for candidate in finite_g:
        pre_candidates = [t for t in t_values if t < candidate]
        if pre_candidates:
            return candidate, pre_candidates[-1], candidate
    raise RuntimeError("No valid treatment group found for binary DID data.")
