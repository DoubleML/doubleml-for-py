import numpy as np
import warnings

from sklearn.utils.multiclass import type_of_target


def _check_in_zero_one(value, name, include_zero=True, include_one=True):
    if not isinstance(value, float):
        raise TypeError(f'{name} must be of float type. '
                        f'{str(value)} of type {str(type(value))} was passed.')
    if include_zero & include_one:
        if (value < 0) | (value > 1):
            raise ValueError(f'{name} must be in [0,1]. '
                             f'{str(value)} was passed.')
    elif (not include_zero) & include_one:
        if (value <= 0) | (value > 1):
            raise ValueError(f'{name} must be in (0,1]. '
                             f'{str(value)} was passed.')
    elif include_zero & (not include_one):
        if (value < 0) | (value >= 1):
            raise ValueError(f'{name} must be in [0,1). '
                             f'{str(value)} was passed.')
    else:
        if (value <= 0) | (value >= 1):
            raise ValueError(f'{name} must be in (0,1). '
                             f'{str(value)} was passed.')
    return


def _check_integer(value, name, lower_bound=None, upper_bound=None):
    if not isinstance(value, int):
        raise TypeError(f'{name} must be an integer.'
                        f' {str(value)} of type {str(type(value))} was passed.')
    if lower_bound is not None:
        if value < lower_bound:
            raise ValueError(f'{name} must be larger or equal to {lower_bound}. '
                             f'{str(value)} was passed.')
    if upper_bound is not None:
        if value > upper_bound:
            raise ValueError(f'{name} must be smaller or equal to {upper_bound}. '
                             f'{str(value)} was passed.')
    return


def _check_float(value, name, lower_bound=None, upper_bound=None):
    if not isinstance(value, float):
        raise TypeError(f'{name} must be of float type.'
                        f' {str(value)} of type {str(type(value))} was passed.')
    if lower_bound is not None:
        if value < lower_bound:
            raise ValueError(f'{name} must be larger or equal to {lower_bound}. '
                             f'{str(value)} was passed.')
    if upper_bound is not None:
        if value > upper_bound:
            raise ValueError(f'{name} must be smaller or equal to {upper_bound}. '
                             f'{str(value)} was passed.')


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f'{name} has to be boolean.'
                        f' {str(value)} of type {str(type(value))} was passed.')


def _check_is_partition(smpls, n_obs):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if len(test_indices) != n_obs:
        return False
    hit = np.zeros(n_obs, dtype=bool)
    hit[test_indices] = True
    if not np.all(hit):
        return False
    return True


def _check_all_smpls(all_smpls, n_obs, check_intersect=False):
    all_smpls_checked = list()
    for smpl in all_smpls:
        all_smpls_checked.append(_check_smpl_split(smpl, n_obs, check_intersect))
    return all_smpls_checked


def _check_smpl_split(smpl, n_obs, check_intersect=False):
    smpl_checked = list()
    for tpl in smpl:
        smpl_checked.append(_check_smpl_split_tpl(tpl, n_obs, check_intersect))
    return smpl_checked


def _check_smpl_split_tpl(tpl, n_obs, check_intersect=False):
    train_index = np.sort(np.array(tpl[0]))
    test_index = np.sort(np.array(tpl[1]))

    if not issubclass(train_index.dtype.type, np.integer):
        raise TypeError('Invalid sample split. Train indices must be of type integer.')
    if not issubclass(test_index.dtype.type, np.integer):
        raise TypeError('Invalid sample split. Test indices must be of type integer.')

    if check_intersect:
        if set(train_index) & set(test_index):
            raise ValueError('Invalid sample split. Intersection of train and test indices is not empty.')

    if len(np.unique(train_index)) != len(train_index):
        raise ValueError('Invalid sample split. Train indices contain non-unique entries.')
    if len(np.unique(test_index)) != len(test_index):
        raise ValueError('Invalid sample split. Test indices contain non-unique entries.')

    # we sort the indices above
    # if not np.all(np.diff(train_index) > 0):
    #     raise NotImplementedError('Invalid sample split. Only sorted train indices are supported.')
    # if not np.all(np.diff(test_index) > 0):
    #     raise NotImplementedError('Invalid sample split. Only sorted test indices are supported.')

    if not set(train_index).issubset(range(n_obs)):
        raise ValueError('Invalid sample split. Train indices must be in [0, n_obs).')
    if not set(test_index).issubset(range(n_obs)):
        raise ValueError('Invalid sample split. Test indices must be in [0, n_obs).')

    return train_index, test_index


def _check_finite_predictions(preds, learner, learner_name, smpls):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if not np.all(np.isfinite(preds[test_indices])):
        raise ValueError(f'Predictions from learner {str(learner)} for {learner_name} are not finite.')
    return


def _check_score(score, valid_score, allow_callable=True):
    if isinstance(score, str):
        if score not in valid_score:
            raise ValueError('Invalid score ' + score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')
    else:
        if allow_callable:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                f'{str(score)} was passed.')
        else:
            raise TypeError('score should be a string. '
                            f'{str(score)} was passed.')
    return


def _check_trimming(trimming_rule, trimming_threshold):
    valid_trimming_rule = ['truncate']
    if trimming_rule not in valid_trimming_rule:
        raise ValueError('Invalid trimming_rule ' + str(trimming_rule) + '. ' +
                         'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
    if not isinstance(trimming_threshold, float):
        raise TypeError('trimming_threshold has to be a float. ' +
                        f'Object of type {str(type(trimming_threshold))} passed.')
    if (trimming_threshold <= 0) | (trimming_threshold >= 0.5):
        raise ValueError('Invalid trimming_threshold ' + str(trimming_threshold) + '. ' +
                         'trimming_threshold has to be between 0 and 0.5.')
    return


def _check_zero_one_treatment(obj_dml):
    one_treat = (obj_dml._dml_data.n_treat == 1)
    binary_treat = (type_of_target(obj_dml._dml_data.d) == 'binary')
    zero_one_treat = np.all((np.power(obj_dml._dml_data.d, 2) - obj_dml._dml_data.d) == 0)
    if not (one_treat & binary_treat & zero_one_treat):
        raise ValueError('Incompatible data. '
                         f'To fit an {str(obj_dml.score)} model with DML '
                         'exactly one binary variable with values 0 and 1 '
                         'needs to be specified as treatment variable.')


def _check_treatment(treatment):
    if not isinstance(treatment, int):
        raise TypeError('Treatment indicator has to be an integer. ' +
                        f'Object of type {str(type(treatment))} passed.')

    if (treatment != 0) & (treatment != 1):
        raise ValueError('Treatment indicator has be either 0 or 1. ' +
                         f'Treatment indicator {str(treatment)} passed.')
    return


def _check_quantile(quantile):
    if not isinstance(quantile, float):
        raise TypeError('Quantile has to be a float. ' +
                        f'Object of type {str(type(quantile))} passed.')

    if (quantile <= 0) | (quantile >= 1):
        raise ValueError('Quantile has be between 0 or 1. ' +
                         f'Quantile {str(quantile)} passed.')
    return


def _check_contains_iv(obj_dml_data):
    if obj_dml_data.z_cols is not None:
        raise ValueError('Incompatible data. ' +
                         ' and '.join(obj_dml_data.z_cols) +
                         ' have been set as instrumental variable(s). '
                         'To fit an local model see the documentation.')
    return


def _check_is_propensity(preds, learner, learner_name, smpls, eps=1e-12):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if any((preds[test_indices] < eps) | (preds[test_indices] > 1 - eps)):
        warnings.warn(f'Propensity predictions from learner {str(learner)} for'
                      f' {learner_name} are close to zero or one (eps={eps}).')
    return


def _check_benchmarks(benchmarks):
    if benchmarks is not None:
        if not isinstance(benchmarks, dict):
            raise TypeError('benchmarks has to be either None or a dictionary. '
                            f'{str(benchmarks)} of type {type(benchmarks)} was passed.')
        if not set(benchmarks.keys()) == {'cf_y', 'cf_d', 'name'}:
            raise ValueError('benchmarks has to be a dictionary with keys cf_y, cf_d and name. '
                             f'Got {str(benchmarks.keys())}.')

        value_lengths = [len(value) for value in benchmarks.values()]
        if not len(set(value_lengths)) == 1:
            raise ValueError('benchmarks has to be a dictionary with values of same length. '
                             f'Got {str(value_lengths)}.')
        for i in (range(value_lengths[0])):
            for key in ['cf_y', 'cf_d']:
                _check_in_zero_one(benchmarks[key][i], f"benchmarks {key}", include_zero=True, include_one=False)
            if not isinstance(benchmarks["name"][i], str):
                raise TypeError('benchmarks name must be of string type. '
                                f'{str(benchmarks["name"][i])} of type {str(type(benchmarks["name"][i]))} was passed.')
    return


def _check_weights(weights, score, n_obs, n_rep):
    if weights is not None:

        # check general type
        if (not isinstance(weights, np.ndarray)) and (not isinstance(weights, dict)):
            raise TypeError("weights must be a numpy array or dictionary. "
                            f"weights of type {str(type(weights))} was passed.")

        # check shape
        if isinstance(weights, np.ndarray):
            if (weights.ndim != 1) or weights.shape[0] != n_obs:
                raise ValueError(f"weights must have shape ({n_obs},). "
                                 f"weights of shape {weights.shape} was passed.")
            if not np.all(0 <= weights):
                raise ValueError("All weights values must be greater or equal 0.")
            if weights.sum() == 0:
                raise ValueError("At least one weight must be non-zero.")

        # check special form for ATTE score
        if score == "ATTE":
            if not isinstance(weights, np.ndarray):
                raise TypeError("weights must be a numpy array for ATTE score. "
                                f"weights of type {str(type(weights))} was passed.")

            is_binary = np.all((np.power(weights, 2) - weights) == 0)
            if not is_binary:
                raise ValueError("weights must be binary for ATTE score.")

        # check general form for ATE score
        if isinstance(weights, dict):
            assert score == "ATE"
            expected_keys = ["weights", "weights_bar"]
            if not set(weights.keys()) == set(expected_keys):
                raise ValueError(f"weights must have keys {expected_keys}. "
                                 f"keys {str(weights.keys())} were passed.")

            expected_shapes = [(n_obs,), (n_obs, n_rep)]
            if weights["weights"].shape != expected_shapes[0]:
                raise ValueError(f"weights must have shape {expected_shapes[0]}. "
                                 f"weights of shape {weights['weights'].shape} was passed.")
            if weights["weights_bar"].shape != expected_shapes[1]:
                raise ValueError(f"weights_bar must have shape {expected_shapes[1]}. "
                                 f"weights_bar of shape {weights['weights_bar'].shape} was passed.")
            if (not np.all(weights["weights"] >= 0)) or (not np.all(weights["weights_bar"] >= 0)):
                raise ValueError("All weights values must be greater or equal 0.")
            if (weights["weights"].sum() == 0) or (weights["weights_bar"].sum() == 0):
                raise ValueError("At least one weight must be non-zero.")
    return


def _check_external_predictions(external_predictions, valid_treatments, valid_learners, n_obs, n_rep):
    if external_predictions is not None:
        if not isinstance(external_predictions, dict):
            raise TypeError('external_predictions must be a dictionary. '
                            f'{str(external_predictions)} of type {str(type(external_predictions))} was passed.')

        supplied_treatments = list(external_predictions.keys())
        if not set(supplied_treatments).issubset(valid_treatments):
            raise ValueError('Invalid external_predictions. '
                             f'Invalid treatment variable in {str(supplied_treatments)}. '
                             'Valid treatment variables ' + ' or '.join(valid_treatments) + '.')

        for treatment in supplied_treatments:
            if not isinstance(external_predictions[treatment], dict):
                raise TypeError('external_predictions must be a nested dictionary. '
                                f'For treatment {str(treatment)} a value of type '
                                f'{str(type(external_predictions[treatment]))} was passed.')

            supplied_learners = list(external_predictions[treatment].keys())
            if not set(supplied_learners).issubset(valid_learners):
                raise ValueError('Invalid external_predictions. '
                                 f'Invalid nuisance learner for treatment {str(treatment)} in {str(supplied_learners)}. '
                                 'Valid nuisance learners ' + ' or '.join(valid_learners) + '.')

            for learner in supplied_learners:
                if not isinstance(external_predictions[treatment][learner], np.ndarray):
                    raise TypeError('Invalid external_predictions. '
                                    'The values of the nested list must be a numpy array. '
                                    'Invalid predictions for treatment ' + str(treatment) +
                                    ' and learner ' + str(learner) + '. ' +
                                    f'Object of type {str(type(external_predictions[treatment][learner]))} was passed.')

                expected_shape = (n_obs, n_rep)
                if external_predictions[treatment][learner].shape != expected_shape:
                    raise ValueError('Invalid external_predictions. '
                                     f'The supplied predictions have to be of shape {str(expected_shape)}. '
                                     'Invalid predictions for treatment ' + str(treatment) +
                                     ' and learner ' + str(learner) + '. ' +
                                     f'Predictions of shape {str(external_predictions[treatment][learner].shape)} passed.')


def _check_bootstrap(method, n_rep_boot):

    if (not isinstance(method, str)) | (method not in ['Bayes', 'normal', 'wild']):
        raise ValueError('Method must be "Bayes", "normal" or "wild". '
                         f'Got {str(method)}.')

    if not isinstance(n_rep_boot, int):
        raise TypeError('The number of bootstrap replications must be of int type. '
                        f'{str(n_rep_boot)} of type {str(type(n_rep_boot))} was passed.')
    if n_rep_boot < 1:
        raise ValueError('The number of bootstrap replications must be positive. '
                         f'{str(n_rep_boot)} was passed.')
    return


def _check_framework_compatibility(dml_framework_1, dml_framework_2, check_treatments=True):
    if not dml_framework_1.n_obs == dml_framework_2.n_obs:
        raise ValueError('The number of observations in DoubleMLFrameworks must be the same. '
                         f'Got {str(dml_framework_1.n_obs)} and {str(dml_framework_2.n_obs)}.')

    if not dml_framework_1.n_rep == dml_framework_2.n_rep:
        raise ValueError('The number of replications in DoubleMLFrameworks must be the same. '
                         f'Got {str(dml_framework_1.n_rep)} and {str(dml_framework_2.n_rep)}.')

    if check_treatments:
        if not dml_framework_1.n_thetas == dml_framework_2.n_thetas:
            raise ValueError('The number of parameters theta in DoubleMLFrameworks must be the same. '
                             f'Got {str(dml_framework_1.n_thetas)} and {str(dml_framework_2.n_thetas)}.')


def _check_set(x):
    return {x} if x is not None else {}
