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
