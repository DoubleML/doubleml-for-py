import numpy as np
from sklearn.model_selection import KFold


def draw_smpls(n_obs, n_folds, n_rep=1):
    all_smpls = []
    for i_rep in range(n_rep):
        resampling = KFold(n_splits=n_folds,
                           shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(np.zeros(n_obs))]
        all_smpls.append(smpls)
    return all_smpls


def fit_predict(y, x, ml_model, params, smpls, train_cond=None):
    y_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if params is not None:
            ml_model.set_params(**params[idx])
        if train_cond is None:
            y_hat.append(ml_model.fit(x[train_index], y[train_index]).predict(x[test_index]))
        else:
            train_index_cond = np.intersect1d(train_cond, train_index)
            y_hat.append(ml_model.fit(x[train_index_cond], y[train_index_cond]).predict(x[test_index]))

    return y_hat


def fit_predict_proba(y, x, ml_model, params, smpls, trimming_threshold=0, train_cond=None):
    y_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if params is not None:
            ml_model.set_params(**params[idx])
        if train_cond is None:
            preds = ml_model.fit(x[train_index], y[train_index]).predict_proba(x[test_index])[:, 1]
        else:
            train_index_cond = np.intersect1d(train_cond, train_index)
            preds = ml_model.fit(x[train_index_cond], y[train_index_cond]).predict_proba(x[test_index])[:, 1]

        if trimming_threshold > 0:
            preds[preds < trimming_threshold] = trimming_threshold
            preds[preds > 1 - trimming_threshold] = 1 - trimming_threshold
        y_hat.append(preds)

    return y_hat
