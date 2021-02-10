import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

from ._utils_boot import boot_manual, draw_weights


def fit_nuisance_pliv_partial_z(y, x, d, z, ml_r, smpls, r_params=None):
    XZ = np.hstack((x, z))
    r_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r_params is not None:
            ml_r.set_params(**r_params[idx])
        r_hat.append(ml_r.fit(XZ[train_index], d[train_index]).predict(XZ[test_index]))

    return r_hat


def tune_nuisance_pliv_partial_z(y, x, d, z, ml_r, smpls, n_folds_tune, param_grid_r):
    XZ = np.hstack((x, z))
    r_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        r_grid_search = GridSearchCV(ml_r, param_grid_r,
                                     cv=r_tune_resampling)
        r_tune_res[idx] = r_grid_search.fit(XZ[train_index, :], d[train_index])

    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return r_best_params


def pliv_partial_z_dml1(y, x, d, z, r_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pliv_partial_z_orth(r_hat[idx], y[test_index], d[test_index], score)
    theta_hat = np.mean(thetas)

    r_hat_array = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        r_hat_array[test_index] = r_hat[idx]
    se = np.sqrt(var_pliv_partial_z(theta_hat, r_hat_array, y, d, score, n_obs))

    return theta_hat, se


def pliv_partial_z_dml2(y, x, d, z, r_hat, smpls, score):
    n_obs = len(y)
    r_hat_array = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        r_hat_array[test_index] = r_hat[idx]
    theta_hat = pliv_partial_z_orth(r_hat_array, y, d, score)
    se = np.sqrt(var_pliv_partial_z(theta_hat, r_hat_array, y, d, score, n_obs))

    return theta_hat, se


def var_pliv_partial_z(theta, r_hat, y, d, score, n_obs):
    assert score == 'partialling out'
    var = 1/n_obs * 1/np.power(np.mean(np.multiply(r_hat, d)), 2) * \
        np.mean(np.power(np.multiply(y - d*theta, r_hat), 2))

    return var


def pliv_partial_z_orth(r_hat, y, d, score):
    assert score == 'partialling out'
    res = np.mean(np.multiply(r_hat, y))/np.mean(np.multiply(r_hat, d))

    return res


def boot_pliv_partial_z(theta, y, d, z, r_hat, smpls, score, se, bootstrap, n_rep, dml_procedure):
    n_obs = len(y)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_pliv_partial_z_single_treat(theta, y, d, z, r_hat,
                                                               smpls, score, se, weights, n_rep, dml_procedure)
    return boot_theta, boot_t_stat


def boot_pliv_partial_z_single_treat(theta, y, d, z, r_hat, smpls, score, se, weights, n_rep, dml_procedure):
    assert score == 'partialling out'
    r_hat_array = np.zeros_like(d, dtype='float64')
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (_, test_index) in enumerate(smpls):
        r_hat_array[test_index] = r_hat[idx]
        if dml_procedure == 'dml1':
            J[idx] = np.mean(-np.multiply(r_hat_array[test_index], d[test_index]))

    if dml_procedure == 'dml2':
        J = np.mean(-np.multiply(r_hat_array, d))

    psi = np.multiply(y - d*theta, r_hat_array)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure)

    return boot_theta, boot_t_stat
