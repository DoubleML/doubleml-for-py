import numpy as np

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, tune_grid_search


def fit_pliv_partial_z(y, x, d, z,
                       learner_r, all_smpls, dml_procedure, score,
                       n_rep=1, r_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_r_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        r_hat = fit_nuisance_pliv_partial_z(y, x, d, z,
                                            learner_r,
                                            smpls,
                                            r_params)

        all_r_hat.append(r_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = pliv_partial_z_dml1(y, x, d,
                                                            z,
                                                            r_hat,
                                                            smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = pliv_partial_z_dml2(y, x, d,
                                                            z,
                                                            r_hat,
                                                            smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_r_hat': all_r_hat}

    return res


def fit_nuisance_pliv_partial_z(y, x, d, z, ml_r, smpls, r_params=None):
    xz = np.hstack((x, z))
    r_hat = fit_predict(d, xz, ml_r, r_params, smpls)

    return r_hat


def tune_nuisance_pliv_partial_z(y, x, d, z, ml_r, smpls, n_folds_tune, param_grid_r):
    xz = np.hstack((x, z))
    r_tune_res = tune_grid_search(d, xz, ml_r, smpls, param_grid_r, n_folds_tune)

    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return r_best_params


def compute_pliv_partial_z_residuals(y, r_hat, smpls):
    r_hat_array = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        r_hat_array[test_index] = r_hat[idx]
    return r_hat_array


def pliv_partial_z_dml1(y, x, d, z, r_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    r_hat_array = compute_pliv_partial_z_residuals(y, r_hat, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pliv_partial_z_orth(r_hat_array[test_index], y[test_index], d[test_index], score)
    theta_hat = np.mean(thetas)

    se = np.sqrt(var_pliv_partial_z(theta_hat, r_hat_array, y, d, score, n_obs))

    return theta_hat, se


def pliv_partial_z_dml2(y, x, d, z, r_hat, smpls, score):
    n_obs = len(y)
    r_hat_array = compute_pliv_partial_z_residuals(y, r_hat, smpls)
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


def boot_pliv_partial_z(y, d, z, thetas, ses, all_r_hat,
                        all_smpls, score, bootstrap, n_rep_boot,
                        n_rep=1):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        n_obs = len(y)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_theta, boot_t_stat = boot_pliv_partial_z_single_split(
            thetas[i_rep], y, d, z, all_r_hat[i_rep], all_smpls[i_rep],
            score, ses[i_rep], weights, n_rep_boot)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_pliv_partial_z_single_split(theta, y, d, z, r_hat,
                                     smpls, score, se, weights, n_rep_boot):
    assert score == 'partialling out'
    r_hat_array = compute_pliv_partial_z_residuals(y, r_hat, smpls)

    J = np.mean(-np.multiply(r_hat_array, d))

    psi = np.multiply(y - d*theta, r_hat_array)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

    return boot_theta, boot_t_stat
