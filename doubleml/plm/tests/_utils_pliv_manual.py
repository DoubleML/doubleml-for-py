import numpy as np
from sklearn.base import clone

from ...tests._utils_boot import boot_manual, draw_weights
from ...tests._utils import fit_predict, tune_grid_search


def fit_pliv(y, x, d, z,
             learner_l, learner_m, learner_r, learner_g, all_smpls, score,
             n_rep=1, l_params=None, m_params=None, r_params=None, g_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_l_hat = list()
    all_m_hat = list()
    all_r_hat = list()
    all_g_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        fit_g = (score == 'IV-type') | callable(score)
        l_hat, m_hat, r_hat, g_hat = fit_nuisance_pliv(y, x, d, z,
                                                       learner_l, learner_m, learner_r, learner_g,
                                                       smpls, fit_g,
                                                       l_params, m_params, r_params, g_params)

        all_l_hat.append(l_hat)
        all_m_hat.append(m_hat)
        all_r_hat.append(r_hat)
        all_g_hat.append(g_hat)

        thetas[i_rep], ses[i_rep] = pliv_dml2(y, x, d, z,
                                              l_hat, m_hat, r_hat, g_hat,
                                              smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_l_hat': all_l_hat, 'all_m_hat': all_m_hat, 'all_r_hat': all_r_hat, 'all_g_hat': all_g_hat}

    return res


def fit_nuisance_pliv(y, x, d, z, ml_l, ml_m, ml_r, ml_g, smpls, fit_g=True,
                      l_params=None, m_params=None, r_params=None, g_params=None):
    l_hat = fit_predict(y, x, ml_l, l_params, smpls)

    m_hat = fit_predict(z, x, ml_m, m_params, smpls)

    r_hat = fit_predict(d, x, ml_r, r_params, smpls)

    if fit_g:
        y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, _ = compute_pliv_residuals(
            y, d, z, l_hat, m_hat, r_hat, [], smpls)
        psi_a = -np.multiply(d_minus_r_hat, z_minus_m_hat)
        psi_b = np.multiply(z_minus_m_hat, y_minus_l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

        ml_g = clone(ml_g)
        g_hat = fit_predict(y - theta_initial * d, x, ml_g, g_params, smpls)
    else:
        g_hat = []

    return l_hat, m_hat, r_hat, g_hat


def tune_nuisance_pliv(y, x, d, z, ml_l, ml_m, ml_r, ml_g, smpls, n_folds_tune,
                       param_grid_l, param_grid_m, param_grid_r, param_grid_g, tune_g=True):
    l_tune_res = tune_grid_search(y, x, ml_l, smpls, param_grid_l, n_folds_tune)

    m_tune_res = tune_grid_search(z, x, ml_m, smpls, param_grid_m, n_folds_tune)

    r_tune_res = tune_grid_search(d, x, ml_r, smpls, param_grid_r, n_folds_tune)

    if tune_g:
        l_hat = np.full_like(y, np.nan)
        m_hat = np.full_like(z, np.nan)
        r_hat = np.full_like(d, np.nan)
        for idx, (train_index, _) in enumerate(smpls):
            l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
            m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
            r_hat[train_index] = r_tune_res[idx].predict(x[train_index, :])
        psi_a = -np.multiply(d - r_hat, z - m_hat)
        psi_b = np.multiply(z - m_hat, y - l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

        g_tune_res = tune_grid_search(y - theta_initial*d, x, ml_g, smpls, param_grid_g, n_folds_tune)
        g_best_params = [xx.best_params_ for xx in g_tune_res]
    else:
        g_best_params = []

    l_best_params = [xx.best_params_ for xx in l_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return l_best_params, m_best_params, r_best_params, g_best_params


def compute_pliv_residuals(y, d, z, l_hat, m_hat, r_hat, g_hat, smpls):
    y_minus_l_hat = np.full_like(y, np.nan, dtype='float64')
    z_minus_m_hat = np.full_like(y, np.nan, dtype='float64')
    d_minus_r_hat = np.full_like(d, np.nan, dtype='float64')
    y_minus_g_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        y_minus_l_hat[test_index] = y[test_index] - l_hat[idx]
        z_minus_m_hat[test_index] = z[test_index] - m_hat[idx]
        d_minus_r_hat[test_index] = d[test_index] - r_hat[idx]
        if len(g_hat) > 0:
            y_minus_g_hat[test_index] = y[test_index] - g_hat[idx]

    return y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat


def pliv_dml2(y, x, d, z, l_hat, m_hat, r_hat, g_hat, smpls, score):
    n_obs = len(y)
    y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat = compute_pliv_residuals(
        y, d, z, l_hat, m_hat, r_hat, g_hat, smpls)
    theta_hat = pliv_orth(y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat, d, score)
    se = np.sqrt(var_pliv(theta_hat, d, y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat, score, n_obs))

    return theta_hat, se


def var_pliv(theta, d, y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(z_minus_m_hat, d_minus_r_hat)), 2) * \
            np.mean(np.power(np.multiply(y_minus_l_hat - d_minus_r_hat*theta, z_minus_m_hat), 2))
    else:
        assert score == 'IV-type'
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(z_minus_m_hat, d)), 2) * \
            np.mean(np.power(np.multiply(y_minus_g_hat - d*theta, z_minus_m_hat), 2))

    return var


def pliv_orth(y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat, d, score):
    if score == 'partialling out':
        res = np.mean(np.multiply(z_minus_m_hat, y_minus_l_hat))/np.mean(np.multiply(z_minus_m_hat, d_minus_r_hat))
    else:
        assert score == 'IV-type'
        res = np.mean(np.multiply(z_minus_m_hat, y_minus_g_hat))/np.mean(np.multiply(z_minus_m_hat, d))

    return res


def boot_pliv(y, d, z, thetas, ses, all_l_hat, all_m_hat, all_r_hat, all_g_hat,
              all_smpls, score, bootstrap, n_rep_boot,
              n_rep=1, apply_cross_fitting=True):
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_t_stat = boot_pliv_single_split(
            thetas[i_rep], y, d, z, all_l_hat[i_rep], all_m_hat[i_rep], all_r_hat[i_rep], all_g_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_t_stat.append(boot_t_stat)

    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_t_stat


def boot_pliv_single_split(theta, y, d, z, l_hat, m_hat, r_hat, g_hat,
                           smpls, score, se, weights, n_rep_boot, apply_cross_fitting):
    y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat = compute_pliv_residuals(
        y, d, z, l_hat, m_hat, r_hat, g_hat, smpls)

    if apply_cross_fitting:
        if score == 'partialling out':
            J = np.mean(-np.multiply(z_minus_m_hat, d_minus_r_hat))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(z_minus_m_hat, d))
    else:
        test_index = smpls[0][1]
        if score == 'partialling out':
            J = np.mean(-np.multiply(z_minus_m_hat[test_index], d_minus_r_hat[test_index]))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(z_minus_m_hat[test_index], d[test_index]))

    if score == 'partialling out':
        psi = np.multiply(y_minus_l_hat - d_minus_r_hat*theta, z_minus_m_hat)
    else:
        assert score == 'IV-type'
        psi = np.multiply(y_minus_g_hat - d*theta, z_minus_m_hat)

    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_t_stat
