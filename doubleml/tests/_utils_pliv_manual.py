import numpy as np

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, tune_grid_search


def fit_pliv(y, x, d, z,
             learner_g, learner_m, learner_r, all_smpls, dml_procedure, score,
             n_rep=1, g_params=None, m_params=None, r_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat = list()
    all_m_hat = list()
    all_r_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, r_hat = fit_nuisance_pliv(y, x, d, z,
                                                learner_g, learner_m, learner_r,
                                                smpls,
                                                g_params, m_params, r_params)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)
        all_r_hat.append(r_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = pliv_dml1(y, x, d,
                                                  z,
                                                  g_hat, m_hat, r_hat,
                                                  smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = pliv_dml2(y, x, d,
                                                  z,
                                                  g_hat, m_hat, r_hat,
                                                  smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat, 'all_r_hat': all_r_hat}

    return res


def fit_nuisance_pliv(y, x, d, z, ml_g, ml_m, ml_r, smpls, g_params=None, m_params=None, r_params=None):
    g_hat = fit_predict(y, x, ml_g, g_params, smpls)

    m_hat = fit_predict(z, x, ml_m, m_params, smpls)

    r_hat = fit_predict(d, x, ml_r, r_params, smpls)

    return g_hat, m_hat, r_hat


def tune_nuisance_pliv(y, x, d, z, ml_g, ml_m, ml_r, smpls, n_folds_tune, param_grid_g, param_grid_m, param_grid_r):
    g_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune)

    m_tune_res = tune_grid_search(z, x, ml_m, smpls, param_grid_m, n_folds_tune)

    r_tune_res = tune_grid_search(d, x, ml_r, smpls, param_grid_r, n_folds_tune)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return g_best_params, m_best_params, r_best_params


def compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls):
    u_hat = np.full_like(y, np.nan, dtype='float64')
    v_hat = np.full_like(z, np.nan, dtype='float64')
    w_hat = np.full_like(d, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = z[test_index] - m_hat[idx]
        w_hat[test_index] = d[test_index] - r_hat[idx]

    return u_hat, v_hat, w_hat


def pliv_dml1(y, x, d, z, g_hat, m_hat, r_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    u_hat, v_hat, w_hat = compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pliv_orth(u_hat[test_index], v_hat[test_index], w_hat[test_index], d[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_pliv(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_pliv(theta_hat, d[test_index],
                              u_hat[test_index], v_hat[test_index], w_hat[test_index],
                              score, n_obs))

    return theta_hat, se


def pliv_dml2(y, x, d, z, g_hat, m_hat, r_hat, smpls, score):
    n_obs = len(y)
    u_hat, v_hat, w_hat = compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls)
    theta_hat = pliv_orth(u_hat, v_hat, w_hat, d, score)
    se = np.sqrt(var_pliv(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))

    return theta_hat, se


def var_pliv(theta, d, u_hat, v_hat, w_hat, score, n_obs):
    assert score == 'partialling out'
    var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, w_hat)), 2) * \
        np.mean(np.power(np.multiply(u_hat - w_hat*theta, v_hat), 2))

    return var


def pliv_orth(u_hat, v_hat, w_hat, d, score):
    assert score == 'partialling out'
    res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, w_hat))

    return res


def boot_pliv(y, d, z, thetas, ses, all_g_hat, all_m_hat, all_r_hat,
              all_smpls, score, bootstrap, n_rep_boot,
              n_rep=1, apply_cross_fitting=True):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_theta, boot_t_stat = boot_pliv_single_split(
            thetas[i_rep], y, d, z, all_g_hat[i_rep], all_m_hat[i_rep], all_r_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_pliv_single_split(theta, y, d, z, g_hat, m_hat, r_hat,
                           smpls, score, se, weights, n_rep_boot, apply_cross_fitting):
    assert score == 'partialling out'
    u_hat, v_hat, w_hat = compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls)

    if apply_cross_fitting:
        J = np.mean(-np.multiply(v_hat, w_hat))
    else:
        test_index = smpls[0][1]
        J = np.mean(-np.multiply(v_hat[test_index], w_hat[test_index]))

    psi = np.multiply(u_hat - w_hat*theta, v_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_theta, boot_t_stat
