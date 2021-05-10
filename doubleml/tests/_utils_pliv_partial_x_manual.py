import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV

from ._utils_boot import boot_manual, draw_weights


def fit_pliv_partial_x(y, x, d, z,
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

        g_hat, m_hat, r_hat = fit_nuisance_pliv_partial_x(y, x, d, z,
                                                          learner_g, learner_m, learner_r,
                                                          smpls,
                                                          g_params, m_params, r_params)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)
        all_r_hat.append(r_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = pliv_partial_x_dml1(y, x, d,
                                                            z,
                                                            g_hat, m_hat, r_hat,
                                                            smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = pliv_partial_x_dml2(y, x, d,
                                                            z,
                                                            g_hat, m_hat, r_hat,
                                                            smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat, 'all_r_hat': all_r_hat}

    return res


def fit_nuisance_pliv_partial_x(y, x, d, z, ml_g, ml_m, ml_r, smpls, g_params=None, m_params=None, r_params=None):
    assert z.ndim == 2
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(x[train_index], y[train_index]).predict(x[test_index]))

    m_hat = []
    m_hat_array = np.zeros_like(z, dtype='float64')
    for i_instr in range(z.shape[1]):
        this_instr_m_hat = []
        for idx, (train_index, test_index) in enumerate(smpls):
            if m_params is not None:
                ml_m.set_params(**m_params[i_instr][idx])
            this_instr_m_hat.append(ml_m.fit(x[train_index], z[train_index, i_instr]).predict(x[test_index]))
            m_hat_array[test_index, i_instr] = this_instr_m_hat[idx]
        m_hat.append(this_instr_m_hat)

    r_hat = []
    r_hat_array = np.zeros_like(d, dtype='float64')
    for idx, (train_index, test_index) in enumerate(smpls):
        if r_params is not None:
            ml_r.set_params(**r_params[idx])
        r_hat.append(ml_r.fit(x[train_index], d[train_index]).predict(x[test_index]))
        r_hat_array[test_index] = r_hat[idx]

    r_hat_tilde = LinearRegression(fit_intercept=True).fit(z - m_hat_array, d - r_hat_array).predict(z - m_hat_array)

    return g_hat, r_hat, r_hat_tilde


def tune_nuisance_pliv_partial_x(y, x, d, z, ml_g, ml_m, ml_r, smpls, n_folds_tune,
                                 param_grid_g, param_grid_m, param_grid_r):
    g_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g_grid_search = GridSearchCV(ml_g, param_grid_g,
                                     cv=g_tune_resampling)
        g_tune_res[idx] = g_grid_search.fit(x[train_index, :], y[train_index])

    m_tune_res = [[None] * len(smpls) for i in range(z.shape[1])]
    for i_instr in range(z.shape[1]):
        for idx, (train_index, _) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                         cv=m_tune_resampling)
            m_tune_res[i_instr][idx] = m_grid_search.fit(x[train_index, :], z[train_index, i_instr])

    r_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        r_grid_search = GridSearchCV(ml_r, param_grid_r,
                                     cv=r_tune_resampling)
        r_tune_res[idx] = r_grid_search.fit(x[train_index, :], d[train_index])

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [[xx.best_params_ for xx in m_tune_res[i_instr]] for i_instr in range(z.shape[1])]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return g_best_params, m_best_params, r_best_params


def compute_pliv_partial_x_residuals(y, d, g_hat, r_hat, smpls):
    u_hat = np.full_like(y, np.nan, dtype='float64')
    w_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        w_hat[test_index] = d[test_index] - r_hat[idx]

    return u_hat, w_hat


def pliv_partial_x_dml1(y, x, d, z, g_hat, r_hat, r_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    u_hat, w_hat = compute_pliv_partial_x_residuals(y, d, g_hat, r_hat, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pliv_partial_x_orth(u_hat[test_index], w_hat[test_index], r_hat_tilde[test_index],
                                          d[test_index], score)
    theta_hat = np.mean(thetas)

    se = np.sqrt(var_pliv_partial_x(theta_hat, d, u_hat, w_hat, r_hat_tilde, score, n_obs))

    return theta_hat, se


def pliv_partial_x_dml2(y, x, d, z, g_hat, r_hat, r_hat_tilde, smpls, score):
    n_obs = len(y)
    u_hat, w_hat = compute_pliv_partial_x_residuals(y, d, g_hat, r_hat, smpls)
    theta_hat = pliv_partial_x_orth(u_hat, w_hat, r_hat_tilde, d, score)
    se = np.sqrt(var_pliv_partial_x(theta_hat, d, u_hat, w_hat, r_hat_tilde, score, n_obs))

    return theta_hat, se


def var_pliv_partial_x(theta, d, u_hat, w_hat, r_hat_tilde, score, n_obs):
    assert score == 'partialling out'
    var = 1/n_obs * 1/np.power(np.mean(np.multiply(r_hat_tilde, w_hat)), 2) * \
        np.mean(np.power(np.multiply(u_hat - w_hat*theta, r_hat_tilde), 2))

    return var


def pliv_partial_x_orth(u_hat, w_hat, r_hat_tilde, d, score):
    assert score == 'partialling out'
    res = np.mean(np.multiply(r_hat_tilde, u_hat))/np.mean(np.multiply(r_hat_tilde, w_hat))

    return res


def boot_pliv_partial_x(y, d, z, thetas, ses, all_g_hat, all_m_hat, all_r_hat,
                        all_smpls, score, bootstrap, n_rep_boot,
                        n_rep=1):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        boot_theta, boot_t_stat = boot_pliv_partial_x_single_split(
            thetas[i_rep], y, d, z, all_g_hat[i_rep], all_m_hat[i_rep], all_r_hat[i_rep], all_smpls[i_rep],
            score, ses[i_rep], bootstrap, n_rep_boot)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_pliv_partial_x_single_split(theta, y, d, z, g_hat, r_hat, r_hat_tilde,
                                     smpls, score, se, bootstrap, n_rep):
    n_obs = len(y)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_pliv_partial_x_single_treat(theta, y, d, z, g_hat, r_hat, r_hat_tilde,
                                                               smpls, score, se, weights, n_rep)
    return boot_theta, boot_t_stat


def boot_pliv_partial_x_single_treat(theta, y, d, z, g_hat, r_hat, r_hat_tilde,
                                     smpls, score, se, weights, n_rep):
    assert score == 'partialling out'
    u_hat, w_hat = compute_pliv_partial_x_residuals(y, d, g_hat, r_hat, smpls)

    J = np.mean(-np.multiply(r_hat_tilde, w_hat))

    psi = np.multiply(u_hat - w_hat*theta, r_hat_tilde)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep)

    return boot_theta, boot_t_stat
