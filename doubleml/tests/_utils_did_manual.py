import numpy as np
from sklearn.base import clone, is_classifier

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_did_ro(y0, y1, x, d,
               learner_g, learner_m, all_smpls, dml_procedure,
               n_rep=1, g_params=None, m_params=None,
               trimming_threshold=1e-12):
    n_obs = len(d)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat = list()
    all_m_hat = list()
    all_p_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, p_hat = fit_nuisance_did_ro(y0, y1, x, d,
                                                  learner_g, learner_m, smpls,
                                                  g_params=g_params, m_params=m_params,
                                                  trimming_threshold=trimming_threshold)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)
        all_p_hat.append(p_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = did_ro_dml1(y0, y1, x, d,
                                                    g_hat, m_hat, p_hat,
                                                    smpls)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = did_ro_dml2(y0, y1, x, d,
                                                    g_hat, m_hat, p_hat,
                                                    smpls)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs +
                 np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat}

    return res


def fit_nuisance_did_ro(y0, y1, x, d, learner_g, learner_m, smpls,
                        g_params=None, m_params=None,
                        trimming_threshold=1e-12):
    ml_g = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]

    g_hat_list = fit_predict(y1 - y0, x, ml_g, g_params, smpls,
                             train_cond=train_cond0)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    p_hat_list = []
    for (_, test_index) in smpls:
        p_hat_list.append(np.mean(d[test_index]))

    return g_hat_list, m_hat_list, p_hat_list


def compute_did_ro_residuals(y0, y1, g_hat_list, m_hat_list, p_hat_list, smpls):
    g_hat = np.full_like(y0, np.nan, dtype='float64')
    m_hat = np.full_like(y0, np.nan, dtype='float64')
    p_hat = np.full_like(y0, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        g_hat[test_index] = g_hat_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]

    return g_hat, m_hat, p_hat


def did_ro_dml1(y0, y1, x, d, g_hat_list, m_hat_list, p_hat_list, smpls):
    thetas = np.zeros(len(smpls))
    n_obs = len(y0)
    g_hat, m_hat, p_hat = compute_did_ro_residuals(
        y0, y1, g_hat_list, m_hat_list, p_hat_list, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = did_ortho_ro(y0[test_index], y1[test_index], d[test_index],
                                   g_hat[test_index], m_hat[test_index], p_hat[test_index])
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = 0
    else:
        assert len(smpls) == 1
        se = 0

    return theta_hat, se


def did_ro_dml2(y0, y1, x, d, g_hat_list, m_hat_list, p_hat_list, smpls):
    n_obs = len(y0)
    g_hat, m_hat, p_hat = compute_did_ro_residuals(
        y0, y1, g_hat_list, m_hat_list, p_hat_list, smpls)

    theta_hat = did_ortho_ro(y0, y1, d, g_hat, m_hat, p_hat)
    se = 0

    return theta_hat, se


def var_irm(theta, g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, d, score, n_obs):
    if score == 'ATE':
        var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0
                                         + np.divide(np.multiply(d,
                                                     u_hat1), m_hat)
                                         - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat) - theta, 2))
    else:
        assert score == 'ATTE'
        var = 1/n_obs * np.mean(np.power(np.divide(np.multiply(d, u_hat0), p_hat)
                                         - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                                                     np.multiply(p_hat, (1.-m_hat)))
                                         - theta * np.divide(d, p_hat), 2)) \
            / np.power(np.mean(np.divide(d, p_hat)), 2)

    return var


def did_ortho_ro(y0, y1, d, g_hat, m_hat, p_hat):
    res = (y1 - y0)/p_hat
    res *= (d - m_hat)/(1-m_hat)
    c_1 = (d - m_hat)/((1 - m_hat)*p_hat)
    c_1 *= g_hat
    res -= c_1

    return np.mean(res)


def boot_did_ro(y0, y1, d, thetas, ses, all_g_hat, all_m_hat, all_p_hat,
                all_smpls, bootstrap, n_rep_boot,
                n_rep=1, apply_cross_fitting=True):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y0)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_theta, boot_t_stat = boot_did_ro_single_split(thetas[i_rep], y0, y1, d,
                                                           all_g_hat[i_rep], all_m_hat[i_rep], all_p_hat[i_rep], smpls,
                                                           ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_did_ro_single_split(theta, y0, y1, d, g_hat_list, m_hat_list, p_hat_list,
                             smpls, se, weights, n_rep_boot, apply_cross_fitting):
    g_hat, m_hat, p_hat = compute_did_ro_residuals(
        y0, y1, g_hat_list, m_hat_list, p_hat_list, smpls)

    if apply_cross_fitting:
        J = np.mean(-np.divide(d, p_hat))
    else:
        test_index = smpls[0][1]
        J = np.mean(-np.divide(d[test_index], p_hat[test_index]))

    psi = (y1 - y0)/p_hat
    psi *= (d - m_hat)/(1-m_hat)
    c_1 = (d - m_hat)/((1 - m_hat)*p_hat)
    c_1 *= g_hat
    psi -= c_1 + theta
    psi = np.mean(psi)

    boot_theta, boot_t_stat = boot_manual(
        psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_theta, boot_t_stat
