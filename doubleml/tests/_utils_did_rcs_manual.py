import numpy as np
from sklearn.base import clone, is_classifier

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_did_rcs(y, x, d, t,
                learner_g, learner_m, all_smpls, dml_procedure,
                n_rep=1, g_params=None, m_params=None,
                trimming_threshold=1e-12):
    n_obs = len(d)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat = list()
    all_m_hat = list()
    all_p_hat = list()
    all_lambda_hat = list()

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, p_hat, lambda_hat = fit_nuisance_did_rcs(y, x, d, t, learner_g,
                                                               learner_m, smpls, g_params=g_params,
                                                               m_params=m_params, trimming_threshold=trimming_threshold)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)
        all_p_hat.append(p_hat)
        all_lambda_hat.append(lambda_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = did_rcs_dml1(y, d, t, g_hat, m_hat,
                                                     p_hat, lambda_hat, smpls)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = did_rcs_dml2(y, d, t, g_hat, m_hat,
                                                     p_hat, lambda_hat, smpls)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs +
                 np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat, 'all_lambda_hat': all_lambda_hat}

    return res


def fit_nuisance_did_rcs(y, x, d, t, learner_g, learner_m, smpls,
                         g_params=None, m_params=None,
                         trimming_threshold=1e-12):
    ml_g = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]

    p_hat_list = []
    lambda_hat_list = []

    lambda_hat = np.full_like(t, np.nan, dtype='float64')
    for (_, test_index) in smpls:
        p_hat_list.append(np.mean(d[test_index]))
        lambda_hat_list.append(np.mean(t[test_index]))
        lambda_hat[test_index] = lambda_hat_list[-1]

    g_hat_list = fit_predict((t - lambda_hat) * y,
                             x, ml_g, g_params, smpls, train_cond=train_cond0)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    return g_hat_list, m_hat_list, p_hat_list, lambda_hat_list


def compute_did_rcs_residuals(y, g_hat_list, m_hat_list, p_hat_list, lambda_hat_list, smpls):
    g_hat = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    lambda_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        g_hat[test_index] = g_hat_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]
        lambda_hat[test_index] = lambda_hat_list[idx]

    return g_hat, m_hat, p_hat, lambda_hat


def did_rcs_dml1(y, d, t, g_hat_list, m_hat_list, p_hat_list, lambda_hat_list, smpls):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    g_hat, m_hat, p_hat, lambda_hat = compute_did_rcs_residuals(
        y, g_hat_list, m_hat_list, p_hat_list, lambda_hat_list, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = did_ortho_rcs(y[test_index], d[test_index], t[test_index], g_hat[test_index],
                                    m_hat[test_index], p_hat[test_index], lambda_hat[test_index])
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_did_rcs(y, d, t, g_hat, m_hat,
                     p_hat, lambda_hat, theta_hat, n_obs))
    else:
        assert len(smpls) == 1
        se = np.sqrt(var_did_rcs(y[test_index], d[test_index],
                                 t[test_index], g_hat[test_index],
                                 m_hat[test_index], p_hat[test_index],
                                 lambda_hat[test_index], theta_hat[test_index], n_obs))

    return theta_hat, se


def did_rcs_dml2(y, d, t, g_hat_list, m_hat_list, p_hat_list, lambda_hat_list, smpls):
    n_obs = len(y)
    g_hat, m_hat, p_hat, lambda_hat = compute_did_rcs_residuals(y, g_hat_list, m_hat_list,
                                                                p_hat_list, lambda_hat_list, smpls)

    theta_hat = did_ortho_rcs(y, d, t, g_hat,
                              m_hat, p_hat, lambda_hat)
    se = np.sqrt(var_did_rcs(y, d, t, g_hat, m_hat,
                             p_hat, lambda_hat, theta_hat, n_obs))

    return theta_hat, se


def var_did_rcs(y, d, t, g_hat, m_hat, p_hat, lambda_hat, theta, n_obs):
    var = 1/n_obs * np.mean(np.power(_calculate_psi_b(y,
                            d, t, g_hat, m_hat, p_hat, lambda_hat) - theta, 2))
    return var


def did_ortho_rcs(y, d, t, g_hat, m_hat, p_hat, lambda_hat):
    res = _calculate_psi_b(y, d, t, g_hat, m_hat, p_hat, lambda_hat)
    return np.mean(res)


def _calculate_psi_b(y, d, t, g_hat, m_hat, p_hat, lambda_hat):
    res = ((t - lambda_hat) * y) - g_hat
    res *= d - m_hat
    res /= lambda_hat * (1 - lambda_hat) * p_hat * (1 - m_hat)
    return res


def boot_did_rcs(y, d, t, thetas, ses, all_g_hat, all_m_hat, all_p_hat,
                all_lambda_hat, all_smpls, bootstrap, n_rep_boot,
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
        boot_theta, boot_t_stat = boot_did_rcs_single_split(thetas[i_rep], y, d, t, all_g_hat[i_rep],
                                                            all_m_hat[i_rep], all_p_hat[i_rep], all_lambda_hat[i_rep],
                                                            smpls, ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_did_rcs_single_split(theta, y, d, t, g_hat_list, m_hat_list, p_hat_list, lambda_hat_list,
                              smpls, se, weights, n_rep_boot, apply_cross_fitting):
    g_hat, m_hat, p_hat, lambda_hat = compute_did_rcs_residuals(y, g_hat_list, m_hat_list,
                                                                p_hat_list, lambda_hat_list, smpls)

    J = -1

    psi = _calculate_psi_b(y, d, t, g_hat, m_hat, p_hat, lambda_hat) - theta

    boot_theta, boot_t_stat = boot_manual(
        psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_theta, boot_t_stat
