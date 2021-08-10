import numpy as np
import scipy
from sklearn.base import clone, is_classifier

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_plr_multitreat(y, x, d, learner_g, learner_m, all_smpls, dml_procedure, score,
                       n_rep=1, g_params=None, m_params=None,
                       use_other_treat_as_covariate=True):
    n_obs = len(y)
    n_d = d.shape[1]

    thetas = list()
    ses = list()
    all_g_hat = list()
    all_m_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        thetas_this_rep = np.full(n_d, np.nan)
        ses_this_rep = np.full(n_d, np.nan)
        all_g_hat_this_rep = list()
        all_m_hat_this_rep = list()

        for i_d in range(n_d):
            if use_other_treat_as_covariate:
                xd = np.hstack((x, np.delete(d, i_d, axis=1)))
            else:
                xd = x

            g_hat, m_hat, thetas_this_rep[i_d], ses_this_rep[i_d] = fit_plr_single_split(
                y, xd, d[:, i_d], learner_g, learner_m, smpls, dml_procedure, score, g_params, m_params)
            all_g_hat_this_rep.append(g_hat)
            all_m_hat_this_rep.append(m_hat)

        thetas.append(thetas_this_rep)
        ses.append(ses_this_rep)
        all_g_hat.append(all_g_hat_this_rep)
        all_m_hat.append(all_m_hat_this_rep)

    theta = np.full(n_d, np.nan)
    se = np.full(n_d, np.nan)
    for i_d in range(n_d):
        theta_vec = np.array([xx[i_d] for xx in thetas])
        se_vec = np.array([xx[i_d] for xx in ses])
        theta[i_d] = np.median(theta_vec)
        se[i_d] = np.sqrt(np.median(np.power(se_vec, 2) * n_obs + np.power(theta_vec - theta[i_d], 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat}

    return res


def fit_plr(y, x, d, learner_g, learner_m, all_smpls, dml_procedure, score,
            n_rep=1, g_params=None, m_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat = list()
    all_m_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        g_hat, m_hat, thetas[i_rep], ses[i_rep] = fit_plr_single_split(
            y, x, d, learner_g, learner_m, smpls, dml_procedure, score, g_params, m_params)
        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat': all_g_hat, 'all_m_hat': all_m_hat}

    return res


def fit_plr_single_split(y, x, d, learner_g, learner_m, smpls, dml_procedure, score, g_params=None, m_params=None):
    if is_classifier(learner_m):
        g_hat, m_hat = fit_nuisance_plr_classifier(y, x, d,
                                                   learner_g, learner_m, smpls,
                                                   g_params, m_params)
    else:
        g_hat, m_hat = fit_nuisance_plr(y, x, d,
                                        learner_g, learner_m, smpls,
                                        g_params, m_params)

    if dml_procedure == 'dml1':
        theta, se = plr_dml1(y, x, d,
                             g_hat, m_hat,
                             smpls, score)
    else:
        assert dml_procedure == 'dml2'
        theta, se = plr_dml2(y, x, d,
                             g_hat, m_hat,
                             smpls, score)

    return g_hat, m_hat, theta, se


def fit_nuisance_plr(y, x, d, learner_g, learner_m, smpls, g_params=None, m_params=None):
    ml_g = clone(learner_g)
    g_hat = fit_predict(y, x, ml_g, g_params, smpls)

    ml_m = clone(learner_m)
    m_hat = fit_predict(d, x, ml_m, m_params, smpls)

    return g_hat, m_hat


def fit_nuisance_plr_classifier(y, x, d, learner_g, learner_m, smpls, g_params=None, m_params=None):
    ml_g = clone(learner_g)
    g_hat = fit_predict(y, x, ml_g, g_params, smpls)

    ml_m = clone(learner_m)
    m_hat = fit_predict_proba(d, x, ml_m, m_params, smpls)

    return g_hat, m_hat


def tune_nuisance_plr(y, x, d, ml_g, ml_m, smpls, n_folds_tune, param_grid_g, param_grid_m):
    g_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_best_params, m_best_params


def compute_plr_residuals(y, d, g_hat, m_hat, smpls):
    u_hat = np.full_like(y, np.nan, dtype='float64')
    v_hat = np.full_like(d, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = d[test_index] - m_hat[idx]
    return u_hat, v_hat


def plr_dml1(y, x, d, g_hat, m_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    u_hat, v_hat = compute_plr_residuals(y, d, g_hat, m_hat, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = plr_orth(v_hat[test_index], u_hat[test_index], d[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_plr(theta_hat, d[test_index], u_hat[test_index], v_hat[test_index], score, n_obs))

    return theta_hat, se


def plr_dml2(y, x, d, g_hat, m_hat, smpls, score):
    n_obs = len(y)
    u_hat, v_hat = compute_plr_residuals(y, d, g_hat, m_hat, smpls)
    theta_hat = plr_orth(v_hat, u_hat, d, score)
    se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, score, n_obs))

    return theta_hat, se


def var_plr(theta, d, u_hat, v_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
            np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    else:
        assert score == 'IV-type'
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, d)), 2) * \
            np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))

    return var


def plr_orth(v_hat, u_hat, d, score):
    if score == 'IV-type':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, d))
    else:
        assert score == 'partialling out'
        res = scipy.linalg.lstsq(v_hat.reshape(-1, 1), u_hat)[0]

    return res


def boot_plr(y, d, thetas, ses, all_g_hat, all_m_hat,
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

        boot_theta, boot_t_stat = boot_plr_single_split(
            thetas[i_rep], y, d, all_g_hat[i_rep], all_m_hat[i_rep], smpls,
            score, ses[i_rep],
            weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_plr_multitreat(y, d, thetas, ses, all_g_hat, all_m_hat,
                        all_smpls, score, bootstrap, n_rep_boot,
                        n_rep=1, apply_cross_fitting=True):
    n_d = d.shape[1]
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

        boot_theta = np.full((n_d, n_rep_boot), np.nan)
        boot_t_stat = np.full((n_d, n_rep_boot), np.nan)
        for i_d in range(n_d):
            boot_theta[i_d, :], boot_t_stat[i_d, :] = boot_plr_single_split(
                thetas[i_rep][i_d], y, d[:, i_d],
                all_g_hat[i_rep][i_d], all_m_hat[i_rep][i_d],
                smpls, score, ses[i_rep][i_d],
                weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_plr_single_split(theta, y, d, g_hat, m_hat,
                          smpls, score, se, weights, n_rep, apply_cross_fitting):
    u_hat, v_hat = compute_plr_residuals(y, d, g_hat, m_hat, smpls)

    if apply_cross_fitting:
        if score == 'partialling out':
            J = np.mean(-np.multiply(v_hat, v_hat))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(v_hat, d))
    else:
        test_index = smpls[0][1]
        if score == 'partialling out':
            J = np.mean(-np.multiply(v_hat[test_index], v_hat[test_index]))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(v_hat[test_index], d[test_index]))

    if score == 'partialling out':
        psi = np.multiply(u_hat - v_hat * theta, v_hat)
    else:
        assert score == 'IV-type'
        psi = np.multiply(u_hat - d * theta, v_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, apply_cross_fitting)

    return boot_theta, boot_t_stat
