import numpy as np
import scipy
from sklearn.base import clone, is_classifier

from ...tests._utils_boot import boot_manual, draw_weights
from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_plr_multitreat(y, x, d, learner_l, learner_m, learner_g, all_smpls, score,
                       n_rep=1, l_params=None, m_params=None, g_params=None,
                       use_other_treat_as_covariate=True):
    n_obs = len(y)
    n_d = d.shape[1]

    thetas = list()
    ses = list()
    all_l_hat = list()
    all_m_hat = list()
    all_g_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        thetas_this_rep = np.full(n_d, np.nan)
        ses_this_rep = np.full(n_d, np.nan)
        all_l_hat_this_rep = list()
        all_m_hat_this_rep = list()
        all_g_hat_this_rep = list()

        for i_d in range(n_d):
            if use_other_treat_as_covariate:
                xd = np.hstack((x, np.delete(d, i_d, axis=1)))
            else:
                xd = x

            l_hat, m_hat, g_hat, thetas_this_rep[i_d], ses_this_rep[i_d] = fit_plr_single_split(
                y, xd, d[:, i_d],
                learner_l, learner_m, learner_g,
                smpls, score,
                l_params, m_params, g_params)
            all_l_hat_this_rep.append(l_hat)
            all_m_hat_this_rep.append(m_hat)
            all_g_hat_this_rep.append(g_hat)

        thetas.append(thetas_this_rep)
        ses.append(ses_this_rep)
        all_l_hat.append(all_l_hat_this_rep)
        all_m_hat.append(all_m_hat_this_rep)
        all_g_hat.append(all_g_hat_this_rep)

    theta = np.full(n_d, np.nan)
    se = np.full(n_d, np.nan)
    for i_d in range(n_d):
        theta_vec = np.array([xx[i_d] for xx in thetas])
        se_vec = np.array([xx[i_d] for xx in ses])
        theta[i_d] = np.median(theta_vec)
        se[i_d] = np.sqrt(np.median(np.power(se_vec, 2) * n_obs + np.power(theta_vec - theta[i_d], 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_l_hat': all_l_hat, 'all_m_hat': all_m_hat, 'all_g_hat': all_g_hat}

    return res


def fit_plr(y, x, d, learner_l, learner_m, learner_g, all_smpls, score,
            n_rep=1, l_params=None, m_params=None, g_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_l_hat = list()
    all_m_hat = list()
    all_g_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        l_hat, m_hat, g_hat, thetas[i_rep], ses[i_rep] = fit_plr_single_split(
            y, x, d,
            learner_l, learner_m, learner_g,
            smpls, score,
            l_params, m_params, g_params)
        all_l_hat.append(l_hat)
        all_m_hat.append(m_hat)
        all_g_hat.append(g_hat)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_l_hat': all_l_hat, 'all_m_hat': all_m_hat, 'all_g_hat': all_g_hat}

    return res


def fit_plr_single_split(y, x, d, learner_l, learner_m, learner_g, smpls, score,
                         l_params=None, m_params=None, g_params=None):
    fit_g = (score == 'IV-type') | callable(score)
    if is_classifier(learner_m):
        l_hat, m_hat, g_hat = fit_nuisance_plr_classifier(y, x, d,
                                                          learner_l, learner_m, learner_g,
                                                          smpls, fit_g,
                                                          l_params, m_params, g_params)
    else:
        l_hat, m_hat, g_hat = fit_nuisance_plr(y, x, d,
                                               learner_l, learner_m, learner_g,
                                               smpls, fit_g,
                                               l_params, m_params, g_params)

    theta, se = plr_dml2(y, x, d, l_hat, m_hat, g_hat,
                         smpls, score)

    return l_hat, m_hat, g_hat, theta, se


def fit_nuisance_plr(y, x, d, learner_l, learner_m, learner_g, smpls, fit_g=True,
                     l_params=None, m_params=None, g_params=None):
    ml_l = clone(learner_l)
    l_hat = fit_predict(y, x, ml_l, l_params, smpls)

    ml_m = clone(learner_m)
    m_hat = fit_predict(d, x, ml_m, m_params, smpls)

    if fit_g:
        y_minus_l_hat, d_minus_m_hat, _ = compute_plr_residuals(y, d, l_hat, m_hat, [], smpls)
        psi_a = -np.multiply(d_minus_m_hat, d_minus_m_hat)
        psi_b = np.multiply(d_minus_m_hat, y_minus_l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

        ml_g = clone(learner_g)
        g_hat = fit_predict(y - theta_initial*d, x, ml_g, g_params, smpls)
    else:
        g_hat = []

    return l_hat, m_hat, g_hat


def fit_nuisance_plr_classifier(y, x, d, learner_l, learner_m, learner_g, smpls, fit_g=True,
                                l_params=None, m_params=None, g_params=None):
    ml_l = clone(learner_l)
    l_hat = fit_predict(y, x, ml_l, l_params, smpls)

    ml_m = clone(learner_m)
    m_hat = fit_predict_proba(d, x, ml_m, m_params, smpls)

    if fit_g:
        y_minus_l_hat, d_minus_m_hat, _ = compute_plr_residuals(y, d, l_hat, m_hat, [], smpls)
        psi_a = -np.multiply(d_minus_m_hat, d_minus_m_hat)
        psi_b = np.multiply(d_minus_m_hat, y_minus_l_hat)
        theta_initial = -np.mean(psi_b) / np.mean(psi_a)

        ml_g = clone(learner_g)
        g_hat = fit_predict(y - theta_initial*d, x, ml_g, g_params, smpls)
    else:
        g_hat = []

    return l_hat, m_hat, g_hat


def tune_nuisance_plr(y, x, d, ml_l, ml_m, ml_g, smpls, n_folds_tune, param_grid_l, param_grid_m, param_grid_g, tune_g=True):
    l_tune_res = tune_grid_search(y, x, ml_l, smpls, param_grid_l, n_folds_tune)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    if tune_g:
        l_hat = np.full_like(y, np.nan)
        m_hat = np.full_like(d, np.nan)
        for idx, (train_index, _) in enumerate(smpls):
            l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
            m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
        psi_a = -np.multiply(d - m_hat, d - m_hat)
        psi_b = np.multiply(d - m_hat, y - l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

        g_tune_res = tune_grid_search(y - theta_initial*d, x, ml_g, smpls, param_grid_g, n_folds_tune)
        g_best_params = [xx.best_params_ for xx in g_tune_res]
    else:
        g_best_params = []

    l_best_params = [xx.best_params_ for xx in l_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return l_best_params, m_best_params, g_best_params


def compute_plr_residuals(y, d, l_hat, m_hat, g_hat, smpls):
    y_minus_l_hat = np.full_like(y, np.nan, dtype='float64')
    d_minus_m_hat = np.full_like(d, np.nan, dtype='float64')
    y_minus_g_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        y_minus_l_hat[test_index] = y[test_index] - l_hat[idx]
        if len(g_hat) > 0:
            y_minus_g_hat[test_index] = y[test_index] - g_hat[idx]
        d_minus_m_hat[test_index] = d[test_index] - m_hat[idx]
    return y_minus_l_hat, d_minus_m_hat, y_minus_g_hat


def plr_dml2(y, x, d, l_hat, m_hat, g_hat, smpls, score):
    n_obs = len(y)
    y_minus_l_hat, d_minus_m_hat, y_minus_g_hat = compute_plr_residuals(y, d, l_hat, m_hat, g_hat, smpls)
    theta_hat = plr_orth(y_minus_l_hat, d_minus_m_hat, y_minus_g_hat, d, score)
    se = np.sqrt(var_plr(theta_hat, d, y_minus_l_hat, d_minus_m_hat, y_minus_g_hat, score, n_obs))

    return theta_hat, se


def var_plr(theta, d, y_minus_l_hat, d_minus_m_hat, y_minus_g_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(d_minus_m_hat, d_minus_m_hat)), 2) * \
            np.mean(np.power(np.multiply(y_minus_l_hat - d_minus_m_hat*theta, d_minus_m_hat), 2))
    else:
        assert score == 'IV-type'
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(d_minus_m_hat, d)), 2) * \
            np.mean(np.power(np.multiply(y_minus_g_hat - d*theta, d_minus_m_hat), 2))

    return var


def plr_orth(y_minus_l_hat, d_minus_m_hat, y_minus_g_hat, d, score):
    if score == 'IV-type':
        res = np.mean(np.multiply(d_minus_m_hat, y_minus_g_hat))/np.mean(np.multiply(d_minus_m_hat, d))
    else:
        assert score == 'partialling out'
        res = scipy.linalg.lstsq(d_minus_m_hat.reshape(-1, 1), y_minus_l_hat)[0]

    return res


def boot_plr(y, d, thetas, ses, all_l_hat, all_m_hat, all_g_hat,
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

        boot_t_stat = boot_plr_single_split(
            thetas[i_rep], y, d, all_l_hat[i_rep], all_m_hat[i_rep], all_g_hat[i_rep], smpls,
            score, ses[i_rep],
            weights, n_rep_boot, apply_cross_fitting)
        all_boot_t_stat.append(boot_t_stat)

    # differently for plr because of n_rep_boot and multiple treatmentsa
    boot_t_stat = np.transpose(np.vstack(all_boot_t_stat))

    return boot_t_stat


def boot_plr_multitreat(y, d, thetas, ses, all_l_hat, all_m_hat, all_g_hat,
                        all_smpls, score, bootstrap, n_rep_boot,
                        n_rep=1, apply_cross_fitting=True):
    n_d = d.shape[1]
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)

        boot_t_stat = np.full((n_d, n_rep_boot), np.nan)
        for i_d in range(n_d):
            boot_t_stat[i_d, :] = boot_plr_single_split(
                thetas[i_rep][i_d], y, d[:, i_d],
                all_l_hat[i_rep][i_d], all_m_hat[i_rep][i_d], all_g_hat[i_rep][i_d],
                smpls, score, ses[i_rep][i_d],
                weights, n_rep_boot, apply_cross_fitting)

        # transpose for shape (n_rep_boot, n_d)
        boot_t_stat = np.transpose(boot_t_stat)
        all_boot_t_stat.append(boot_t_stat)

    # stack repetitions along the last axis
    boot_t_stat = np.stack(all_boot_t_stat, axis=2)

    return boot_t_stat


def boot_plr_single_split(theta, y, d, l_hat, m_hat, g_hat,
                          smpls, score, se, weights, n_rep, apply_cross_fitting):
    y_minus_l_hat, d_minus_m_hat, y_minus_g_hat = compute_plr_residuals(y, d, l_hat, m_hat, g_hat, smpls)

    if apply_cross_fitting:
        if score == 'partialling out':
            J = np.mean(-np.multiply(d_minus_m_hat, d_minus_m_hat))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(d_minus_m_hat, d))
    else:
        test_index = smpls[0][1]
        if score == 'partialling out':
            J = np.mean(-np.multiply(d_minus_m_hat[test_index], d_minus_m_hat[test_index]))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(d_minus_m_hat[test_index], d[test_index]))

    if score == 'partialling out':
        psi = np.multiply(y_minus_l_hat - d_minus_m_hat * theta, d_minus_m_hat)
    else:
        assert score == 'IV-type'
        psi = np.multiply(y_minus_g_hat - d * theta, d_minus_m_hat)

    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, apply_cross_fitting)

    return boot_t_stat


def fit_sensitivity_elements_plr(y, d, all_coef, predictions, score, n_rep):
    n_treat = d.shape[1]
    n_obs = len(y)

    sigma2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    nu2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    psi_sigma2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)
    psi_nu2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)

    for i_rep in range(n_rep):
        for i_treat in range(n_treat):
            d_tilde = d[:, i_treat]
            m_hat = predictions['ml_m'][:, i_rep, i_treat]
            theta = all_coef[i_treat, i_rep]
            if score == 'partialling out':
                l_hat = predictions['ml_l'][:, i_rep, i_treat]
                sigma2_score_element = np.square(y - l_hat - np.multiply(theta, d_tilde-m_hat))
            else:
                assert score == 'IV-type'
                g_hat = predictions['ml_g'][:, i_rep, i_treat]
                sigma2_score_element = np.square(y - g_hat - np.multiply(theta, d_tilde))

            sigma2[0, i_rep, i_treat] = np.mean(sigma2_score_element)
            psi_sigma2[:, i_rep, i_treat] = sigma2_score_element - sigma2[0, i_rep, i_treat]

            nu2[0, i_rep, i_treat] = np.divide(1.0, np.mean(np.square(d_tilde-m_hat)))
            psi_nu2[:, i_rep, i_treat] = nu2[0, i_rep, i_treat] - \
                np.multiply(np.square(d_tilde-m_hat), np.square(nu2[0, i_rep, i_treat]))

    element_dict = {'sigma2': sigma2,
                    'nu2': nu2,
                    'psi_sigma2': psi_sigma2,
                    'psi_nu2': psi_nu2}
    return element_dict
