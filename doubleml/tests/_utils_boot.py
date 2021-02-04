import numpy as np


def draw_weights(method, n_rep_boot, n_obs):
    if method == 'Bayes':
        weights = np.random.exponential(scale=1.0, size=(n_rep_boot, n_obs)) - 1.
    elif method == 'normal':
        weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
    else:
        assert method == 'wild'
        xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2

    return weights


def boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure, apply_cross_fitting=True):
    n_folds = len(smpls)
    boot_theta = np.zeros(n_rep)
    boot_t_stat = np.zeros(n_rep)
    for i_rep in range(n_rep):
        this_weights = weights[i_rep, :]
        if apply_cross_fitting:
            if dml_procedure == 'dml1':
                this_boot_theta = np.zeros(n_folds)
                this_boot_t_stat = np.zeros(n_folds)
                for idx, (_, test_index) in enumerate(smpls):
                    this_boot_theta[idx] = np.mean(np.multiply(this_weights[test_index],
                                                               psi[test_index] / J[idx]))
                    this_boot_t_stat[idx] = np.mean(np.multiply(np.divide(this_weights[test_index], se),
                                                                psi[test_index] / J[idx]))
                boot_theta[i_rep] = np.mean(this_boot_theta)
                boot_t_stat[i_rep] = np.mean(this_boot_t_stat)
            elif dml_procedure == 'dml2':
                boot_theta[i_rep] = np.mean(np.multiply(this_weights,
                                                        psi / J))
                boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(this_weights, se),
                                                         psi / J))
        else:
            test_index = smpls[0][1]
            boot_theta[i_rep] = np.mean(np.multiply(this_weights,
                                                    psi[test_index] / J[0]))
            boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(this_weights, se),
                                                     psi[test_index] / J[0]))

    return boot_theta, boot_t_stat
