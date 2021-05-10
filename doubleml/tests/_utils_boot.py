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


def boot_manual(psi, J, smpls, se, weights, n_rep, apply_cross_fitting=True):
    boot_theta = np.zeros(n_rep)
    boot_t_stat = np.zeros(n_rep)
    for i_rep in range(n_rep):
        this_weights = weights[i_rep, :]
        if apply_cross_fitting:
            boot_theta[i_rep] = np.mean(np.multiply(this_weights,
                                                    psi / J))
            boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(this_weights, se),
                                                     psi / J))
        else:
            test_index = smpls[0][1]
            boot_theta[i_rep] = np.mean(np.multiply(this_weights,
                                                    psi[test_index] / J))
            boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(this_weights, se),
                                                     psi[test_index] / J))

    return boot_theta, boot_t_stat
