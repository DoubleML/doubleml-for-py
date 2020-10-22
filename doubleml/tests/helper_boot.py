import numpy as np


def boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure, apply_cross_fitting=True):

    if apply_cross_fitting:
        n_obs = len(psi)
    else:
        test_index = smpls[0][1]
        n_obs = len(test_index)
    n_folds = len(smpls)

    boot_theta = np.zeros(n_rep)
    boot_t_stat = np.zeros(n_rep)
    if bootstrap == 'wild':
        # if method wild for unit test comparability draw all rv at one step
        xx_sample = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
        yy_sample = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))

    for i_rep in range(n_rep):
        if bootstrap == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=n_obs) - 1.
        elif bootstrap == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=n_obs)
        elif bootstrap == 'wild':
            xx = xx_sample[i_rep, :]
            yy = yy_sample[i_rep, :]
            weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
        else:
            raise ValueError('invalid bootstrap method')

        if apply_cross_fitting:
            if dml_procedure == 'dml1':
                this_boot_theta = np.zeros(n_folds)
                this_boot_t_stat = np.zeros(n_folds)
                for idx, (train_index, test_index) in enumerate(smpls):
                    this_boot_theta[idx] = np.mean(np.multiply(weights[test_index],
                                                               psi[test_index] / J[idx]))
                    this_boot_t_stat[idx] = np.mean(np.multiply(np.divide(weights[test_index], se),
                                                                psi[test_index] / J[idx]))
                boot_theta[i_rep] = np.mean(this_boot_theta)
                boot_t_stat[i_rep] = np.mean(this_boot_t_stat)
            elif dml_procedure == 'dml2':
                boot_theta[i_rep] = np.mean(np.multiply(weights,
                                                        psi / J))
                boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(weights, se),
                                                         psi / J))
        else:
            boot_theta[i_rep] = np.mean(np.multiply(weights,
                                                    psi[test_index] / J[0]))
            boot_t_stat[i_rep] = np.mean(np.multiply(np.divide(weights, se),
                                                     psi[test_index] / J[0]))

    return boot_theta, boot_t_stat
