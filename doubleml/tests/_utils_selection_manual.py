import numpy as np
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, tune_grid_search


def fit_selection(y, x, d, z, s,
               learner_mu, learner_pi, learner_p, 
               all_smpls, dml_procedure, score, 
               dtreatment, dcontrol,
               trimming_rule='truncate',
               trimming_threshold=1e-2,
               normalize_ipw=True,
               n_rep=1, 
               mu_d0_params=None, mu_d1_params=None,
               pi_d0_params=None, pi_d1_params=None, 
               p_d0_params=None, p_d1_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)


def fit_nuisance_selection(y, x, d, z, s,
               learner_mu, learner_pi, learner_p, 
               smpls, dml_procedure, score, 
               dtreatment, dcontrol,
               trimming_rule='truncate',
               trimming_threshold=1e-2,
               normalize_ipw=True,
               mu_d0_params=None, mu_d1_params=None,
               pi_d0_params=None, pi_d1_params=None, 
               p_d0_params=None, p_d1_params=None):
    pass