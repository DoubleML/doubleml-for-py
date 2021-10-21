import numpy as np


class LinearScoreMixin:
    _score_type = 'linear'

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']

    @staticmethod
    def _compute_score(psi_elements, coef, inds=None):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        if inds is not None:
            psi_a = psi_a[inds]
            psi_b = psi_b[inds]
        psi = psi_a * coef + psi_b
        return psi

    @staticmethod
    def _compute_score_deriv(psi_elements, coef, inds=None):
        psi_a = psi_elements['psi_a']
        if inds is not None:
            psi_a = psi_a[inds]
        return psi_a

    def _est_coef(self, psi_elements, inds=None):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        if inds is not None:
            psi_a = psi_a[inds]
            psi_b = psi_b[inds]

        coef = -np.mean(psi_b) / np.mean(psi_a)

        return coef

    @staticmethod
    def _est_coef_cluster_data(psi_elements, dml_procedure, smpls, smpls_cluster):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        dml1_coefs = None

        if dml_procedure == 'dml1':
            # note that in the dml1 case we could also simply apply the standard function without cluster adjustment
            dml1_coefs = np.zeros(len(smpls))
            for i_fold, (_, test_index) in enumerate(smpls):
                test_cluster_inds = smpls_cluster[i_fold][1]
                scaling_factor = 1./np.prod(np.array([len(inds) for inds in test_cluster_inds]))
                dml1_coefs[i_fold] = - (scaling_factor * np.sum(psi_b[test_index])) / \
                    (scaling_factor * np.sum(psi_a[test_index]))
            coef = np.mean(dml1_coefs)
        else:
            assert dml_procedure == 'dml2'
            # See Chiang et al. (2021) Algorithm 1
            psi_a_subsample_mean = 0.
            psi_b_subsample_mean = 0.
            for i_fold, (_, test_index) in enumerate(smpls):
                test_cluster_inds = smpls_cluster[i_fold][1]
                scaling_factor = 1./np.prod(np.array([len(inds) for inds in test_cluster_inds]))
                psi_a_subsample_mean += scaling_factor * np.sum(psi_a[test_index])
                psi_b_subsample_mean += scaling_factor * np.sum(psi_b[test_index])
            coef = -psi_b_subsample_mean / psi_a_subsample_mean

        return coef, dml1_coefs
