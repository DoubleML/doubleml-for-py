import numpy as np
from scipy.optimize import root_scalar
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold, train_test_split

from .double_ml import DoubleML
from .double_ml_score_mixins import NonLinearScoreMixin
from ._utils import _dml_cv_predict, _trimm, _predict_zero_one_propensity, _check_zero_one_treatment, _check_score,\
    _check_trimming, _check_quantile, _check_treatment, _get_bracket_guess, _default_kde, _normalize_ipw
from .double_ml_data import DoubleMLData
from ._utils_resampling import DoubleMLResampling


class DoubleMLLPQ(NonLinearScoreMixin, DoubleML):
    """Double machine learning for local potential quantiles

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_pi : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the propensity nuisance functions.

    treatment : int
        Binary treatment indicator. Has to be either ``0`` or ``1``. Determines the potential outcome to be considered.
        Default is ``1``.

    quantile : float
        Quantile of the potential outcome. Has to be between ``0`` and ``1``.
        Default is ``0.5``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'PQ'`` is the only choice) specifying the score function
        for potential quantiles.
        Default is ``'PQ'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    kde : callable or None
        A callable object / function with signature ``deriv = kde(u, weights)`` for weighted kernel density estimation.
        Here ``deriv`` should evaluate the density in ``0``.
        Default is ``'None'``, which uses :py:class:`statsmodels.nonparametric.kde.KDEUnivariate` with a
        gaussian kernel and silverman for bandwidth determination.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied(``True`` is the only choice).
        Default is ``True``.
    """

    def __init__(self,
                 obj_dml_data,
                 ml_pi,
                 treatment=1,
                 quantile=0.5,
                 n_folds=5,
                 n_rep=1,
                 score='LPQ',
                 dml_procedure='dml2',
                 normalize_ipw=True,
                 kde=None,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._quantile = quantile
        self._treatment = treatment
        if kde is None:
            self._kde = _default_kde
        else:
            if not callable(kde):
                raise TypeError('kde should be either a callable or None. '
                                '%r was passed.' % kde)
            self._kde = kde
        self._normalize_ipw = normalize_ipw

        if self._is_cluster_data:
            raise NotImplementedError('Estimation with clustering not implemented.')
        self._check_data(self._dml_data)

        valid_score = ['LPQ']
        _check_score(self.score, valid_score)
        _check_quantile(self.quantile)
        _check_treatment(self.treatment)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')

        # initialize starting values and bounds
        y_treat = self._dml_data.y[self._dml_data.d == self.treatment]
        self._coef_bounds = (y_treat.min(), y_treat.max())
        self._coef_start_val = np.quantile(y_treat, self.quantile)

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        _ = self._check_learner(ml_pi, 'ml_pi', regressor=False, classifier=True)
        self._learner = {'ml_pi_z': clone(ml_pi),
                         'ml_pi_du_z0': clone(ml_pi), 'ml_pi_du_z1': clone(ml_pi),
                         'ml_pi_d_z0': clone(ml_pi), 'ml_pi_d_z1': clone(ml_pi)}
        self._predict_method = {'ml_pi_z': 'predict_proba',
                                'ml_pi_du_z0': 'predict_proba', 'ml_pi_du_z1': 'predict_proba',
                                'ml_pi_d_z0': 'predict_proba', 'ml_pi_d_z1': 'predict_proba'}

        self._initialize_ml_nuisance_params()

        if draw_sample_splitting:
            strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.z.reshape(-1, 1)
            obj_dml_resampling = DoubleMLResampling(n_folds=self.n_folds,
                                                    n_rep=self.n_rep,
                                                    n_obs=self._dml_data.n_obs,
                                                    apply_cross_fitting=self.apply_cross_fitting,
                                                    stratify=strata)
            self._smpls = obj_dml_resampling.split_samples()

    @property
    def quantile(self):
        """
        Quantile for potential outcome.
        """
        return self._quantile

    @property
    def treatment(self):
        """
        Treatment indicator for potential outcome.
        """
        return self._treatment

    @property
    def kde(self):
        """
        The kernel density estimation of the derivative.
        """
        return self._kde

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

    @property
    def _score_element_names(self):
        return ['ind_d', 'pi_z', 'pi_du_z0', 'pi_du_z1', 'y', 'z', 'comp_prob']

    def _compute_ipw_score(self, theta, d, y, prop, z, comp_prob):
        sign = 2 * self.treatment - 1.0
        weights = sign * (z / prop - (1 - z) / (1 - prop)) / comp_prob
        u = (d == self._treatment) * (y <= theta)
        v = -1. * self.quantile
        score = weights * u + v
        return score

    def _compute_score(self, psi_elements, coef, inds=None):
        sign = 2 * self.treatment - 1.0
        ind_d = psi_elements['ind_d']
        pi_z = psi_elements['pi_z']
        pi_du_z0 = psi_elements['pi_du_z0']
        pi_du_z1 = psi_elements['pi_du_z1']
        y = psi_elements['y']
        z = psi_elements['z']
        comp_prob = psi_elements['comp_prob']

        if inds is not None:
            ind_d = psi_elements['ind_d'][inds]
            pi_z = psi_elements['pi_z']
            pi_du_z0 = psi_elements['pi_du_z0'][inds]
            pi_du_z1 = psi_elements['pi_du_z1'][inds]
            y = psi_elements['y'][inds]
            z = psi_elements['z'][inds]

        score1 = pi_du_z1 - pi_du_z0
        score2 = (z / pi_z) * (ind_d * (y <= coef) - pi_du_z1)
        score3 = (1 - z) / (1 - pi_z) * (ind_d * (y <= coef) - pi_du_z0)
        score = sign * (score1 + score2 - score3) / comp_prob - self.quantile
        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        sign = 2 * self.treatment - 1.0
        ind_d = psi_elements['ind_d']
        y = psi_elements['y']
        pi_z = psi_elements['pi_z']
        z = psi_elements['z']
        comp_prob = psi_elements['comp_prob']

        if inds is not None:
            ind_d = psi_elements['ind_d'][inds]
            y = psi_elements['y'][inds]
            pi_z = psi_elements['pi_z'][inds]
            z = psi_elements['z'][inds]

        score_weights = sign * ((z / pi_z) - (1 - z) / (1 - pi_z)) * ind_d / comp_prob
        u = (y - coef).reshape(-1, 1)
        deriv = self.kde(u, score_weights)

        return deriv

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in ['ml_pi_z', 'ml_pi_du_z0', 'ml_pi_du_z1',
                                        'ml_pi_d_z0', 'ml_pi_d_z1']}

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z),
                         force_all_finite=False)

        # create strata for splitting
        strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.z.reshape(-1, 1)

        # initialize nuisance predictions
        pi_z_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        pi_d_z0_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        pi_d_z1_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        pi_du_z0_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        pi_du_z1_hat = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)

        ipw_vec = np.full(shape=self.n_folds, fill_value=np.nan)
        # caculate nuisance functions over different folds
        for i_fold in range(self.n_folds):
            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                          random_state=42, stratify=strata[train_inds])
            smpls_prelim = [(train, test) for train, test in
                            StratifiedKFold(n_splits=self.n_folds).split(X=train_inds_1, y=strata[train_inds_1])]

            d_train_1 = d[train_inds_1]
            y_train_1 = y[train_inds_1]
            x_train_1 = x[train_inds_1, :]
            z_train_1 = z[train_inds_1]

            # preliminary propensity for z
            pi_z_hat_prelim = _dml_cv_predict(self._learner['ml_pi_z'], x_train_1, z_train_1,
                                              method='predict_proba', smpls=smpls_prelim)['preds']
            pi_z_hat_prelim = _trimm(pi_z_hat_prelim, self.trimming_rule, self.trimming_threshold)

            if self._normalize_ipw:
                pi_z_hat_prelim = _normalize_ipw(pi_z_hat_prelim, z_train_1)

            # todo add extra fold loop
            # propensity for d == 1 cond. on z == 0 (training set 1)
            x_z0_train_1 = x_train_1[z_train_1 == 0, :]
            d_z0_train_1 = d_train_1[z_train_1 == 0]
            self._learner['ml_pi_d_z0'].fit(x_z0_train_1, d_z0_train_1)
            pi_d_z0_hat_prelim = _predict_zero_one_propensity(self._learner['ml_pi_d_z0'], x_train_1)
            pi_d_z0_hat_prelim = _trimm(pi_d_z0_hat_prelim, self.trimming_rule, self.trimming_threshold)

            # propensity for d == 1 cond. on z == 1 (training set 1)
            x_z1_train_1 = x_train_1[z_train_1 == 1, :]
            d_z1_train_1 = d_train_1[z_train_1 == 1]
            self._learner['ml_pi_d_z1'].fit(x_z1_train_1, d_z1_train_1)
            pi_d_z1_hat_prelim = _predict_zero_one_propensity(self._learner['ml_pi_d_z1'], x_train_1)
            pi_d_z1_hat_prelim = _trimm(pi_d_z1_hat_prelim, self.trimming_rule, self.trimming_threshold)

            # preliminary estimate of theta_2_aux
            comp_prob_prelim = np.mean(pi_d_z1_hat_prelim - pi_d_z0_hat_prelim
                                       + z_train_1 / pi_z_hat_prelim * (d_train_1 - pi_d_z1_hat_prelim)
                                       - (1 - z_train_1) / (1 - pi_z_hat_prelim) * (d_train_1 - pi_d_z0_hat_prelim))

            # preliminary ipw estimate
            def ipw_score(theta):
                res = np.mean(self._compute_ipw_score(theta, d_train_1, y_train_1, pi_z_hat_prelim,
                                                      z_train_1, comp_prob_prelim))
                return res

            _, bracket_guess = _get_bracket_guess(ipw_score, self._coef_start_val, self._coef_bounds)
            root_res = root_scalar(ipw_score,
                                   bracket=bracket_guess,
                                   method='brentq')
            ipw_est = root_res.root
            ipw_vec[i_fold] = ipw_est

            # use the preliminary estimates to fit the nuisance parameters on train_2
            d_train_2 = d[train_inds_2]
            y_train_2 = y[train_inds_2]
            x_train_2 = x[train_inds_2, :]
            z_train_2 = z[train_inds_2]

            # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 0
            x_z0_train_2 = x_train_2[z_train_2 == 0, :]
            du_z0_train_2 = (d_train_2[z_train_2 == 0] == self._treatment) * (y_train_2[z_train_2 == 0] <= ipw_est)
            self._learner['ml_pi_du_z0'].fit(x_z0_train_2, du_z0_train_2)
            pi_du_z0_hat[test_inds] = _predict_zero_one_propensity(self._learner['ml_pi_du_z0'], x[test_inds, :])

            # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 1
            x_z1_train_2 = x_train_2[z_train_2 == 1, :]
            du_z1_train_2 = (d_train_2[z_train_2 == 1] == self._treatment) * (y_train_2[z_train_2 == 1] <= ipw_est)
            self._learner['ml_pi_du_z1'].fit(x_z1_train_2, du_z1_train_2)
            pi_du_z1_hat[test_inds] = _predict_zero_one_propensity(self._learner['ml_pi_du_z1'], x[test_inds, :])

            # refit nuisance elements for the local potential quantile
            z_train = z[train_inds]
            x_train = x[train_inds]
            d_train = d[train_inds]

            # refit propensity for z (whole training set)
            self._learner['ml_pi_z'].fit(x_train, z_train)
            pi_z_hat[test_inds] = _predict_zero_one_propensity(self._learner['ml_pi_z'], x[test_inds, :])

            # refit propensity for d == 1 cond. on z == 0 (whole training set)
            x_z0_train = x_train[z_train == 0, :]
            d_z0_train = d_train[z_train == 0]
            self._learner['ml_pi_d_z0'].fit(x_z0_train, d_z0_train)
            pi_d_z0_hat[test_inds] = _predict_zero_one_propensity(self._learner['ml_pi_d_z0'], x[test_inds, :])

            # propensity for d == 1 cond. on z == 1 (whole training set)
            x_z1_train = x_train[z_train == 1, :]
            d_z1_train = d_train[z_train == 1]
            self._learner['ml_pi_d_z1'].fit(x_z1_train, d_z1_train)
            pi_d_z1_hat[test_inds] = _predict_zero_one_propensity(self._learner['ml_pi_d_z1'], x[test_inds, :])

        # clip propensities
        pi_z_hat = _trimm(pi_z_hat, self.trimming_rule, self.trimming_threshold)
        pi_d_z0_hat = _trimm(pi_d_z0_hat, self.trimming_rule, self.trimming_threshold)
        pi_d_z1_hat = _trimm(pi_d_z1_hat, self.trimming_rule, self.trimming_threshold)
        pi_du_z0_hat = _trimm(pi_du_z0_hat, self.trimming_rule, self.trimming_threshold)
        pi_du_z1_hat = _trimm(pi_du_z1_hat, self.trimming_rule, self.trimming_threshold)

        if self._normalize_ipw:
            if self.dml_procedure == 'dml1':
                for _, test_index in smpls:
                    pi_z_hat[test_index] = _normalize_ipw(pi_z_hat[test_index], z[test_index])
            else:
                pi_z_hat = _normalize_ipw(pi_z_hat, z)

        # estimate final nuisance parameter
        comp_prob_hat = np.mean(pi_d_z1_hat - pi_d_z0_hat
                                + z / pi_z_hat * (d - pi_d_z1_hat)
                                - (1 - z) / (1 - pi_z_hat) * (d - pi_d_z0_hat))

        # readjust start value for minimization
        self._coef_start_val = np.mean(ipw_vec)

        psi_elements = {'ind_d': d == self._treatment, 'pi_z': pi_z_hat,
                        'pi_du_z0': pi_du_z0_hat, 'pi_du_z1': pi_du_z1_hat,
                        'y': y, 'z': z, 'comp_prob': comp_prob_hat}
        preds = {'ml_pi_z': pi_z_hat,
                 'ml_pi_d_z0': pi_d_z0_hat, 'ml_pi_d_z1': pi_d_z1_hat,
                 'ml_pi_du_z0': pi_du_z0_hat, 'ml_pi_du_z1': pi_du_z1_hat}
        return psi_elements, preds

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        raise NotImplementedError('Nuisance tuning not implemented for potential quantiles.')

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        _check_zero_one_treatment(self)
        one_instr = (obj_dml_data.n_instr == 1)
        err_msg = ('Incompatible data. '
                   'To fit an LPQ model with DML '
                   'exactly one binary variable with values 0 and 1 '
                   'needs to be specified as instrumental variable.')
        if one_instr:
            binary_instr = (type_of_target(obj_dml_data.z) == 'binary')
            zero_one_instr = np.all((np.power(obj_dml_data.z, 2) - obj_dml_data.z) == 0)
            if not (one_instr & binary_instr & zero_one_instr):
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        return
