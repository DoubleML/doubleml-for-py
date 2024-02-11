from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
import numpy as np
import copy

from doubleml.double_ml import DoubleML
from doubleml.double_ml_data import DoubleMLData
# from .double_ml import DoubleML -- not working
from doubleml._utils import (
    _dml_cv_predict, 
    _dml_tune, 
    _get_cond_smpls, 
    _get_cond_smpls_2d,
    _predict_zero_one_propensity,
    _normalize_ipw)
from doubleml._utils_checks  import _check_finite_predictions, _check_is_propensity, _check_zero_one_treatment
#from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions -- also not working
from doubleml.double_ml_score_mixins import LinearScoreMixin


class DoubleMLSS(LinearScoreMixin, DoubleML):
    """Double machine learning for sample selection models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    # TODO add a description for each nuisance function (ml_g is a regression example; ml_m a classification example)
    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.
    
    selection :

    non_random_missing : 

    trim : Trimming rule for discarding observations with (products of) propensity scores that are smaller 
        than trim (to avoid too small denominators in weighting by the inverse of the propensity scores). 
        If selected is 0 (ATE estimation for the total population), observations with products of the treatment and 
        selection propensity scores that are smaller than trim are discarded. If selected is 1 (ATE estimation for 
        the subpopulation with observed outcomes), observations with treatment propensity scores smaller than trim are 
        discarded. 
        Default is ``0.01``.
    
    dtreat : Value of the treatment in the treatment group.
        Default is ``1``.

    dcontrol : Value of the treatment in the control group.
        Default is ``0``.

    n_folds : int
        Number of folds.
        Default is ``3``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'mar'`` or ``'nonignorable_nonresponse'``) specifying the score function.
        Default is ``'mar'`` (missing at random).

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.
    
    normalize_ipw : bool
    Indicates whether the inverse probability weights are normalized.
    Default is ``True``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied.
        Default is ``True``.

    Examples
    --------
    # TODO add an example

    Notes
    -----
    # TODO add an description of the model
    """
    def __init__(self,
                 obj_dml_data,
                 ml_mu,
                 ml_pi,  # propensity score
                 ml_p,   # propensity score
                 selection=0,  # if 0, ATE is estimated, if 1, ATE for selection is estimated
                 trimming_threshold=0.01,
                 treatment = 1,
                 control = 0,
                 n_folds=3,
                 n_rep=1,
                 score='mar',  # TODO implement other scores apart from MAR, this will determine the estimator type
                 dml_procedure='dml2',
                 normalize_ipw=True,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        
        self._trimming_threshold = trimming_threshold
        self._normalize_ipw = normalize_ipw  ## TODO
        self._selection = selection  ## TODO
        self._treatment = treatment
        self._control = control
        self._external_predictions_implemented = True

        self._check_data(self._dml_data)
        self._check_score(self.score)
        
        _ = self._check_learner(ml_mu, 'ml_mu', regressor=True, classifier=False)  # learner must be a regression method
        _ = self._check_learner(ml_pi, 'ml_pi', regressor=False, classifier=True)  # pi is probability
        _ = self._check_learner(ml_p, 'ml_p', regressor=False, classifier=True)  # p is probability
        self._learner = {'ml_mu_d0': clone(ml_mu), 
                         'ml_mu_d1': clone(ml_mu),
                         'ml_pi_d0': clone(ml_pi),
                         'ml_pi_d1': clone(ml_pi),
                         'ml_p_d0': clone(ml_p),
                         'ml_p_d1': clone(ml_p)
        }
        self._predict_method = {'ml_mu_d0': 'predict', 
                                'ml_mu_d1': 'predict',
                                'ml_pi_d0': 'predict_proba', 
                                'ml_pi_d1': 'predict_proba',
                                'ml_p_d0': 'predict_proba',
                                'ml_p_d1': 'predict_proba'
        }

        self._initialize_ml_nuisance_params()

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold
    
    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_mu_d0', 'ml_mu_d1', 
                         'ml_pi_d0', 'ml_pi_d1', 
                         'ml_p_d0', 'ml_p_d1']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                        valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['mar', 'nonignorable_nonresponse']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None and self._score == 'mar':  #TODO: raise warning instead
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'You are estimating the effect under the assumption of data missing at random. \
                             Instrumental variables will not be used in estimation.')
        if obj_dml_data.z_cols is None and self._score == 'nonignorable_nonresponse':
            raise ValueError('Sample selection by nonignorable nonresponse was set but instrumental variable \
                             is None. To estimate treatment effect under nonignorable nonresponse, \
                             specify an instrument for the selection variable.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        x, s = check_X_y(x, self._dml_data.t, force_all_finite=False)
        
        if self._dml_data.z is not None:
            x, z = check_X_y(x, np.ravel(self._dml_data.z), force_all_finite=False)
            dx = np.column_stack((x, d, z))
        else:
            dx = np.column_stack((x, d))
        
        sx = np.column_stack((x, s)) # s and x as controls for mu estimation
        
        # initialize nuisance predictions, targets and models
        mu_hat_d1 = {'models': None,
                 'targets': np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                 'preds': np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
                 }
        mu_hat_d0 = copy.deepcopy(mu_hat_d1)
        pi_hat = copy.deepcopy(mu_hat_d1)
        pi_hat_d1 = copy.deepcopy(mu_hat_d1)
        pi_hat_d0 = copy.deepcopy(mu_hat_d1)
        p_hat_d1 = copy.deepcopy(mu_hat_d1)
        p_hat_d0 = copy.deepcopy(mu_hat_d1)

        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        _, smpls_d0_s1, _, smpls_d1_s1 = _get_cond_smpls_2d(smpls, d, s)  # we only need S = 1
        smpls_s0, smpls_s1 = _get_cond_smpls(smpls, s)

        # initialize models
        fitted_models = {}
        for learner in self.params_names:
            # set nuisance model parameters
            est_params = self._get_params(learner)
            if est_params is not None:
                fitted_models[learner] = [
                    clone(self._learner[learner]).set_params(**est_params[i_fold]) for i_fold in range(self.n_folds)
                ]
            else:
                fitted_models[learner] = [clone(self._learner[learner]) for i_fold in range(self.n_folds)]

        # caculate nuisance functions over different folds
        if self._score == 'nonignorable_nonresponse':

            # create strata for splitting
            strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.t.reshape(-1, 1)

            # TREATMENT
            mu_d1 = {'models': None,
                 'targets': [],
                 'preds': []
                 }
            mu_d0 = copy.deepcopy(mu_d1)
            pi_d1 = copy.deepcopy(mu_d1)
            pi_d0 = copy.deepcopy(mu_d1)
            p_d1 = copy.deepcopy(mu_d1)
            p_d0 = copy.deepcopy(mu_d1)
            s_1 = []
            s_0 = []
            y_1 = []
            y_0 = []
            dtreat = []
            dcontrol = []

            for i_fold in range(self.n_folds):
                # smpls will be based on whether it is treatment or control
                train_inds = smpls[i_fold][0]
                test_inds = smpls[i_fold][1]

                # start nested crossfitting - split training data into two equal parts
                train_inds_1, train_inds_2 = train_test_split(
                    train_inds, test_size=0.5, random_state=42, stratify=s[train_inds]
                )

                s_train_1 = s[train_inds_1]
                dx_train_1 = dx[train_inds_1]

                # preliminary propensity score for selection
                ml_pi_prelim = clone(self._learner['ml_pi_d1'])
                # fit on first part of training set
                ml_pi_prelim.fit(dx_train_1, s_train_1)
                pi_hat['preds'] = _predict_zero_one_propensity(ml_pi_prelim, dx)
                pi_hat['targets'] = s

                pi_hat_d1['preds'] = pi_hat['preds'][test_inds]
                pi_hat_d1['targets'] = s[test_inds]

                # add selection indicator to covariates
                xpi = np.column_stack((x, pi_hat['preds']))
                
                # estimate propensity score p using the second training sample
                xpi_train_2 = xpi[train_inds_2, :]
                d_train_2 = d[train_inds_2]
                xpi_test = xpi[test_inds, :]

                ml_p_d1_prelim = clone(self._learner['ml_p_d1'])
                ml_p_d1_prelim.fit(xpi_train_2, d_train_2)
                
                p_hat_d1['preds'] = _predict_zero_one_propensity(ml_p_d1_prelim, xpi_test)
                p_hat_d1['targets'] = d[test_inds]

                # nuisance mu
                dxpi = np.column_stack((dx, pi_hat['preds']))
                dxpi_s1_train_2 = dxpi[np.intersect1d(np.where(s == 1)[0], train_inds_2), :]
                d_s1_train_2 = d[np.intersect1d(np.where(s == 1)[0], train_inds_2)]
                dxpi_test = dxpi[test_inds, :]

                xpi_s1_train_2 = xpi[np.intersect1d(np.where(s == 1)[0], train_inds_2), :]

                ml_mu_d1_prelim = clone(self._learner['ml_mu_d1'])
                ml_mu_d1_prelim.fit(xpi_s1_train_2, d_s1_train_2)

                mu_hat_d1['preds'] = ml_mu_d1_prelim.predict(xpi_test)
                mu_hat_d1['targets'] = y[test_inds]

                dtreat.append((d == self._treatment)[test_inds])
                s_1.append(s[test_inds])
                y_1.append(y[test_inds])

                mu_d1['preds'].append(mu_hat_d1['preds'])
                pi_d1['preds'].append(pi_hat_d1['preds'])
                p_d1['preds'].append(p_hat_d1['preds'])
                mu_d1['targets'].append(mu_hat_d1['targets'])
                pi_d1['targets'].append(pi_hat_d1['targets'])
                p_d1['targets'].append(p_hat_d1['targets'])
            
            mu_hat_d1['preds'] = np.hstack(mu_d1['preds'])
            pi_hat_d1['preds'] = np.hstack(pi_d1['preds'])
            p_hat_d1['preds'] = np.hstack(p_d1['preds'])
            mu_hat_d1['targets'] = np.hstack(mu_d1['targets'])
            pi_hat_d1['targets'] = np.hstack(pi_d1['targets'])
            p_hat_d1['targets'] = np.hstack(p_d1['targets'])
            s_1 = np.hstack(s_1)
            y_1 = np.hstack(y_1)
            dtreat = np.hstack(dtreat)             
            
            #### CONTROL
            for i_fold in range(self.n_folds):
                train_inds = smpls[i_fold][0]
                test_inds = smpls[i_fold][1]

                train_inds_1, train_inds_2 = train_test_split(
                    train_inds, test_size=0.5, random_state=42, stratify=s[train_inds]
                )

                s_train_1 = s[train_inds_1]
                dx_train_1 = dx[train_inds_1]

                ml_pi_prelim = clone(self._learner['ml_pi_d0'])
                ml_pi_prelim.fit(dx_train_1, s_train_1)
                pi_hat['preds'] = _predict_zero_one_propensity(ml_pi_prelim, dx)
                pi_hat['targets'] = s

                pi_hat_d0['preds'] = pi_hat['preds'][test_inds]
                pi_hat_d0['targets'] = s[test_inds]

                xpi = np.column_stack((x, pi_hat['preds']))
                xpi_train_2 = xpi[train_inds_2, :]
                d_train_2 = d[train_inds_2]
                xpi_test = xpi[test_inds, :]

                ml_p_d0_prelim = clone(self._learner['ml_p_d0'])
                ml_p_d0_prelim.fit(xpi_train_2, d_train_2)
                
                p_hat_d0['preds'] = _predict_zero_one_propensity(ml_p_d0_prelim, xpi_test)
                p_hat_d0['preds'] = 1 - p_hat_d0['preds'] #TODO: should this be here?
                p_hat_d0['targets'] = d[test_inds]

                dxpi = np.column_stack((dx, pi_hat['preds']))
                dxpi_s1_train_2 = dxpi[np.intersect1d(np.where(s == 1)[0], train_inds_2), :]
                d_s1_train_2 = d[np.intersect1d(np.where(s == 1)[0], train_inds_2)]
                dxpi_test = dxpi[test_inds, :]

                xpi_s1_train_2 = xpi[np.intersect1d(np.where(s == 1)[0], train_inds_2), :]

                ml_mu_d0_prelim = clone(self._learner['ml_mu_d0'])
                ml_mu_d0_prelim.fit(xpi_s1_train_2, d_s1_train_2)

                mu_hat_d0['preds'] = ml_mu_d0_prelim.predict(xpi_test)
                mu_hat_d0['targets'] = y[test_inds]
                
            
                dcontrol.append((d == self._control)[test_inds])
                s_0.append(s[test_inds])
                y_0.append(y[test_inds])

                mu_d0['preds'].append(mu_hat_d0['preds'])
                pi_d0['preds'].append(pi_hat_d0['preds'])
                p_d0['preds'].append(p_hat_d0['preds'])
                mu_d0['targets'].append(mu_hat_d0['targets'])
                pi_d0['targets'].append(pi_hat_d0['targets'])
                p_d0['targets'].append(p_hat_d0['targets'])
            
            mu_hat_d0['preds'] = np.hstack(mu_d0['preds'])
            pi_hat_d0['preds'] = np.hstack(pi_d0['preds'])
            p_hat_d0['preds'] = np.hstack(p_d0['preds'])
            mu_hat_d0['targets'] = np.hstack(mu_d0['targets'])
            pi_hat_d0['targets'] = np.hstack(pi_d0['targets'])
            p_hat_d0['targets'] = np.hstack(p_d0['targets'])

            pi_hat_d1['targets'] = pi_hat_d1['targets'].astype(float)
            pi_hat_d1['targets'][d != self._treatment] = np.nan
            mu_hat_d1['targets'] = mu_hat_d1['targets'].astype(float)
            mu_hat_d1['targets'][d != self._treatment] = np.nan
            p_hat_d1['targets'] = p_hat_d1['targets'].astype(float)
            p_hat_d1['targets'][d != self._treatment] = np.nan

            pi_hat_d0['targets'] = pi_hat_d0['targets'].astype(float)
            pi_hat_d0['targets'][d != self._control] = np.nan
            mu_hat_d0['targets'] = mu_hat_d0['targets'].astype(float)
            mu_hat_d0['targets'][d != self._control] = np.nan
            p_hat_d0['targets'] = p_hat_d0['targets'].astype(float)
            p_hat_d0['targets'][d != self._control] = np.nan

            s_0 = np.hstack(s_0)
            y_0 = np.hstack(y_0)
            dcontrol = np.hstack(dcontrol)
        
        else:  # missing at random (MAR)
            pi_hat_d1 = _dml_cv_predict(self._learner['ml_pi_d1'], dx, s, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_pi_d1'), method=self._predict_method['ml_pi_d1'],
                                return_models=return_models)
            pi_hat_d1['targets'] = pi_hat_d1['targets'].astype(float)
            pi_hat_d1['targets'][d != self._treatment] = np.nan
            _check_finite_predictions(pi_hat_d1['preds'], self._learner['ml_pi_d1'], 'ml_pi_d1', smpls)


            pi_hat_d0 = _dml_cv_predict(self._learner['ml_pi_d0'], dx, s, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_pi_d0'), method=self._predict_method['ml_pi_d0'],
                                    return_models=return_models)
            pi_hat_d0['targets'] = pi_hat_d0['targets'].astype(float)
            pi_hat_d0['targets'][d != self._control] = np.nan
            _check_finite_predictions(pi_hat_d0['preds'], self._learner['ml_pi_d0'], 'ml_pi_d0', smpls)
            _check_is_propensity(pi_hat_d1['preds'], self._learner['ml_pi_d1'], 'ml_pi_d1', smpls, eps=1e-12)
            _check_is_propensity(pi_hat_d0['preds'], self._learner['ml_pi_d0'], 'ml_pi_d0', smpls, eps=1e-12)

            # propensity score p
            p_hat_d1 = _dml_cv_predict(self._learner['ml_p_d1'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_p_d1'), method=self._predict_method['ml_p_d1'],
                                    return_models=return_models)
            p_hat_d1['targets'] = p_hat_d1['targets'].astype(float)
            p_hat_d1['targets'][d != self._treatment] = np.nan
            _check_finite_predictions(p_hat_d1['preds'], self._learner['ml_p_d1'], 'ml_p_d1', smpls)
            

            p_hat_d0 = _dml_cv_predict(self._learner['ml_p_d0'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_p_d0'), method=self._predict_method['ml_p_d0'],
                                    return_models=return_models)
            p_hat_d0['preds'] = 1 - p_hat_d0['preds']
            p_hat_d0['targets'][d != self._control] = np.nan
            _check_finite_predictions(p_hat_d0['preds'], self._learner['ml_p_d0'], 'ml_p_d0', smpls)
            _check_is_propensity(p_hat_d1['preds'], self._learner['ml_p_d1'], 'ml_p_d1', smpls, eps=1e-12)
            _check_is_propensity(p_hat_d0['preds'], self._learner['ml_p_d0'], 'ml_p_d0', smpls, eps=1e-12)

            # nuisance mu
            mu_hat_d1 = _dml_cv_predict(self._learner['ml_mu_d1'], x, y, smpls=smpls_s1, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_mu_d1'), method=self._predict_method['ml_mu_d1'],
                                    return_models=return_models)
            mu_hat_d1['targets'] = mu_hat_d1['targets'].astype(float)
            mu_hat_d1['targets'][d != self._treatment] = np.nan
            _check_finite_predictions(mu_hat_d1['preds'], self._learner['ml_mu_d1'], 'ml_mu_d1', smpls)

            mu_hat_d0 = _dml_cv_predict(self._learner['ml_mu_d0'], x, y, smpls=smpls_s1, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_mu_d0'), method=self._predict_method['ml_mu_d0'],
                                    return_models=return_models)
            mu_hat_d0['targets'] = mu_hat_d0['targets'].astype(float)
            mu_hat_d0['targets'][d != self._control] = np.nan
            _check_finite_predictions(mu_hat_d0['preds'], self._learner['ml_mu_d0'], 'ml_mu_d0', smpls)

            # treatment indicator
            dtreat = d == self._treatment
            dcontrol = d == self._control

            s_0 = s
            s_1 = s
            y_0 = y
            y_1 = y
    
            ## Trimming - done differently in Bia, Huber and Laffers than in DoubleML - dropping observations
            if not self._selection: #TODO: move outside the MAR assumption AND CHECK IF IT IS REALLY CORRECT, DIMENSIONS SHOULD NOT ADD UP!
                mask_d1 = np.multiply(pi_hat_d1['preds'], p_hat_d1['preds']) >= self._trimming_threshold
                mask_d0 = np.multiply(pi_hat_d0['preds'], p_hat_d0['preds']) >= self._trimming_threshold
                
                mu_hat_d1['preds'] = mu_hat_d1['preds'][mask_d1]
                mu_hat_d1['targets'] = mu_hat_d1['targets'][mask_d1]
                mu_hat_d0['preds'] = mu_hat_d0['preds'][mask_d0]
                mu_hat_d0['targets'] = mu_hat_d0['targets'][mask_d0]

                pi_hat_d1['preds'] = pi_hat_d1['preds'][mask_d1]
                pi_hat_d1['targets'] = pi_hat_d1['targets'][mask_d1]
                pi_hat_d0['preds'] = pi_hat_d0['preds'][mask_d0]
                pi_hat_d0['targets'] = pi_hat_d0['targets'][mask_d0]

                p_hat_d1['preds'] = p_hat_d1['preds'][mask_d1]
                p_hat_d1['targets'] = p_hat_d1['targets'][mask_d1]
                p_hat_d0['preds'] = p_hat_d0['preds'][mask_d0]
                p_hat_d0['targets'] = p_hat_d0['targets'][mask_d0]

                s_0 = s_0[mask_d0]
                s_1 = s_1[mask_d1]

                y_0 = y_0[mask_d0]
                y_1 = y_1[mask_d1] ##TODO: have to change, if some observations are dropped, it will not work
        
        psi_a, psi_b = self._score_elements(dtreat, dcontrol, mu_hat_d1['preds'],
                                            mu_hat_d0['preds'], pi_hat_d1['preds'],
                                            pi_hat_d0['preds'],
                                            p_hat_d1['preds'], p_hat_d0['preds'], 
                                            s_0, s_1, y_0, y_1) 
        
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        
        preds = {'predictions': {'ml_mu_d0': mu_hat_d0['preds'],
                                 'ml_mu_d1': mu_hat_d1['preds'], 
                                 'ml_pi_d0': pi_hat_d0['preds'],
                                 'ml_pi_d1': pi_hat_d1['preds'],
                                 'ml_p_d0': p_hat_d0['preds'],
                                 'ml_p_d1': p_hat_d1['preds']},
                'targets': {'ml_mu_d0': mu_hat_d0['targets'],
                            'ml_mu_d1': mu_hat_d1['targets'],
                            'ml_pi_d0': pi_hat_d0['targets'],
                            'ml_pi_d1': pi_hat_d1['targets'],
                            'ml_p_d0': p_hat_d0['targets'],
                            'ml_p_d1': p_hat_d1['targets']},
                'models': {'ml_mu_d0': mu_hat_d0['models'],
                            'ml_mu_d1': mu_hat_d1['models'],
                            'ml_pi_d0': pi_hat_d0['models'],
                            'ml_pi_d1': pi_hat_d1['models'],
                            'ml_p_d0': p_hat_d0['models'],
                            'ml_p_d0': p_hat_d1['models']}
                }

        return psi_elements, preds
    

    ## TODO : DIMENSTIONS DO NOT ADD UP FOR SUBSAMPLES
    def _score_elements(self, dtreat, dcontrol, mu_d1, mu_d0, 
                        pi_d1, pi_d0, p_d1, p_d0, s_0, s_1, y_0, y_1):
        # psi_a
        psi_a = -1

        # psi_b
        psi_b1 = (dtreat * s_1 * (y_1 - mu_d1)) / (p_d1 * pi_d1) + mu_d1
        psi_b0 = (dcontrol * s_0 * (y_0 - mu_d0)) / (p_d0 * pi_d0) + mu_d0

        psi_b = psi_b1 - psi_b0

        return psi_a, psi_b


    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        x, s = check_X_y(x, self._dml_data.t,
                          force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_mu': None,
                               'ml_pi': None,
                               'ml_p': None}

        # nuisance training sets conditional on d
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        
        # hyperparameter tuning for ML 
        mu_d0_tune_res = _dml_tune(y, x, train_inds_d0,
                               self._learner['ml_mu'], param_grids['ml_mu'], scoring_methods['ml_mu'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        mu_d1_tune_res = _dml_tune(y, x, train_inds_d1,
                               self._learner['ml_mu'], param_grids['ml_mu'], scoring_methods['ml_mu'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        pi_d0_tune_res = _dml_tune(y, x, train_inds_d0,
                               self._learner['ml_pi'], param_grids['ml_pi'], scoring_methods['ml_pi'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        pi_d1_tune_res = _dml_tune(y, x, train_inds_d1,
                               self._learner['ml_pi'], param_grids['ml_pi'], scoring_methods['ml_pi'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        p_d0_tune_res = _dml_tune(y, x, train_inds_d0,
                               self._learner['ml_p'], param_grids['ml_p'], scoring_methods['ml_p'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        p_d1_tune_res = _dml_tune(y, x, train_inds_d1,
                               self._learner['ml_p'], param_grids['ml_p'], scoring_methods['ml_p'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        mu_d0_best_params = [xx.best_params_ for xx in mu_d0_tune_res]
        mu_d1_best_params = [xx.best_params_ for xx in mu_d1_tune_res]
        pi_d0_best_params = [xx.best_params_ for xx in pi_d0_tune_res]
        pi_d1_best_params = [xx.best_params_ for xx in pi_d1_tune_res]
        p_d0_best_params = [xx.best_params_ for xx in pi_d0_tune_res]
        p_d1_best_params = [xx.best_params_ for xx in pi_d1_tune_res]

        params = {'ml_mu_d0': mu_d0_best_params,
                  'ml_mu_d1': mu_d1_best_params,
                  'ml_pi_d0': pi_d0_best_params,
                  'ml_pi_d1': pi_d1_best_params,
                  'ml_p_d0': p_d0_best_params,
                  'ml_p_d1': p_d1_best_params}

        tune_res = {'mu_d0_tune': mu_d0_tune_res,
                    'mu_d1_tune': mu_d1_tune_res,
                    'pi_d0_tune': pi_d0_tune_res,
                    'pi_d1_tune': pi_d1_tune_res,
                    'p_d0_tune': p_d0_tune_res,
                    'p_d1_tune': p_d1_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
    

    def _sensitivity_element_est(self, preds):
        y = self._dml_data.y
        d = self._dml_data.d
        s = self._dml_data.t  ## again, DoubleML does not have a specified column for selection, using t instead

        mu_hat_d1 = preds['predictions']['ml_mu_d1']
        mu_hat_d0 = preds['predictions']['ml_mu_d0']
        pi_hat_d1 = preds['predictions']['ml_pi_d1']
        pi_hat_d0 = preds['predictions']['ml_pi_d0']
        p_hat_d1 = preds['predictions']['ml_p_d1']
        p_hat_d0 = preds['predictions']['ml_p_d0']