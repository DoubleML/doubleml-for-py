from sklearn.utils import check_X_y
import numpy as np
import copy

from doubleml.double_ml import DoubleML
from doubleml.double_ml_data import DoubleMLData
# from .double_ml import DoubleML -- not working
from doubleml._utils import _dml_cv_predict, _dml_tune, _get_cond_smpls
from doubleml._utils_checks  import _check_finite_predictions
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
        A str (``'mar_score'``) specifying the score function.
        Default is ``'mar_score'``.

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
                 ml_mu,  # default should be lasso
                 ml_pi,  # propensity score
                 ml_p,   # propensity score
                 selection=0,  # if 0, ATE is estimated, if 1, ATE for selection is estimated
                 trimming_threshold = 0.01, 
                 treatment = 1,
                 control = 0,
                 n_folds=3,
                 n_rep=1,
                 score='mar_score',  # TODO implement other scores apart from MAR, this will determine the estimator type
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
        
        self._normalize_ipw = normalize_ipw  ## TODO
        self._selection = selection  ## TODO
        self._treatment = treatment
        self._control = control

        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_mu, 'ml_mu', regressor=True, classifier=False)  # learner must be a regression method
        _ = self._check_learner(ml_pi, 'ml_pi', regressor=False, classifier=True)  # learner must be a regression method, pi is probability
        _ = self._check_learner(ml_p, 'ml_p', regressor=False, classifier=True)  # learner must be a regression method, p is probability
        self._learner = {'ml_mu': ml_mu, 'ml_pi': ml_pi, 'ml_p': ml_p}
        self._predict_method = {'ml_mu': 'predict', 
                                'ml_pi': 'predict_proba', 
                                'ml_p': 'predict_proba'}
        self._initialize_ml_nuisance_params()
    
    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_mu', 'ml_pi', 'ml_p']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                        valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['mar_score']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):  ## TODO add checks for missingness, treatment etc.
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLSS.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(self._dml_data.x, self._dml_data.d,
                         force_all_finite=False)
        x, s = check_X_y(self._dml_data.x, self._dml_data.t,
                          force_all_finite=False)
        
        dx = np.column_stack((x, d))  # use d among control variables for pi estimation
        dsx = np.column_stack((dx, s))
        
        # initialize nuisance predictions, targets and models
        mu_hat = {'models': None,
                 'targets': np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                 'preds': np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
                 }
        pi_hat = copy.deepcopy(mu_hat)
        p_hat = copy.deepcopy(mu_hat)

        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        # nuisance mu
        mu_hat = _dml_cv_predict(self._learner['ml_mu'], dsx, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_mu'), method=self._predict_method['ml_mu'],
                                return_models=return_models)
        mu_hat['targets'] = mu_hat['targets'].astype(float)
        mu_hat['targets'][d != self._treatment] = np.nan
        # is this necessary?
        # _check_finite_predictions(mu_hat, self._learner['ml_mu'], 'ml_mu', smpls)

        # propensity score pi
        pi_hat = _dml_cv_predict(self._learner['ml_pi'], dx, s, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_pi'), method=self._predict_method['ml_pi'],
                                return_models=return_models)
        # TODO: control print
        print("Pi hat:", pi_hat)
        pi_hat['targets'] = pi_hat['targets'].astype(float)
        pi_hat['targets'][d != self._treatment] = np.nan

        # propensity score p
        p_hat = _dml_cv_predict(self._learner['ml_p'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_p'), method=self._predict_method['ml_p'],
                                return_models=return_models)
        p_hat['targets'] = p_hat['targets'].astype(float)
        p_hat['targets'][np.invert(d == 0)] = np.nan

        # TODO: control print
        print("p hat:", p_hat)
        
        ind_d = d == self._treatment
        
        psi_a, psi_b = self._score_elements(ind_d, 
                                            mu_hat['preds'], pi_hat['preds'],
                                            p_hat['preds'], s, y) 
        
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        
        preds = {'predictions': {'ml_mu': mu_hat['preds'], 
                                 'ml_pi': pi_hat['preds'],
                                 'ml_p': p_hat['preds']},
                'targets': {'ml_mu': mu_hat['targets'],
                            'ml_pi': pi_hat['targets'],
                            'ml_p': p_hat['targets']},
                'models': {'ml_mu': mu_hat['models'],
                            'ml_pi': pi_hat['models'],
                            'ml_p': p_hat['models']}
                }

        return psi_elements, preds
    

    def _score_elements(self, ind_d, mu, pi, p, s, y):
        # psi_a
        psi_a = -1

        # psi_b
        psi_b = (ind_d * s * (y - mu)) / (p * pi) + mu

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

        # TODO: This will need adaptation
        train_inds = [train_index for (train_index, _) in smpls]
        
        # hyperparameter tuning for ML 
        mu_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_mu'], param_grids['ml_mu'], scoring_methods['ml_mu'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        pi_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_pi'], param_grids['ml_pi'], scoring_methods['ml_pi'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        p_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_p'], param_grids['ml_p'], scoring_methods['ml_p'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        mu_best_params = [xx.best_params_ for xx in mu_tune_res]
        pi_best_params = [xx.best_params_ for xx in pi_tune_res]
        p_best_params = [xx.best_params_ for xx in p_tune_res]

        params = {'ml_mu': mu_best_params,
                  'ml_pi': pi_best_params,
                  'ml_p': p_best_params}

        tune_res = {'mu_tune': mu_tune_res,
                    'pi_tune': pi_tune_res,
                    'p_tune': p_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
    

    def _sensitivity_element_est(self, preds):
        pass