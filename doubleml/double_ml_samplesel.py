from sklearn.utils import check_X_y

from doubleml.double_ml import DoubleML
from doubleml.double_ml_data import DoubleMLData
# from .double_ml import DoubleML -- not working
from doubleml._utils import _dml_cv_predict, _dml_tune
from doubleml._utils_checks  import _check_finite_predictions
#from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions -- also not working


class DoubleMLSS(DoubleML):
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

    # TODO give a name for your orthogonal score function
    score : str or callable
        A str (``'my_orthogonal_score'``) specifying the score function.
        Default is ``'my_orthogonal_score'``.

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

    # Simulation study
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_pliv_CHS2015
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)

    >>> n = 2000  # sample size
    >>> p = 100  # number of covariates
    >>> s = 2  # number of covariates that are confounders
    >>> sigma = np.array([1, 0.5], [0.5, 1])
    >>> e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n).T
    >>> x = np.random.randn(n, p)  # covariate matrix
    >>> beta = np.hstack((np.repeat(0.25, s), np.repeat(0, p - s)))  # Coefficients determining the degree of confounding
    >>> d = np.where(np.dot(x, beta) + np.random.randn(n) > 0, 1, 0)  # Treatment equation
    >>> z = np.random.randn(n)
    >>> s = np.where(np.dot(x, beta) + 0.25 * d + z + e[0] > 0, 1, 0)  # Selection equation
    >>> y = np.dot(x, beta) + 0.5 * d + e[1]  # Outcome equation
    >>> y[s == 0] = 0  # Setting values to 0 based on the selection equation

    #  The true ATE is equal to 0.5
    


    Notes
    -----
    # TODO add an description of the model
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,  # TODO add a entry for each nuisance function
                 ml_m,
                 selection,  # 1 if y is observed and 0 y is not observed (missing)
                 non_random_missing = False,  # indicates whether MAR holds or not
                 trim = 0.01, 
                 dtreat = 1,
                 dcontrol = 1,
                 n_folds=3,
                 n_rep=1,
                 score='my_orthogonal_score',  # TODO give a name for your orthogonal score function
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
        
        self._normalize_ipw = normalize_ipw
        self.__selection = selection

        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)  # TODO may needs adaption
        _ = self._check_learner(ml_g, 'ml_m', regressor=False, classifier=True)  # TODO may needs adaption
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}  # TODO may needs adaption
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}  # TODO may needs adaption

        self._initialize_ml_nuisance_params()

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                        ['ml_g', 'ml_m']}  # TODO may needs adaption

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['my_orthogonal_score']  # TODO give a name for your orthogonal score function
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
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLSS.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv):
        # TODO data checks may need adaptions
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # TODO add a entry for each nuisance function
        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat, self._learner['ml_g'], 'ml_g', smpls)

        # TODO add a entry for each nuisance function
        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        psi_a, psi_b = self._score_elements(y, d, g_hat, m_hat, smpls)  # TODO may needs adaption
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}

        return psi_a, psi_b, preds

    def _score_elements(self, y, d, g_hat, m_hat, smpls):  # TODO may needs adaption
        # TODO Here the score elements psi_a and psi_b of a linear score psi = psi_a * theta + psi_b should be computed
        # TODO See also https://docs.doubleml.org/stable/guide/scores.html
        # return psi_a, psi_b
        pass

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        # TODO data checks may need adaptions
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}  # TODO may needs adaption

        train_inds = [train_index for (train_index, _) in smpls]
        # TODO add a entry for each nuisance function
        g_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params}  # TODO may needs adaption

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}  # TODO may needs adaption

        res = {'params': params,
               'tune_res': tune_res}

        return res
