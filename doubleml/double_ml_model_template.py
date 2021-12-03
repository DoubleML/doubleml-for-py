from sklearn.utils import check_X_y

from .double_ml import DoubleML
from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions


class DoubleMLNewModel(DoubleML):  # TODO change DoubleMLNewModel to your model name
    """Double machine learning for ??? TODO add your model description

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

    n_folds : int
        Number of folds.
        Default is ``5``.

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
                 ml_g,  # TODO add a entry for each nuisance function
                 ml_m,  # TODO add a entry for each nuisance function
                 n_folds=5,
                 n_rep=1,
                 score='my_orthogonal_score',  # TODO give a name for your orthogonal score function
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

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
        # TODO model specific data requirements can be checked here
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
        # TODO here the score elements psi_a and psi_b should be computed
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
