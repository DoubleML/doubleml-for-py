import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from .double_ml import DoubleML
from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions


class DoubleMLPLR(DoubleML):
    """Double machine learning for partially linear regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        For binary treatment variables :math:`D` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'partialling out'`` or ``'IV-type'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat, m_hat, smpls)``.
        Default is ``'partialling out'``.

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
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_plr_CCDDHNR2018
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_g = learner
    >>> ml_m = learner
    >>> obj_dml_data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20)
    >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
    >>> dml_plr_obj.fit().summary
           coef  std err          t         P>|t|     2.5 %    97.5 %
    d  0.462321  0.04107  11.256983  2.139582e-29  0.381826  0.542816

    Notes
    -----
    **Partially linear regression (PLR)** models take the form

    .. math::

        Y = D \\theta_0 + g_0(X) + \\zeta, & &\\mathbb{E}(\\zeta | D,X) = 0,

        D = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
    The high-dimensional vector :math:`X = (X_1, \\ldots, X_p)` consists of other confounding covariates,
    and :math:`\\zeta` and :math:`V` are stochastic errors.
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
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
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        ml_m_is_classifier = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        if ml_m_is_classifier:
            if obj_dml_data.binary_treats.all():
                self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_m learner {str(ml_m)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict'}

        self._initialize_ml_nuisance_params()

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['IV-type', 'partialling out']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat, self._learner['ml_g'], 'ml_g', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            binary_preds = (type_of_target(m_hat) == 'binary')
            zero_one_preds = np.all((np.power(m_hat, 2) - m_hat) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary treatment variable {self._dml_data.d_cols[self._i_treat]}, '
                                 f'predictions obtained with the ml_m learner {str(self._learner["ml_m"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

        psi_a, psi_b = self._score_elements(y, d, g_hat, m_hat, smpls)
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}

        return psi_a, psi_b, preds

    def _score_elements(self, y, d, g_hat, m_hat, smpls):
        # compute residuals
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.multiply(v_hat, d)

        if isinstance(self.score, str):
            if self.score == 'IV-type':
                psi_a = -v_hatd
            else:
                assert self.score == 'partialling out'
                psi_a = -np.multiply(v_hat, v_hat)
            psi_b = np.multiply(v_hat, u_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y, d, g_hat, m_hat, smpls)

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        g_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
