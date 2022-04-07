import numpy as np
from sklearn.utils import check_X_y


from .double_ml import DoubleML
from .double_ml_data import DiffInDiffRCDoubleMLData, DiffInDiffRODoubleMLData
from ._utils import _dml_cv_predict, _get_cond_smpls, _check_finite_predictions


class DoubleMLDiD(DoubleML):
    """Double machine learning for Difference in Difference models

    Parameters
    ----------
    obj_dml_data : :class:`DiffInDiffRODoubleMLData` or :class:`DiffInDiffRCDoubleMLData` object
        The :class:`DiffInDiffRODoubleMLData` or :class:`DiffInDiffRCDoubleMLData` object providing the data
         and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(D,X) = E[Y|X,D]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'ortho_ro'`` or ``'ortho_rcs'``) specifying the score function
        Default is ``'ortho_ro'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

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
    >>> from doubleml.datasets import make_diff_in_diff_chang2020
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> obj_dml_data = make_diff_in_diff_chang2020(theta = 3)
    >>> dml_did_obj = dml.DoubleMLDiD(obj_dml_data, ml_g, ml_m)
    >>> dml_did_obj.fit().summary
            coef	std err	t	P>|t|	2.5 %	97.5 %
        D	2.998143	0.260907	11.49124	1.460007e-30	2.486775	3.509511
    """

    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='ortho_ro',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
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
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}

        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def _check_data(self, obj_dml_data):
        if not (isinstance(obj_dml_data, DiffInDiffRCDoubleMLData) or isinstance(obj_dml_data, DiffInDiffRODoubleMLData)):
            raise TypeError('Incompatible data. '
                            'The data object should be an instance of either '
                            '`DiffInDiffRODoubleMLData` or `DiffInDiffRODoubleMLData`')

    def _check_score(self, score):
        if not isinstance(score, str):
            raise TypeError('score should be a string.'
                            ' %r was passed.' % score)

        valid_score = ['ortho_ro', 'ortho_rcs']
        if score not in valid_score:
            raise ValueError('Invalid score ' + score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')

        if score == 'ortho_ro' and isinstance(self._dml_data, DiffInDiffRCDoubleMLData):
            raise ValueError('Invalid Score `ortho_ro` with `DiffInDiffRCDoubleMLData`. '
                             'Use `ortho_rcs` or change the databackend to `DiffInDiffRODoubleMLData`')
        if score == 'ortho_rcs' and isinstance(self._dml_data, DiffInDiffRODoubleMLData):
            raise ValueError('Invalid Score `ortho_rcs` with `DiffInDiffRODoubleMLData`. '
                             'Use `ortho_ro` or change the databackend to `DiffInDiffRCDoubleMLData`')

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in ['ml_g', 'ml_m']}

    def _nuisance_est(self, smpls, n_jobs_cv):
        if self.score == "ortho_ro":
            return self._nuisance_est_ro(smpls, n_jobs_cv)

        return

    def _nuisance_est_ro(self, smpls, n_jobs_cv):
        x, y0 = check_X_y(self._dml_data.x, self._dml_data.y,
                          force_all_finite=False)
        x, y1 = check_X_y(self._dml_data.x, self._dml_data.y_treated,
                          force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, _ = _get_cond_smpls(smpls, d)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y1 - y0, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat, self._learner['ml_g'], 'ml_g', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        psi_a, psi_b = self._score_elements_ro(y1, y0, d, g_hat, m_hat, smpls)
        preds = {'ml_g': g_hat, 'ml_m': m_hat}
        return psi_a, psi_b, preds

    def _score_elements_ro(self, y1, y0, d, g_hat, m_hat, smpls):
        p_hat = np.full_like(d, np.nan, dtype='float64')
        for _, test_index in smpls:
            p_hat[test_index] = np.mean(d[test_index])

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - \
                self.trimming_threshold

        psi_b = (y1 - y0)/p_hat
        psi_b *= (d - m_hat)/(1-m_hat)
        c_1 = (d - m_hat)/((1 - m_hat)*p_hat)
        c_1 *= g_hat
        psi_b -= c_1

        psi_a = np.full_like(m_hat, -1.0)

        return psi_a, psi_b

    def _nuisance_tuning(self):
        pass
