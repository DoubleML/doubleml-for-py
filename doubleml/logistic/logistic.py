import numpy as np
from doubleml.utils._estimation import (
    _dml_cv_predict,
    _trimm,
    _predict_zero_one_propensity,
    _cond_targets,
    _get_bracket_guess,
    _default_kde,
    _normalize_ipw,
    _dml_tune,
    _solve_ipw_score,
)
from sklearn.base import clone
from sklearn.utils import check_X_y
import scipy
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLData, DoubleMLBLP
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import NonLinearScoreMixin
from doubleml.utils import DoubleMLClusterResampling
from doubleml.utils._checks import _check_score, _check_finite_predictions, _check_is_propensity
from doubleml.utils.resampling import DoubleMLDoubleResampling


class DoubleMLLogit(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partially linear regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_r : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`\\ell_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        For binary treatment variables :math:`D` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function
        :math:`g_0(X) = E[Y - D \\theta_0|X]`.
        Note: The learner `ml_g` is only required for the score ``'IV-type'``. Optionally, it can be specified and
        estimated for callable scores.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'partialling out'`` or ``'IV-type'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, l_hat, m_hat, g_hat, smpls)``.
        Default is ``'partialling out'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
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
                 ml_r,
                 ml_m,
                 ml_M,
                 ml_t,
                 ml_a=None,
                 n_folds=5,
                 n_folds_inner=5,
                 n_rep=1,
                 score='logistic',
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        self._check_data(self._dml_data)
        valid_scores = ['logistic']
        _check_score(self.score, valid_scores, allow_callable=True)

        _ = self._check_learner(ml_r, 'ml_r', regressor=True, classifier=False)
        _ = self._check_learner(ml_t, 'ml_t', regressor=True, classifier=False)
        _ = self._check_learner(ml_M, 'ml_M', regressor=False, classifier=True)
        ml_m_is_classifier = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=True)
        self._learner = {'ml_l': ml_r, 'ml_m': ml_m, 'ml_t': ml_t, 'ml_M': ml_M}

        if ml_a is not None:
            ml_a_is_classifier = self._check_learner(ml_a, 'ml_a', regressor=True, classifier=True)
            self._learner['ml_a'] = ml_a
        else:
            self._learner['ml_a'] = clone(ml_m)
            ml_a_is_classifier = ml_m_is_classifier

        self._predict_method = {'ml_r': 'predict', 'ml_t': 'predict', 'ml_M': 'predict_proba'}

        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method['ml_m'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_m learner {str(ml_m)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method['ml_m'] = 'predict'

        if ml_a_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method['ml_a'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_a learner {str(ml_a)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method['ml_a'] = 'predict'

        self._initialize_ml_nuisance_params()
        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        return

    def _double_dml_cv_predict(self, estimator, estimator_name,  x, y, smpls=None, smpls_inner=None,
                    n_jobs=None, est_params=None, method='predict'):
        res = {}
        res['preds'] = np.zeros_like(y)
        res['preds_inner'] = np.zeros_like(y)
        for smpls_single_split, smpls_double_split in zip(smpls, smpls_inner):
            res_inner = _dml_cv_predict(estimator, x, y, smpls=smpls_double_split, n_jobs=n_jobs,
                                    est_params=est_params, method=method,
                                    return_models=True)
            _check_finite_predictions(res_inner['preds'], estimator, estimator_name, smpls_double_split)

            res['preds_inner'] += res_inner['preds']
            for model in res_inner['models']:
                res['models'].append(model)
                res['preds'][smpls_single_split[1]] += model.predict(x[smpls_single_split[1]])

        res["preds"] /= len(smpls)
        res['targets'] = np.copy(y)



    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        x_d_concat = np.hstack([[d, np.newaxis], x])
        r_external = external_predictions['ml_r'] is not None
        m_external = external_predictions['ml_m'] is not None
        M_external = external_predictions['ml_M'] is not None
        t_external = external_predictions['ml_t'] is not None
        if 'ml_a' in self._learner:
            a_external = external_predictions['ml_a'] is not None
        else:
            a_external = False

        # nuisance m
        if m_external:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        if self._check_learner(self._learner['ml_m'], 'ml_m', regressor=True, classifier=True):
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            binary_preds = (type_of_target(m_hat['preds']) == 'binary')
            zero_one_preds = np.all((np.power(m_hat['preds'], 2) - m_hat['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary treatment variable {self._dml_data.d_cols[self._i_treat]}, '
                                 f'predictions obtained with the ml_m learner {str(self._learner["ml_m"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')


        if M_external:
            M_hat = {'preds': external_predictions['ml_M'],
                     'targets': None,
                     'models': None}
        else:
            M_hat = (self.double_dml_cv_predict(self._learner['ml_M'], 'ml_M', x_d_concat, y, smpls=smpls, smpls_inner=smpls_inner,
                                                n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_M'), method=self._predict_method['ml_M']))

        if a_external:
            a_hat = {'preds': external_predictions['ml_a'],
                     'targets': None,
                     'models': None}
        else:
            a_hat = (self.double_dml_cv_predict(self._learner['ml_a'], 'ml_a', x_d_concat, y, smpls=smpls, smpls_inner=smpls_inner,
                                                n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_a'), method=self._predict_method['ml_a']))


        W = scipy.special.logit(M_hat['preds'])
        d_tilde_full = d - a_hat['preds']

        beta_notFold = np.zeros_like(d)

        for _, test in smpls:
            beta_notFold[test] = np.sum(d_tilde_full[test] * W[test]) / np.sum(d_tilde_full[test] ** 2)

        # nuisance t
        if t_external:
            t_hat = {'preds': external_predictions['ml_t'],
                     'targets': None,
                     'models': None}
        else:
            t_hat = _dml_cv_predict(self._learner['ml_t'], x, W, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_t'), method=self._predict_method['ml_t'],
                                    return_models=return_models)
            _check_finite_predictions(t_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        W = scipy.special.expit(M_hat['preds'])

        # nuisance W
        if t_external:
            t_hat = {'preds': external_predictions['ml_t'],
                     'targets': None,
                     'models': None}
        else:
            t_hat = _dml_cv_predict(self._learner['ml_t'], x, W, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_t'), method=self._predict_method['ml_t'],
                                    return_models=return_models)
            _check_finite_predictions(t_hat['preds'], self._learner['ml_t'], 'ml_t', smpls)

        r_hat = {}
        r_hat['preds'] = t_hat['preds'] - beta_notFold * a_hat['preds']


        psi_elements = self._score_elements(y, d, r_hat['preds'], m_hat['preds'])

        preds = {'predictions': {'ml_r': r_hat['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_a': a_hat['preds'],
                                 'ml_t': t_hat['preds'],
                                 'ml_M': M_hat['preds']},
                 'targets': {'ml_r': r_hat['targets'],
                             'ml_m': m_hat['targets'],
                             'ml_a': a_hat['targets'],
                             'ml_t': t_hat['targets'],
                             'ml_M': M_hat['targets']},
                 'models': {'ml_r': None,
                            'ml_m': m_hat['models'],
                            'ml_a': a_hat['models'],
                            'ml_t': t_hat['models'],
                            'ml_M': M_hat['models']}}

        return psi_elements, preds

    def _score_elements(self, y, d, r_hat, m_hat):
        # compute residual
        d_tilde = d - m_hat
        psi_hat = scipy.special.expit(-r)
        score_const = d_tilde * (1 - y) * np.exp(r)
        psi_elements = {"y": y, "d": d, "r_hat": r_hat, "m_hat": m_hat, "psi_hat": psi_hat, "score_const": score_const}

        return psi_elements

    def _sensitivity_element_est(self, preds):
       pass

    def _nuisance_tuning(self):
        pass

    @property
    def __smpls__inner(self):
        return self._smpls[self._i_rep]

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """

        obj_dml_resampling = DoubleMLDoubleResampling(n_folds=self.n_folds,
                                                      n_folds_inner=self.n_folds_inner,
                                                      n_rep=self.n_rep,
                                                      n_obs=self._dml_data.n_obs,
                                                      stratify=self._strata)
        self._smpls, self._smpls_inner = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self):
        raise NotImplementedError('set_sample_splitting is not implemented for DoubleMLLogit.')

    def _compute_score(self, psi_elements, coef):

        score_1 = psi_elements["y"] * np.exp(-coef * psi_elements["r_hat"]) * psi_elements["d_tilde"]


        return psi_elements["psi_hat"] * (score_1 - psi_elements["score_const"])

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        deriv_1 = - psi_elements["y"] * np.exp(-coef * psi_elements["r_hat"]) * psi_elements["d"]

        return psi_elements["psi_hat"] * psi_elements["d_tilde"] *  deriv_1


    def cate(self, basis, is_gate=False):
        """
        Calculate conditional average treatment effects (CATE) for a given basis.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
            where ``n_obs`` is the number of observations and ``d`` is the number of predictors.
        is_gate : bool
            Indicates whether the basis is constructed for GATEs (dummy-basis).
            Default is ``False``.

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model.
        """
        if self._dml_data.n_treat > 1:
            raise NotImplementedError('Only implemented for single treatment. ' +
                                      f'Number of treatments is {str(self._dml_data.n_treat)}.')
        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        Y_tilde, D_tilde = self._partial_out()

        D_basis = basis * D_tilde
        model = DoublelMLBLP(
            orth_signal=Y_tilde.reshape(-1),
            basis=D_basis,
            is_gate=is_gate,
        )
        model.fit()

        ## TODO: Solve score


        return model

    def gate(self, groups):
        """
        Calculate group average treatment effects (GATE) for groups.

        Parameters
        ----------
        groups : :class:`pandas.DataFrame`
            The group indicator for estimating the best linear predictor. Groups should be mutually exclusive.
            Has to be dummy coded with shape ``(n_obs, d)``, where ``n_obs`` is the number of observations
            and ``d`` is the number of groups or ``(n_obs, 1)`` and contain the corresponding groups (as str).

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model for Group Effects.
        """

        if not isinstance(groups, pd.DataFrame):
            raise TypeError('Groups must be of DataFrame type. '
                            f'Groups of type {str(type(groups))} was passed.')
        if not all(groups.dtypes == bool) or all(groups.dtypes == int):
            if groups.shape[1] == 1:
                groups = pd.get_dummies(groups, prefix='Group', prefix_sep='_')
            else:
                raise TypeError('Columns of groups must be of bool type or int type (dummy coded). '
                                'Alternatively, groups should only contain one column.')

        if any(groups.sum(0) <= 5):
            warnings.warn('At least one group effect is estimated with less than 6 observations.')

        model = self.cate(groups, is_gate=True)
        return model

    def _partial_out(self):
        """
        Helper function. Returns the partialled out quantities of Y and D.
        Works with multiple repetitions.

        Returns
        -------
        Y_tilde : :class:`numpy.ndarray`
            The residual of the regression of Y on X.
        D_tilde : :class:`numpy.ndarray`
            The residual of the regression of D on X.
        """
        if self.predictions is None:
            raise ValueError('predictions are None. Call .fit(store_predictions=True) to store the predictions.')

        y = self._dml_data.y.reshape(-1, 1)
        d = self._dml_data.d.reshape(-1, 1)
        ml_m = self.predictions["ml_m"].squeeze(axis=2)

        if self.score == "partialling out":
            ml_l = self.predictions["ml_l"].squeeze(axis=2)
            Y_tilde = y - ml_l
            D_tilde = d - ml_m
        else:
            assert self.score == "IV-type"
            ml_g = self.predictions["ml_g"].squeeze(axis=2)
            Y_tilde = y - (self.coef * ml_m) - ml_g
            D_tilde = d - ml_m

        return Y_tilde, D_tilde