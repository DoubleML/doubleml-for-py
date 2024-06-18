import numpy as np

from ..double_ml import DoubleML

from ..double_ml_score_mixins import LinearScoreMixin
from ..double_ml_data import DoubleMLData

from ..utils._checks import _check_score, _check_trimming, _check_weights


class DoubleMLAPO(LinearScoreMixin, DoubleML):
    """Double machine learning average potential outcomes for interactive regression models

    Parameters
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 treatment_level,
                 n_folds=5,
                 n_rep=1,
                 score='APO',
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        # set up treatment level and check data
        self._treatment_level = treatment_level
        self._treated = self._dml_data.d == self._treatment_level

        self._check_data(self._dml_data)
        valid_scores = ['APO']
        _check_score(self.score, valid_scores, allow_callable=False)

        # set stratication for resampling
        self._strata = self._dml_data.d
        if draw_sample_splitting:
            self.draw_sample_splitting()

        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._normalize_ipw = normalize_ipw
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        self._normalize_ipw = normalize_ipw
        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

        # ATE weights are the standard case
        _check_weights(weights, score="ATE", n_obs=obj_dml_data.n_obs, n_rep=self.n_rep)
        self._initialize_weights(weights)

        return self

    @property
    def treatment_level(self):
        """
        Chosen treatment level for average potential outcomes.
        """
        return self._treatment_level

    @property
    def treated(self):
        """
        Indicator for treated observations (with the corresponding treatment level).
        """
        return self._treated

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
    def weights(self):
        """
        Specifies the weights for a weighted average potential outcome.
        """
        return self._weights

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _nuisance_est(self):
        # Estimate nuisance parameters
        # This is a placeholder for the estimation logic
        print("Estimating nuisance parameters...")

    def _nuisance_tuning(self):
        # Tune nuisance parameters
        # This is a placeholder for tuning logic
        print("Tuning nuisance parameters...")

    def _sensitivity_element_est(self):
        # Estimate sensitivity elements
        # This is a placeholder for sensitivity estimation logic
        print("Estimating sensitivity elements...")

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s).')

        # check if treatment level is valid
        if np.sum(self.treated) < 5:
            raise ValueError(
                'The number of treated observations is less than 5. ' +
                f'Number of treated observations: {np.sum(self.treated)} for treatment level {self.treatment_level}.'
            )

        return
