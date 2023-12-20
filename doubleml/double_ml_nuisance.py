import numpy as np
from abc import ABC, abstractmethod


class DoubleMLNuisance(ABC):
    """Double Machine Learning Nuisance estimation."""

    def __init__(
        self,
        obj_dml_data,
        n_folds,
        n_rep,
        score,
    ):
        # initialize learners and parameters which are set model specific
        self._learner = None
        self._params = None

        # initialize predictions and target to None which are only stored if method fit is called with store_predictions=True
        self._predictions = None
        self._nuisance_targets = None
        self._rmses = None

        # initialize models to None which are only stored if method fit is called with store_models=True
        self._models = None

        # initialize sensitivity elements to None (only available if implemented for the class
        self._sensitivity_implemented = False
        self._sensitivity_elements = None
        self._sensitivity_params = None

        # initialize arrays according to obj_dml_data and the resampling settings
        self._psi, self._psi_deriv, self._psi_elements = self._initialize_arrays()

        # initialize instance attributes which are later used for iterating
        self._i_rep = None

    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection.
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.

    @property
    def __smpls(self):
        return self._smpls[self._i_rep]

    @property
    def __smpls_cluster(self):
        return self._smpls_cluster[self._i_rep]

    @property
    def __psi(self):
        return self._psi[:, self._i_rep]

    @property
    def __psi_deriv(self):
        return self._psi_deriv[:, self._i_rep]

    def fit(self, dml_data, n_jobs_cv=None, store_models=False):
        """
        Estimate DoubleML models.

        Parameters
        ----------
        dml_data : DoubleMLData
            The DoubleMLData instance providing the data.
        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        store_models : bool
            Indicates whether the fitted models for the nuisance functions should be stored in ``models``. This allows
            to analyze the fitted models or extract information like variable importance.
            Default is ``False``.

        Returns
        -------
        self : object
        """
        if n_jobs_cv is not None:
            if not isinstance(n_jobs_cv, int):
                raise TypeError(
                    "The number of CPUs used to fit the learners must be of int type. "
                    f"{str(n_jobs_cv)} of type {str(type(n_jobs_cv))} was passed."
                )

        if not isinstance(store_models, bool):
            raise TypeError(
                "store_models must be True or False. " f"Got {str(store_models)}."
            )

        # initialize arrays for nuisance functions evaluation
        self._initialize_rmses()
        self._initialize_predictions_and_targets()
        if store_models:
            self._initialize_models()

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            # ml estimation of nuisance models and computation of score elements
            score_elements, preds = self._nuisance_est(
                self.__smpls, n_jobs_cv, return_models=store_models
            )
            # calculate rmses and store predictions and targets of the nuisance models
            self._calc_rmses(preds['predictions'], preds['targets'])
            self._store_predictions_and_targets(preds['predictions'], preds['targets'])
            if store_models:
                self._store_models(preds['models'])
        return self

    @abstractmethod
    def _nuisance_est(self, smpls, n_jobs_cv, return_models):
        pass

    def _initialize_arrays(self):
        # scores
        psi = np.full(
            (self._dml_data.n_obs, self.n_rep), np.nan
        )
        psi_deriv = np.full(
            (self._dml_data.n_obs, self.n_rep), np.nan
        )
        psi_elements = self._initialize_score_elements(
            (self._dml_data.n_obs, self.n_rep)
        )

        return psi, psi_deriv, psi_elements

    def _initialize_rmses(self):
        self._rmses = {
            learner: np.full((self.n_rep), np.nan) for learner in self.params_names
        }

    def _initialize_models(self):
        self._models = {
            learner: {
                treat_var: [None] * self.n_rep for treat_var in self._dml_data.d_cols
            }
            for learner in self.params_names
        }

    def _initialize_predictions_and_targets(self):
        self._predictions = {
            learner: np.full(
                (self._dml_data.n_obs, self.n_rep, self._dml_data.n_coefs), np.nan
            )
            for learner in self.params_names
        }
        self._nuisance_targets = {
            learner: np.full(
                (self._dml_data.n_obs, self.n_rep, self._dml_data.n_coefs), np.nan
            )
            for learner in self.params_names
        }
