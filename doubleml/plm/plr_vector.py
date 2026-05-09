"""Partially Linear Regression (PLR) multi-treatment model based on the DoubleMLVector hierarchy."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from ..data.base_data import DoubleMLData
from ..double_ml_scalar import DoubleMLScalar
from ..double_ml_vector import DoubleMLVector
from .plr_scalar import PLR


class PLRVector(DoubleMLVector):
    """Multi-treatment double machine learning for partially linear regression models.

    Orchestrates one :class:`~doubleml.plm.plr_scalar.PLR` instance per treatment column
    in ``d_cols``. Sample splits are drawn once and shared across all sub-models;
    learners are propagated (and cloned per sub-model) via :meth:`set_learners`.
    The scalar :class:`~doubleml.DoubleMLFramework` objects are concatenated into a
    single multi-treatment framework after fit.

    Parameters
    ----------
    obj_dml_data : DoubleMLData
        The data object providing the data and specifying the variables for the causal
        model. May contain multiple treatment columns in ``d_cols``.
    score : str
        The score function (``'partialling out'`` or ``'IV-type'``).
        Default is ``'partialling out'``.
    ml_l : estimator, optional
        Learner for E[Y|X]. Can be regressor or classifier.
    ml_m : estimator, optional
        Learner for E[D|X]. Can be regressor or classifier.
    ml_g : estimator, optional
        Learner for E[Y - D*theta|X]. Only for IV-type. Must be regressor.
    """

    def __init__(
        self,
        obj_dml_data: DoubleMLData,
        score: str = "partialling out",
        ml_l: object | None = None,
        ml_m: object | None = None,
        ml_g: object | None = None,
    ) -> None:
        # Validate at the vector level so the error fires before sub-model construction.
        self._check_data(obj_dml_data)
        valid_scores = ["partialling out", "IV-type"]
        if score not in valid_scores:
            raise ValueError(f"Invalid score '{score}'. Valid scores: {valid_scores}.")
        if score == "IV-type" and obj_dml_data.binary_outcome:
            raise ValueError("For score = 'IV-type', additive probability models (binary outcomes) are not supported.")

        super().__init__(obj_dml_data=obj_dml_data, score=score)
        self._modellist = self._initialize_models()

        if any(learner is not None for learner in (ml_l, ml_m, ml_g)):
            self.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)

    @staticmethod
    def _check_data(obj_dml_data: Any) -> None:
        """Validate the data object for PLR vector estimation.

        Parameters
        ----------
        obj_dml_data : Any
            Data candidate. Must be a :class:`~doubleml.data.DoubleMLData` without
            instrumental variables.

        Raises
        ------
        TypeError
            If ``obj_dml_data`` is not a :class:`~doubleml.data.DoubleMLData`.
        ValueError
            If ``obj_dml_data`` defines instrumental variables (``z_cols``).
        """
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. {str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise ValueError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "PLRVector does not support instrumental variables."
            )

    @property
    def required_learners(self) -> list[str]:
        """Required learners for the current score."""
        names = ["ml_l", "ml_m"]
        if self.score == "IV-type":
            names.append("ml_g")
        return names

    def set_learners(
        self,
        ml_l: object | None = None,
        ml_m: object | None = None,
        ml_g: object | None = None,
    ) -> Self:
        """Set the learners for nuisance estimation on every sub-model.

        Parameters
        ----------
        ml_l : estimator or None, optional
            Learner for :math:`\\ell_0(X) = E[Y|X]`.
        ml_m : estimator or None, optional
            Learner for :math:`m_0(X) = E[D|X]`.
        ml_g : estimator or None, optional
            Learner for :math:`g_0(X) = E[Y - D\\theta_0|X]`. Required for ``score='IV-type'``.

        Returns
        -------
        self : PLRVector
        """
        if self._modellist is None:
            raise RuntimeError("Sub-models are not initialized. _initialize_models() must run in __init__.")
        for model in self._modellist:
            model.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)
        self._reset_fit_state()
        return self

    def _initialize_models(self) -> list[DoubleMLScalar]:
        """Create one PLR sub-model per treatment column."""
        return [PLR(obj_dml_data=self._get_data_for_model(d_col), score=self.score) for d_col in self._dml_data.d_cols]

    def cate(self, *args: Any, **kwargs: Any) -> Any:
        """Not implemented for multi-treatment PLR."""
        raise NotImplementedError(
            "cate() is not defined for multi-treatment PLR. "
            "Use the single-treatment PLR (doubleml.plm.plr_scalar.PLR) instead."
        )

    def gate(self, *args: Any, **kwargs: Any) -> Any:
        """Not implemented for multi-treatment PLR."""
        raise NotImplementedError(
            "gate() is not defined for multi-treatment PLR. "
            "Use the single-treatment PLR (doubleml.plm.plr_scalar.PLR) instead."
        )
