"""MakeTables plug-in support for models inheriting from DoubleML."""

import numpy as np
import pandas as pd

_COEF_COLUMNS = ["b", "se", "t", "p", "ci95l", "ci95u"]

_STAT_LABELS = {
    "N": "Observations",
    "se_type": "S.E. type",
    "n_folds": "Folds",
    "n_rep": "Sample-split repetitions",
}


class MakeTablesMixin:
    """Expose fitted DoubleML results through the MakeTables plug-in protocol.

    MakeTables detects these attributes without DoubleML importing it. Users
    install ``maketables`` separately to create tables from fitted models.
    """

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """Return estimates and inference for MakeTables.

        Returns
        -------
        pd.DataFrame
            Estimates (``b``), standard errors (``se``),
            t-statistics (``t``), p-values (``p``), and pointwise 95% confidence
            interval bounds (``ci95l``, ``ci95u``). Confidence intervals use
            DoubleML's aggregation across repeated sample splits. After
            :meth:`bootstrap`, joint 95% bounds are added as ``ci95l_joint`` and
            ``ci95u_joint`` and can be shown through ``ETable(coef_fmt=...)``.

        Notes
        -----
        Rows are named ``"<treatment> (<estimand>)"``, e.g. ``"d (ATE)"``,
        ``"d (APO(1))"`` or ``"d (PQ(0.5, d=1))"``, so that columns from models
        estimating different quantities do not share a row. Pass these names
        as keys to ``ETable(labels=...)`` to relabel them.

        Before fitting or after resetting sample splits, returns an empty table
        with the same columns, consistent with ``DoubleML.summary``.
        """
        if self.framework is None or np.isnan(self.coef).all():
            return pd.DataFrame(columns=_COEF_COLUMNS)

        ci = self.confint(joint=False, level=0.95)
        table = pd.DataFrame(
            {
                "b": self.coef,
                "se": self.se,
                "t": self.t_stat,
                "p": self.pval,
                "ci95l": ci.iloc[:, 0].to_numpy(),
                "ci95u": ci.iloc[:, 1].to_numpy(),
            },
            index=[f"{name} ({self._maketables_estimand()})" for name in ci.index],
        )
        if self.boot_t_stat is not None:
            ci_joint = self.confint(joint=True, level=0.95)
            table["ci95l_joint"] = ci_joint.iloc[:, 0].to_numpy()
            table["ci95u_joint"] = ci_joint.iloc[:, 1].to_numpy()
        return table

    def __maketables_stat__(self, key: str) -> int | str | None:
        """Return a model statistic for MakeTables.

        Parameters
        ----------
        key : str
            Statistic identifier: ``'N'`` (unweighted count of observations in
            the data object, including those without an observed outcome in
            sample selection models), ``'n_folds'``, or ``'n_rep'``
            (sample-split repetitions; with more than one, estimates and
            standard errors are medians across repetitions).

        Returns
        -------
        int, str or None
            The statistic, or ``None`` for an unsupported key.
        """
        if key == "N":
            return self.n_obs
        if key == "n_folds":
            return self.n_folds
        if key == "n_rep":
            return self.n_rep if self.n_rep == 1 else f"{self.n_rep} (median)"
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return the outcome variable name.

        Returns
        -------
        str
            Outcome column of the model's data object.
        """
        return self._dml_data.y_col

    @property
    def __maketables_vcov_info__(self) -> dict[str, object]:
        """Return the variance estimator for the MakeTables ``se_type`` row.

        Returns
        -------
        dict
            ``vcov_type`` is ``'robust'``: standard errors come from the
            variance of the estimated score, without assuming homoskedasticity.
            ``clustervar`` lists the cluster columns, or is ``None`` without
            clustering.
        """
        cluster_cols = self._dml_data.cluster_cols if self._is_cluster_data else None
        return {"vcov_type": "robust", "clustervar": cluster_cols}

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return row labels for the statistics in ``__maketables_stat__``."""
        return dict(_STAT_LABELS)

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Return the default model statistics for MakeTables.

        Returns
        -------
        list[str]
            ``['N', 'se_type', 'n_rep']``. Users can override this
            through ``ETable(model_stats=...)``.
        """
        return ["N", "se_type", "n_rep"]

    def _maketables_estimand(self) -> str:
        # Partially linear models have no named estimand, so the model is shown.
        model = type(self).__name__.removeprefix("DoubleML")
        score = self.score if isinstance(self.score, str) else "custom score"
        if hasattr(self, "quantile"):
            estimand = f"{score}({self.quantile}, d={self.treatment})"
        elif hasattr(self, "treatment_level"):
            estimand = f"{score}({self.treatment_level})"
        elif model in ("IRM", "IIVM"):
            estimand = score
        elif model.startswith("DID"):
            estimand = "ATTE"
        elif model == "SSM":
            estimand = "ATE"
        else:
            estimand = model
        weights = getattr(self, "weights", None)
        if isinstance(weights, dict) and ("weights_bar" in weights or not np.all(weights["weights"] == 1)):
            estimand = f"Weighted {estimand}"
        return estimand
