"""MakeTables plug-in support for models inheriting from DoubleML."""

import numpy as np
import pandas as pd


class MakeTablesMixin:
    """Expose fitted DoubleML results through the MakeTables plug-in protocol.

    MakeTables detects these attributes without DoubleML importing it. Users
    install ``maketables`` separately to create tables from fitted models.
    """

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """Return estimates and pointwise inference for MakeTables.

        Returns
        -------
        pd.DataFrame
            Treatment-indexed estimates (``b``), standard errors (``se``),
            t-statistics (``t``), p-values (``p``), and pointwise 95% confidence
            interval bounds (``ci95l``, ``ci95u``). Confidence intervals use
            DoubleML's aggregation across repeated sample splits.

        Notes
        -----
        Before fitting or after resetting sample splits, returns an empty table
        with the same columns, consistent with ``DoubleML.summary``.
        """
        if self.framework is None or np.isnan(self.coef).all():
            return pd.DataFrame(columns=["b", "se", "t", "p", "ci95l", "ci95u"])

        ci = self.confint(joint=False, level=0.95)
        return pd.DataFrame(
            {
                "b": self.coef,
                "se": self.se,
                "t": self.t_stat,
                "p": self.pval,
                "ci95l": ci.iloc[:, 0],
                "ci95u": ci.iloc[:, 1],
            },
            index=ci.index,
        )

    def __maketables_stat__(self, key: str) -> int | None:
        """Return a model statistic for MakeTables.

        Parameters
        ----------
        key : str
            Statistic identifier. Only ``'N'`` (sample size) is supported.

        Returns
        -------
        int or None
            Number of observations, or ``None`` for an unsupported statistic.
        """
        return self.n_obs if key == "N" else None

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
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Return the default model statistics for MakeTables.

        Returns
        -------
        list[str]
            ``['N']``. Users can override this through ``ETable(model_stats=...)``.
        """
        return ["N"]
