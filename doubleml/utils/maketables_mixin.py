"""
MakeTables Mixin for DoubleML Models.

This module provides a mixin class that adds MakeTables plug-in support to DoubleML models.
The mixin implements the three required attributes for MakeTables compatibility:
- __maketables_coef_table__: Returns coefficient table as DataFrame
- __maketables_stat__: Returns model statistics by key
- __maketables_depvar__: Returns dependent variable name

This enables zero-coupling integration with MakeTables - DoubleML never imports maketables,
but models automatically work with it when users have maketables installed.
"""

import numpy as np
import pandas as pd


class MakeTablesMixin:
    """
    Mixin class for MakeTables plug-in support.

    This mixin adds three attributes that enable DoubleML models to automatically work
    with the MakeTables package for creating publication-ready regression tables.

    The plug-in format uses duck typing - MakeTables automatically detects these
    attributes when present, without requiring any imports or dependencies.

    Attributes
    ----------
    __maketables_coef_table__ : pd.DataFrame (property)
        Coefficient table with columns 'b' (estimates), 'se' (standard errors),
        'p' (p-values), 't' (t-statistics), 'ci95l', 'ci95u' (95% CI bounds).

    __maketables_depvar__ : str (property)
        Name of the dependent variable.

    __maketables_default_stat_keys__ : list (property)
        Default statistics to display in tables.

    Methods
    -------
    __maketables_stat__(key)
        Return model statistic by key (e.g., 'N' for number of observations).

    Examples
    --------
    >>> from doubleml import DoubleMLPLR
    >>> # After fitting a DoubleML model
    >>> dml_plr.fit()
    >>> # Access maketables attributes
    >>> coef_table = dml_plr.__maketables_coef_table__
    >>> n_obs = dml_plr.__maketables_stat__('N')
    >>> depvar = dml_plr.__maketables_depvar__
    """

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        """
        Return coefficient table with all required and optional columns for MakeTables.

        Returns a pandas DataFrame with coefficient estimates, standard errors, p-values,
        t-statistics, and 95% confidence intervals. The DataFrame index matches the
        treatment variable names from the fitted model.

        Returns
        -------
        pd.DataFrame
            Coefficient table with columns:
            - 'b': coefficient estimates (required)
            - 'se': standard errors (required)
            - 'p': p-values (required)
            - 't': t-statistics (optional)
            - 'ci95l': lower 95% confidence interval bound (optional)
            - 'ci95u': upper 95% confidence interval bound (optional)

        Notes
        -----
        - Returns empty DataFrame with correct columns if model is unfitted or all coefficients are NaN
        - Index is set to match the summary table index (treatment variable names)
        - Handles edge cases gracefully without raising errors
        """
        # Handle unfitted model
        if not hasattr(self, "coef") or self.coef is None:
            return pd.DataFrame(columns=["b", "se", "t", "p", "ci95l", "ci95u"])

        # Handle NaN coefficients (model fitted but no valid estimates)
        if np.isnan(self.coef).all():
            return pd.DataFrame(columns=["b", "se", "t", "p", "ci95l", "ci95u"])

        # Get 95% confidence intervals
        ci = self.confint(level=0.95)

        # Build coefficient table with required and optional columns
        coef_table = pd.DataFrame(
            {
                "b": self.coef,  # Required: coefficient estimates
                "se": self.se,  # Required: standard errors
                "p": self.pval,  # Required: p-values
                "t": self.t_stat,  # Optional: t-statistics
                "ci95l": ci.iloc[:, 0],  # Optional: lower 95% CI bound
                "ci95u": ci.iloc[:, 1],  # Optional: upper 95% CI bound
            }
        )

        # Set index to match summary table (handles treatment variable names)
        if hasattr(self, "summary") and self.summary is not None and len(self.summary) > 0:
            coef_table.index = self.summary.index

        return coef_table

    def __maketables_stat__(self, key: str):
        """
        Return model statistic by key.

        Parameters
        ----------
        key : str
            The statistic key to retrieve. Common keys include:
            - 'N': number of observations
            - 'r2': R-squared (not applicable for DoubleML)
            - 'adj_r2': adjusted R-squared (not applicable for DoubleML)
            - 'aic': Akaike Information Criterion (not applicable for DoubleML)
            - 'bic': Bayesian Information Criterion (not applicable for DoubleML)
            - 'll': log-likelihood (not applicable for DoubleML)

        Returns
        -------
        float, int, or None
            The requested statistic value, or None if not available or not applicable.

        Notes
        -----
        DoubleML focuses on causal inference, not prediction, so traditional model fit
        statistics like R-squared, AIC, and BIC are not applicable and will return None.
        Currently only 'N' (number of observations) is supported.

        Examples
        --------
        >>> n_obs = dml_model.__maketables_stat__('N')
        >>> r2 = dml_model.__maketables_stat__('r2')  # Returns None
        """
        stats_map = {
            "N": self.n_obs if hasattr(self, "n_obs") else None,
        }
        return stats_map.get(key, None)

    @property
    def __maketables_depvar__(self) -> str:
        """
        Return the name of the dependent variable.

        Returns
        -------
        str
            Name of the dependent (outcome) variable. Defaults to "Y" if not available.

        Notes
        -----
        Retrieves the dependent variable name from the DoubleMLData object's y_col attribute.
        Falls back to "Y" if the attribute is not available.

        Examples
        --------
        >>> depvar = dml_model.__maketables_depvar__
        >>> print(depvar)
        'Y'
        """
        if hasattr(self, "_dml_data") and hasattr(self._dml_data, "y_col"):
            return self._dml_data.y_col
        return "Y"  # Fallback

    @property
    def __maketables_default_stat_keys__(self) -> list:
        """
        Return default statistics to display in MakeTables output.

        Returns
        -------
        list
            List of statistic keys to display by default. For DoubleML models,
            this is ['N'] (number of observations).

        Notes
        -----
        This is an optional attribute that helps MakeTables know which statistics
        to include in the table by default. Users can override this when calling
        ETable() by specifying the model_stats parameter.

        Examples
        --------
        >>> default_stats = dml_model.__maketables_default_stat_keys__
        >>> print(default_stats)
        ['N']
        """
        return ["N"]
