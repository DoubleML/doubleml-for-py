There are three cases for the PLIV model, depending on the definition of the nuisance part, i.e., the relationships that should be learned with machine learning.

1.  ``partialX`` : Partialling out :math:`X` in a IV model with one or multiple instruments,
2.  ``partialXZ`` : Partialling out :math:`X` and :math:`Z` in a IV model with one or multiple instruments,
3.  ``partialZ`` : Partialling out :math:`Z` in a IV model with multiple instruments.

For example, in high-dimensional instrumental variable regression with lasso, this corresponds to the cases where

1. Variable selection is performed for potentially high-dimensional controls :math:`X` only,
2. Variable selection is performed for potentially high-dimensional controls :math:`X` and potentially high-dimensional instruments :math:`Z`, or
3. Variable selection is performed for potentially high-dimensional instruments :math:`Z`.



