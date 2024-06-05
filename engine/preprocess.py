import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler


class PreprocessModule:

    scalers = {
        "maxabs": {
            "model": MaxAbsScaler(),
            "docstr": """
Scale each feature by its maximum absolute value.

This estimator scales and translates each feature individually such that the maximal absolute value
of each feature in the training set will be 1.0. It does not shift/center the data, and thus does
not destroy any sparsity.""",
        },
        "minmax": {
            "model": MinMaxScaler(),
            "docstr": """
Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on
the training set, e.g. between zero and one.

The transformation is given by:
```
    min = X.min(axis=0)
    max = X.max(axis=0)
    r = max - min
    X_std = (X - min) / r
    X_scaled = X_std * r + min
```
where min, max = feature_range.

This transformation is often used as an alternative to zero mean, unit variance scaling.""",
        },
        "standard": {
            "model": StandardScaler(),
            "docstr": """
Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample x is calculated as:
```
    z = (x - u) / s
```
where u is the mean of the training samples""",
        },
        "robust": {
            "model": RobustScaler(),
            "docstr": """
Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to the IQR. The IQR is the range
between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).""",
        },
    }

    imputers = {
        "mean": {
            "model": SimpleImputer(strategy="mean"),
            "docstr": """
Replace missing values using the mean along each column. 

Can only be used with numeric data.""",
        },
        "median": {
            "model": SimpleImputer(strategy="median"),
            "docstr": """
Replace missing values using the median along each column. 

Can only be used with numeric data.
""",
        },
        "most_frequent": {
            "model": SimpleImputer(strategy="most_frequent"),
            "docstr": """
Replace missing using the most frequent value along each column. 

Can be used with strings or numeric data. If there is more than one such value, only the smallest is
returned.""",
        },
    }

    @classmethod
    def scale(self, X: pd.DataFrame, columns: list[str], method: str = "standard") -> pd.DataFrame:
        """Transforms data using one of the specified scalers.

        Args:
            X: Data set. Shape (n_samples, n_features).
            columns: List of column names to which to apply scaler.
            method: Method. Should be one of the keys of `PreprocessModule.scalers` field.

        Returns:
            Transformed data set.
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"
        assert method in self.scalers, f"Unrecognized method. Should be one of {self.scalers.keys()}"
        assert len(columns) > 0, "Expected at least one column"

        scaler = self.scalers[method]["model"].set_output(transform="pandas")
        X = X.copy()
        X.loc[:, columns] = scaler.fit_transform(X.loc[:, columns])
        return X

    @classmethod
    def impute(self, X: pd.DataFrame, columns: list[str], method: str = "mean") -> pd.DataFrame:
        """Imputes missing data using one of the specified imputers.

        Args:
            X: Data set. Shape (n_samples, n_features).
            columns: List of column names to which to apply imputer.
            method: Method. Should be one of the keys of `PreprocessModule.imputers` field.

        Returns:
            Transformed data set.
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"
        assert method in self.imputers, f"Unrecognized method. Should be one of {self.imputers.keys()}"
        assert len(columns) > 0, "Expected at least one column"

        imputer = self.imputers[method]["model"].set_output(transform="pandas")
        X = X.copy()
        X.loc[:, columns] = imputer.fit_transform(X.loc[:, columns])
        return X
