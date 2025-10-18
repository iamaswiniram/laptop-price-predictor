from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

class ColumnAs2D(BaseEstimator, TransformerMixin):
    """Ensure a pandas Series becomes 2D for downstream transformers."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, (pd.Series, list, np.ndarray)):
            return np.asarray(X).reshape(-1, 1)
        raise ValueError("ColumnAs2D expects a 1D array-like or Series.")

class DataFrameMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    """
    Fits one MultiLabelBinarizer per specified list-like column.
    Columns should contain sequences (lists); empty sequence becomes [].
    """
    def __init__(self, columns=None):
        self.columns = columns
        self._mlbs = {}
        self._feature_names = []

    def fit(self, X, y=None):
        if self.columns is None:
            # Auto-detect list-like object columns
            self.columns = [
                c for c in X.columns
                if X[c].apply(lambda v: isinstance(v, (list, tuple, set))).all()
            ]
        for col in self.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col].apply(lambda v: v if isinstance(v, (list, tuple, set)) else []))
            self._mlbs[col] = mlb
        # Build feature names
        self._feature_names = []
        for col, mlb in self._mlbs.items():
            self._feature_names.extend([f"{col}__{cls}" for cls in mlb.classes_])
        return self

    def transform(self, X):
        transformed_parts = []
        for col, mlb in self._mlbs.items():
            col_values = X[col].apply(lambda v: v if isinstance(v, (list, tuple, set)) else [])
            arr = mlb.transform(col_values)
            transformed_parts.append(arr)
        if not transformed_parts:
            return np.empty((len(X), 0))
        return np.concatenate(transformed_parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names, dtype=object)
