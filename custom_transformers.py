import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class ColumnAs2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.to_numpy().reshape(-1, 1)

class DataFrameMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binarizers = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        self.feature_names_ = []
        for col in X.columns:
            mlb = MultiLabelBinarizer(sparse_output=False)
            mlb.fit(X[col])
            self.binarizers[col] = mlb
            self.feature_names_.extend([f"{col}_{cls}" for cls in mlb.classes_])
        return self

    def transform(self, X):
        all_transformed = []
        for col in X.columns:
            all_transformed.append(self.binarizers[col].transform(X[col]))
        return np.hstack(all_transformed)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)