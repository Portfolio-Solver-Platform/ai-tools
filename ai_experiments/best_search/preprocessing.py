"""Preprocessing transformers for the portfolio-selection features.

The raw features for cpsat8_k1 mix small values (medians ~1-1000) with
columns going to 1e19 (combinatorial counts). A plain StandardScaler is
dominated by those outliers and the RBF kernel collapses. The transforms
here squash the outliers before scaling.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SignedLog1p(BaseEstimator, TransformerMixin):
    """y = sign(x) * log1p(|x|). Preserves sign, monotone, collapses big tails."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


class Asinh(BaseEstimator, TransformerMixin):
    """y = arcsinh(x). Sign-preserving and smooth at zero (no kink like signed log1p)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.arcsinh(np.asarray(X, dtype=np.float64))


class RankNormal(BaseEstimator, TransformerMixin):
    """Per-column rank then inverse-normal-CDF (Van der Waerden scores).

    Fit stores sorted training values per column; transform uses searchsorted
    so unseen test values are placed deterministically. LOYO-safe.
    """
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.sorted_ = np.sort(X, axis=0)
        self.n_train_ = X.shape[0]
        return self

    def transform(self, X):
        from scipy.stats import norm
        X = np.asarray(X, dtype=np.float64)
        n = self.n_train_
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            r = np.searchsorted(self.sorted_[:, j], X[:, j], side="left")
            r = np.clip(r, 0, n)
            u = (r + 0.5) / (n + 1)  # in (0, 1) so norm.ppf is finite
            out[:, j] = norm.ppf(u)
        return out
