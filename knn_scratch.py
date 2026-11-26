"""k-Nearest Neighbours implemented from scratch with NumPy (vectorised brute force).

Included to show how the algorithm works; the tests check it agrees with scikit-learn.
"""
import numpy as np


class KNNScratch:
    def __init__(self, k=5, p=2, weights="uniform"):
        self.k, self.p, self.weights = k, p, weights

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, int)
        return self

    def _distances(self, Q):
        diff = np.abs(Q[:, None, :] - self.X[None, :, :])          # (queries, train, features)
        return diff.sum(-1) if self.p == 1 else np.sqrt((diff ** 2).sum(-1))

    def predict_proba(self, Q, chunk=256):
        Q, out = np.asarray(Q, float), []
        for s in range(0, len(Q), chunk):
            d = self._distances(Q[s:s + chunk])
            idx = np.argpartition(d, self.k - 1, axis=1)[:, :self.k]       # k nearest (unordered)
            nd, ny = np.take_along_axis(d, idx, 1), self.y[idx]
            w = np.ones_like(nd) if self.weights == "uniform" else 1.0 / np.maximum(nd, 1e-12)
            out.append((w * ny).sum(1) / w.sum(1))                          # weighted share of class 1
        return np.concatenate(out)

    def predict(self, Q, threshold=0.5):
        return (self.predict_proba(Q) >= threshold).astype(int)
