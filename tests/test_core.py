import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, load_dataset
from knn_scratch import KNNScratch


def test_dataset_loading():
    X, y, ids = load_dataset("data/bank_customers.csv")
    assert X.shape == (10000, 10) and list(X.columns) == ALL_FEATURES
    assert int(y.sum()) == 2037 and not X.isna().any().any() and ids.is_unique


def test_scaling_is_essential_for_knn():
    """Balance/salary are ~1e5 while age is ~40: without scaling they dominate the distance."""
    X, y, _ = load_dataset("data/bank_customers.csv")
    cols = ["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary"]
    Xs, ys = X[cols].iloc[:3000], y.iloc[:3000]
    raw = cross_val_score(KNeighborsClassifier(25), Xs, ys, cv=5, scoring="roc_auc").mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(25)), Xs, ys, cv=5, scoring="roc_auc").mean()
    assert scaled > raw + 0.10


def test_scratch_knn_matches_sklearn():
    X, y, _ = load_dataset("data/bank_customers.csv")
    Z = StandardScaler().fit_transform(X[["CreditScore", "Age", "Balance", "EstimatedSalary"]])[:1500]
    y = y.to_numpy()[:1500]
    Xtr, ytr, Xte = Z[:1200], y[:1200], Z[1200:]
    for p in (1, 2):
        for w in ("uniform", "distance"):
            ref = KNeighborsClassifier(15, p=p, weights=w).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            mine = KNNScratch(15, p=p, weights=w).fit(Xtr, ytr).predict_proba(Xte)
            assert np.allclose(ref, mine, atol=1e-6), (p, w)


def test_scratch_knn_obvious_case():
    knn = KNNScratch(3).fit([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], [0, 0, 0, 1, 1, 1])
    assert knn.predict([[0.5, 0.5], [10.5, 10.5]]).tolist() == [0, 1]


def test_trained_model_quality(bundle):
    m = bundle["metrics"]
    assert m["repeated_cv"]["roc_auc"]["mean"] > 0.84
    t = m["test_at_threshold"]
    assert t["roc_auc"] > 0.83 and t["recall"] > 0.55 and t["accuracy"] > m["baseline_accuracy_majority_class"]
    assert m["benchmark_test"]["KNN (tuned)"]["roc_auc"] > m["benchmark_test"]["Logistic Regression"]["roc_auc"]
    assert 0.1 <= bundle["threshold"] <= 0.7 and bundle["k"] >= 15
    assert len(bundle["reference"]) == 10000
    assert "NumOfProducts" in list(m["permutation_importance"])[:2]
