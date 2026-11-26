"""Train, tune and evaluate the KNN churn classifier.

Usage: python train.py [--data data/bank_customers.csv]
Outputs: models/knn_model.joblib, models/metrics.json, reports/*.png
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import ALL_FEATURES, BINARY, CATEGORICAL, FEATURE_SETS, NUMERIC, load_dataset

SEED = 42
GRID = {"knn__n_neighbors": [15, 25, 35, 45, 55, 75, 101],
        "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}


def build_pipeline(features, classifier=None) -> Pipeline:
    """Scaling is essential for KNN: without it, big-number columns (balance, salary) swamp the distance."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), [f for f in NUMERIC if f in features]),
        ("bin", StandardScaler(), [f for f in BINARY if f in features]),
        ("cat", OneHotEncoder(drop="if_binary"), [f for f in CATEGORICAL if f in features]),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("knn", KNeighborsClassifier() if classifier is None else classifier)])


def metrics_at(y_true, proba, t):
    pred = (proba >= t).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred)}


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def pick_threshold(model, X_tr, y_tr):
    """Cut-off maximising F1 on cross-validated TRAINING predictions (the test set is never touched)."""
    oof = cross_val_predict(clone(model), X_tr, y_tr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
    grid = np.arange(0.10, 0.71, 0.01)
    return round(float(grid[int(np.argmax([f1_score(y_tr, oof >= t) for t in grid]))]), 2)


def train(X, y, ids=None, verbose=False):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # 1) choose feature set + k + weighting + distance metric by 5-fold CV (training data only)
    searches = {}
    for name, feats in FEATURE_SETS.items():
        gs = GridSearchCV(build_pipeline(feats), GRID, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr[feats], y_tr)
        searches[name] = gs
        if verbose:
            print(f"  feature set {name:17s} best CV AUC={gs.best_score_:.4f}  {gs.best_params_}")
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    gs, feats = searches[best_name], FEATURE_SETS[best_name]
    model, params = gs.best_estimator_, {k.split("__")[1]: v for k, v in gs.best_params_.items()}
    threshold = pick_threshold(model, X_tr[feats], y_tr)

    # 2) honest evaluation on the untouched test set
    proba = model.predict_proba(X_te[feats])[:, 1]
    base_acc = 1 - y_te.mean()

    # 3) context: other algorithms with the same preprocessing + all features
    others = {"KNN (tuned)": (model, feats),
              "Logistic Regression": (build_pipeline(ALL_FEATURES, LogisticRegression(max_iter=1000)), ALL_FEATURES),
              "Random Forest": (build_pipeline(ALL_FEATURES, RandomForestClassifier(300, random_state=SEED, n_jobs=-1)),
                                ALL_FEATURES)}
    bench, curves = {}, {}
    for name, (est, fs) in others.items():
        est = clone(est).fit(X_tr[fs], y_tr) if name != "KNN (tuned)" else est
        p = est.predict_proba(X_te[fs])[:, 1]
        bench[name] = {"roc_auc": round(float(roc_auc_score(y_te, p)), 4),
                       "pr_auc": round(float(average_precision_score(y_te, p)), 4)}
        curves[name] = p

    rep = cross_validate(clone(model), X[feats], y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                         scoring=["roc_auc", "average_precision"], n_jobs=-1)
    imp = permutation_importance(model, X_te[feats], y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)

    res = gs.cv_results_
    k_curve = {str(k): round(float(max(s for kk, w, p_, s in zip(res["param_knn__n_neighbors"], res["param_knn__weights"],
                                                                 res["param_knn__p"], res["mean_test_score"])
                                       if kk == k and w == params["weights"] and p_ == params["p"])), 4)
               for k in GRID["knn__n_neighbors"]}

    metrics = {
        "dataset": {"rows": int(len(X)), "churned": int(y.sum()), "churn_rate": round(float(y.mean()), 4),
                    "train": int(len(X_tr)), "test": int(len(X_te))},
        "feature_set": best_name, "features_used": feats, "best_params": params,
        "feature_set_cv_auc": {n: round(float(s.best_score_), 4) for n, s in searches.items()},
        "threshold": threshold,
        "baseline_accuracy_majority_class": round(float(base_acc), 4),
        "test_at_threshold": r4({**metrics_at(y_te, proba, threshold), "roc_auc": roc_auc_score(y_te, proba),
                                 "pr_auc": average_precision_score(y_te, proba)}),
        "test_at_0.5": r4(metrics_at(y_te, proba, 0.5)),
        "confusion_matrix": confusion_matrix(y_te, proba >= threshold).tolist(),
        "repeated_cv": {"roc_auc": {"mean": round(float(rep["test_roc_auc"].mean()), 4),
                                    "std": round(float(rep["test_roc_auc"].std()), 4)},
                        "pr_auc": {"mean": round(float(rep["test_average_precision"].mean()), 4),
                                   "std": round(float(rep["test_average_precision"].std()), 4)}},
        "benchmark_test": bench,
        "k_curve_cv_auc": k_curve,
        "permutation_importance": {f: round(float(m), 4) for f, m in
                                   sorted(zip(feats, imp.importances_mean), key=lambda t: -t[1])},
    }

    # 4) final model for deployment: same settings, fitted on ALL data; keeps a reference table for "similar customers"
    final = clone(model).fit(X[feats], y)
    reference = X[feats].copy()
    reference[TARGET_COL] = y.values
    ranges = {f: [float(X[f].min()), float(X[f].max())] for f in feats if f in NUMERIC}
    bundle = {"pipeline": final, "features": feats, "threshold": threshold, "k": params["n_neighbors"],
              "ranges": ranges, "reference": reference.reset_index(drop=True), "metrics": metrics}
    return bundle, {"y_te": y_te, "curves": curves, "proba": proba, "searches": searches}


TARGET_COL = "Exited"


def make_plots(bundle, art, out="reports"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).mkdir(exist_ok=True)
    m, y_te = bundle["metrics"], art["y_te"]

    ks = [int(k) for k in m["k_curve_cv_auc"]]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.plot(ks, list(m["k_curve_cv_auc"].values()), "o-", color="#4f46e5")
    ax.axvline(m["best_params"]["n_neighbors"], ls="--", color="gray")
    ax.set(xlabel="k (number of neighbours)", ylabel="5-fold CV ROC-AUC", title="Choosing k")
    fig.tight_layout(); fig.savefig(f"{out}/k_curve.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for name, p in art["curves"].items():
        fpr, tpr, _ = roc_curve(y_te, p); axes[0].plot(fpr, tpr, lw=2, label=f"{name} ({m['benchmark_test'][name]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p); axes[1].plot(rc, pr, lw=2, label=f"{name} ({m['benchmark_test'][name]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (AUC)")
    axes[1].axhline(m["dataset"]["churn_rate"], ls="--", color="gray"); axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (AP)")
    for a in axes: a.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/roc_pr_curves.png", dpi=150); plt.close(fig)

    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1], ["Stayed", "Left"]); ax.set_yticks([0, 1], ["Stayed", "Left"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {m['threshold']})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(f"{out}/confusion_matrix.png", dpi=150); plt.close(fig)

    imp = list(m["permutation_importance"].items())[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([k for k, _ in imp], [v for _, v in imp], color="#4f46e5")
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled", title="Permutation importance")
    fig.tight_layout(); fig.savefig(f"{out}/feature_importance.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bank_customers.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    X, y, ids = load_dataset(args.data)
    print(f"Loaded {len(X)} customers, {int(y.sum())} churned ({y.mean():.1%}).\nTuning (5-fold CV over feature set, k, weights, distance)...")
    bundle, art = train(X, y, ids, verbose=True)
    m = bundle["metrics"]
    print(f"\nBest: feature set '{m['feature_set']}', {m['best_params']}")
    print(f"Repeated 5x3 CV on all data: ROC-AUC {m['repeated_cv']['roc_auc']['mean']:.3f} +/- {m['repeated_cv']['roc_auc']['std']:.3f}, "
          f"PR-AUC {m['repeated_cv']['pr_auc']['mean']:.3f}")
    t = m["test_at_threshold"]
    print(f"\nTEST SET ({m['dataset']['test']} customers) at threshold {m['threshold']}  [majority-class accuracy = {m['baseline_accuracy_majority_class']:.3f}]")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in t.items()), "\n  confusion [[TN FP] [FN TP]]:", m["confusion_matrix"])
    d = m["test_at_0.5"]
    print("  at default 0.5: " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))
    print("\nOther algorithms on the same test set:")
    for n, v in m["benchmark_test"].items():
        print(f"  {n:20s} ROC-AUC={v['roc_auc']:.3f}  PR-AUC={v['pr_auc']:.3f}")
    print("\nPermutation importance:", m["permutation_importance"])

    Path(args.models).mkdir(exist_ok=True)
    joblib.dump(bundle, Path(args.models) / "knn_model.joblib", compress=3)
    (Path(args.models) / "metrics.json").write_text(json.dumps(m, indent=2))
    make_plots(bundle, art, args.reports)
    print(f"\nSaved model to {args.models}/ and plots to {args.reports}/")


if __name__ == "__main__":
    main()
