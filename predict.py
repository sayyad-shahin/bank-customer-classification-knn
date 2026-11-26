"""Churn prediction for one customer, with input validation and 'most similar customers' explanation.

CLI:  python predict.py --example high
      python predict.py --age 52 --geography Germany --gender Female --balance 120000 \
                        --num-of-products 1 --is-active-member 0
"""
import argparse
import math
import re
from pathlib import Path

import joblib
import pandas as pd

from common import EXAMPLES, FIELDS

MODEL_PATH = Path(__file__).parent / "models" / "knn_model.joblib"
HIGH_RISK_P = 0.60
_cache = {}


def load_bundle(path=MODEL_PATH):
    path = str(path)
    if path not in _cache:
        if not Path(path).exists():
            raise FileNotFoundError("Model not found. Run `python train.py` first.")
        try:
            _cache[path] = joblib.load(path)
        except Exception as e:   # typically a scikit-learn version mismatch
            raise RuntimeError("Could not load the saved model (scikit-learn version mismatch?). "
                               "Re-create it with `python train.py`.") from e
    return _cache[path]


def _parse(feature, value):
    """Returns (parsed_value, error_message_or_None)."""
    spec = FIELDS[feature]
    label = spec["label"]
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, f"{label} is required."
    if spec["kind"] == "select":
        opts = spec["options"]
        if isinstance(opts[0], str):                                    # text options, case-insensitive
            match = [o for o in opts if o.lower() == str(value).strip().lower()]
            return (match[0], None) if match else (None, f"{label} must be one of: {', '.join(opts)}.")
        txt = str(value).strip().lower()                                # numeric / yes-no options
        txt = {"yes": "1", "true": "1", "no": "0", "false": "0"}.get(txt, txt)
        try:
            num = int(float(txt))
        except ValueError:
            return None, f"{label} must be one of: {', '.join(map(str, opts))}."
        return (num, None) if num in opts else (None, f"{label} must be one of: {', '.join(map(str, opts))}.")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None, f"{label} must be a number."
    if not math.isfinite(x) or not spec["min"] <= x <= spec["max"]:
        return None, f"{label} must be between {spec['min']:g} and {spec['max']:g}."
    return x, None


def validate(raw: dict, features):
    clean, errors = {}, []
    for f in features:
        v, err = _parse(f, raw.get(f))
        if err:
            errors.append(err)
        clean[f] = v
    return clean, errors


def risk_level(p, threshold):
    if p < threshold:
        return "Low"
    return "High" if p >= max(HIGH_RISK_P, threshold) else "Elevated"


def predict_customer(raw: dict, bundle=None, n_closest=5) -> dict:
    """raw: dict of customer fields. Raises ValueError(list_of_messages) on invalid input."""
    bundle = bundle or load_bundle()
    feats = bundle["features"]
    values, errors = validate(raw, feats)
    if errors:
        raise ValueError(errors)

    pipe, thr, k = bundle["pipeline"], bundle["threshold"], bundle["k"]
    df = pd.DataFrame([values], columns=feats)
    p = float(pipe.predict_proba(df)[0, 1])

    # the k most similar customers in the training data (KNN's built-in explanation)
    Xq = pipe.named_steps["pre"].transform(df)
    dist, idx = pipe.named_steps["knn"].kneighbors(Xq, n_neighbors=k)
    ref = bundle["reference"].iloc[idx[0]]
    closest = []
    for (_, row), d in list(zip(ref.iterrows(), dist[0]))[:n_closest]:
        item = {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in feats}
        item.update(outcome="Left" if row["Exited"] == 1 else "Stayed", distance=round(float(d), 3))
        closest.append(item)

    warnings = []
    for f, (lo, hi) in bundle["ranges"].items():
        if not lo <= values[f] <= hi:
            warnings.append(f"{FIELDS[f]['label']} = {values[f]:g} is outside the range in the training data "
                            f"({lo:g}-{hi:g}); the estimate may be less reliable.")

    return {"probability": round(p * 100, 1), "flagged": p >= thr, "risk_level": risk_level(p, thr),
            "threshold": round(thr * 100), "k": k, "neighbors_churned": int(ref["Exited"].sum()),
            "neighbors_total": int(len(ref)), "closest": closest, "warnings": warnings, "used_features": feats}


def _kebab(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def main():
    bundle = load_bundle()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", choices=list(EXAMPLES), help="use a built-in example customer")
    for f in bundle["features"]:
        ap.add_argument("--" + _kebab(f), dest=f, default=None)
    args = vars(ap.parse_args())
    raw = dict(EXAMPLES.get(args.pop("example"), {})) if args.get("example") else {}
    raw.update({k: v for k, v in args.items() if v is not None and k != "example"})
    try:
        r = predict_customer(raw, bundle)
    except ValueError as e:
        raise SystemExit("Invalid input:\n  - " + "\n  - ".join(e.args[0]))
    print(f"Churn risk: {r['probability']}%  ->  {r['risk_level']}  (alert threshold {r['threshold']}%)")
    print(f"Of the {r['neighbors_total']} most similar customers, {r['neighbors_churned']} left the bank.\nClosest 3:")
    for c in r["closest"][:3]:
        print("  - " + ", ".join(f"{k}={v}" for k, v in c.items() if k != "distance") + f"  (distance {c['distance']})")
    for w in r["warnings"]:
        print("Warning:", w)


if __name__ == "__main__":
    main()
