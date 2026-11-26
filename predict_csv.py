"""Classify many customers at once.

Usage: python predict_csv.py data/sample_input.csv predictions.csv
Input needs the columns the model uses (see `python predict.py --help`). Output adds:
churn_probability (%), prediction (Likely to leave / Likely to stay), risk_level, error.
"""
import sys

import pandas as pd

from predict import load_bundle, risk_level, validate


def classify_dataframe(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    feats, thr = bundle["features"], bundle["threshold"]
    out = df.copy()
    out["churn_probability"], out["prediction"], out["risk_level"], out["error"] = None, None, None, ""
    ok_idx, ok_rows = [], []
    for i, row in df.iterrows():
        vals, errs = validate(row.to_dict(), feats)
        if errs:
            out.at[i, "error"] = " ".join(errs)
        else:
            ok_idx.append(i); ok_rows.append(vals)
    if ok_rows:
        p = bundle["pipeline"].predict_proba(pd.DataFrame(ok_rows, columns=feats))[:, 1]
        for i, pi in zip(ok_idx, p):
            out.at[i, "churn_probability"] = round(float(pi) * 100, 1)
            out.at[i, "prediction"] = "Likely to leave" if pi >= thr else "Likely to stay"
            out.at[i, "risk_level"] = risk_level(float(pi), thr)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    result = classify_dataframe(pd.read_csv(sys.argv[1]))
    result.to_csv(sys.argv[2], index=False)
    bad = int((result["error"] != "").sum())
    print(f"Classified {len(result) - bad} customers ({bad} rows had errors) -> {sys.argv[2]}")
    print(result["prediction"].value_counts().to_string())
