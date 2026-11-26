"""Feature definitions and data loading shared by training, prediction and the web app."""
import pandas as pd

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY = ["HasCrCard", "IsActiveMember"]
CATEGORICAL = ["Geography", "Gender"]
ALL_FEATURES = NUMERIC + BINARY + CATEGORICAL
TARGET = "Exited"

# Candidate feature sets - the best one is chosen by cross-validation in train.py.
FEATURE_SETS = {
    "all": ALL_FEATURES,
    "no_salary_tenure": [f for f in ALL_FEATURES if f not in ("EstimatedSalary", "Tenure")],
    "core": ["Age", "Balance", "NumOfProducts", "IsActiveMember", "Geography", "Gender"],
}

# Input form / validation spec (also drives the web form).
FIELDS = {
    "CreditScore": dict(label="Credit score", kind="number", min=300, max=900, step=1, help="300 - 900"),
    "Geography": dict(label="Country", kind="select", options=["France", "Germany", "Spain"]),
    "Gender": dict(label="Gender", kind="select", options=["Female", "Male"]),
    "Age": dict(label="Age", kind="number", min=18, max=100, step=1, help="years"),
    "Tenure": dict(label="Tenure", kind="number", min=0, max=40, step=1, help="years with the bank"),
    "Balance": dict(label="Account balance", kind="number", min=0, max=1_000_000, step=0.01, help="0 allowed"),
    "NumOfProducts": dict(label="Number of products", kind="select", options=[1, 2, 3, 4]),
    "HasCrCard": dict(label="Has credit card", kind="select", options=[1, 0], names={1: "Yes", 0: "No"}),
    "IsActiveMember": dict(label="Active member", kind="select", options=[1, 0], names={1: "Yes", 0: "No"}),
    "EstimatedSalary": dict(label="Estimated salary", kind="number", min=0, max=1_000_000, step=0.01),
}
EXAMPLES = {
    "high": dict(CreditScore=600, Geography="Germany", Gender="Female", Age=52, Tenure=3, Balance=120000,
                 NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=90000),
    "low": dict(CreditScore=720, Geography="France", Gender="Male", Age=32, Tenure=6, Balance=0,
                NumOfProducts=2, HasCrCard=1, IsActiveMember=1, EstimatedSalary=80000),
}


def load_dataset(path="data/bank_customers.csv"):
    """Returns (X, y, customer_ids)."""
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    ids = df["CustomerId"] if "CustomerId" in df.columns else pd.Series(range(len(df)))
    return df[ALL_FEATURES].copy(), df[TARGET].astype(int), ids
