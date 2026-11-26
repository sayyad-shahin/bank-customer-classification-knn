import pandas as pd
import pytest

from app import create_app
from common import EXAMPLES
from predict import predict_customer, validate
from predict_csv import classify_dataframe


def _ex(name):
    return dict(EXAMPLES[name])


def test_at_risk_vs_loyal(bundle):
    hi, lo = predict_customer(_ex("high"), bundle), predict_customer(_ex("low"), bundle)
    assert hi["probability"] > lo["probability"] + 40
    assert hi["flagged"] and hi["risk_level"] == "High" and not lo["flagged"] and lo["risk_level"] == "Low"


def test_neighbour_explanation(bundle):
    r = predict_customer(_ex("high"), bundle)
    assert r["neighbors_total"] == bundle["k"] and 0 <= r["neighbors_churned"] <= r["neighbors_total"]
    dists = [c["distance"] for c in r["closest"]]
    assert len(dists) == 5 and dists == sorted(dists)
    assert {"outcome", "Age"} <= set(r["closest"][0])
    # uniform-weight model: probability equals the share of neighbours that left
    if bundle["metrics"]["best_params"]["weights"] == "uniform":
        assert abs(r["probability"] - r["neighbors_churned"] / r["neighbors_total"] * 100) < 0.1


def test_input_parsing_and_validation(bundle):
    feats = bundle["features"]
    ok, errs = validate({**_ex("high"), "Geography": " germany ", "IsActiveMember": "No", "Age": "52"}, feats)
    assert not errs and ok["Geography"] == "Germany" and ok["IsActiveMember"] == 0
    _, errs = validate({}, feats)
    assert len(errs) == len(feats)
    _, errs = validate({**_ex("high"), "Geography": "Atlantis", "Age": 500, "NumOfProducts": 9}, feats)
    assert len(errs) == 3
    _, errs = validate({**_ex("high"), "Age": float("nan")}, feats)
    assert errs
    with pytest.raises(ValueError):
        predict_customer({}, bundle)


def test_extra_fields_ignored_and_range_warning(bundle):
    r = predict_customer({**_ex("low"), "Surname": "Smith", "Age": 92, "Balance": 900000}, bundle)
    assert 0 <= r["probability"] <= 100 and any("balance" in w.lower() for w in r["warnings"])


def test_batch_csv(bundle):
    rows = pd.DataFrame([_ex("high"), _ex("low"), {**_ex("low"), "Geography": "Mars"}])
    out = classify_dataframe(rows, bundle)
    assert out.loc[0, "prediction"] == "Likely to leave" and out.loc[1, "prediction"] == "Likely to stay"
    assert out.loc[2, "error"] and pd.isna(out.loc[2, "churn_probability"])


@pytest.fixture()
def client(bundle):
    return create_app(bundle).test_client()


def test_api_flow(client, bundle):
    assert client.get("/").status_code == 200 and client.get("/health").json["status"] == "ok"
    schema = client.get("/api/schema").json
    assert [f["name"] for f in schema["fields"]] == bundle["features"] and set(schema["examples"]) == {"high", "low"}
    r = client.post("/api/predict", json=_ex("high"))
    assert r.status_code == 200 and r.json["flagged"] is True and len(r.json["closest"]) == 5
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", json={**_ex("low"), "Gender": "Robot"}).json["errors"]
    assert client.post("/api/predict", data="nope").status_code == 400
    form = {k: str(v) for k, v in _ex("low").items()}                      # browsers send strings
    assert client.post("/api/predict", json=form).status_code == 200
    info = client.get("/api/model").json
    assert info["dataset"]["rows"] == 10000 and info["k"] == bundle["k"]
