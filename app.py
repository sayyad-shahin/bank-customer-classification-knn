"""Flask web app + JSON API for bank customer churn classification (KNN)."""
import os

from flask import Flask, jsonify, render_template, request

from common import EXAMPLES, FIELDS
from predict import load_bundle, predict_customer


def create_app(bundle=None):
    app = Flask(__name__)
    bundle = bundle or load_bundle()
    feats = bundle["features"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/schema")
    def schema():
        """Describes the input form (only the features this trained model actually uses)."""
        fields = [{"name": f, **FIELDS[f]} for f in feats]
        examples = {k: {f: v[f] for f in feats} for k, v in EXAMPLES.items()}
        return jsonify({"fields": fields, "examples": examples})

    @app.get("/api/model")
    def model_info():
        m = bundle["metrics"]
        return jsonify({"k": bundle["k"], "threshold": round(bundle["threshold"] * 100), "features": feats,
                        "dataset": m["dataset"], "repeated_cv": m["repeated_cv"], "test": m["test_at_threshold"],
                        "benchmark": m["benchmark_test"]})

    @app.post("/api/predict")
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"errors": ["Send a JSON object with the customer's details."]}), 400
        try:
            return jsonify(predict_customer(data, bundle))
        except ValueError as e:
            return jsonify({"errors": e.args[0]}), 400

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    create_app().run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=int(os.getenv("PORT", "5000")))
