"""
serve_api.py -- CO6: REST-based serving of the packaged model
-----------------------------------------------------------------
Minimal Flask app that loads the model package produced by
co_extension_pipeline.py (outputs/model_package.joblib) and exposes a
single /predict endpoint. This demonstrates the "model packaging +
REST-based serving" practice named in CO6, using the SAME feature
computation logic as training (no train/serve skew).

Run locally with:
    pip install flask joblib scikit-learn pandas numpy
    python serve_api.py
Then:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"WEATHER":1,"LGT_COND":2,"RUR_URB":2,"HOUR":22,"DAY_WEEK":6,"hotspot_member":1}'
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load once at startup, not per-request -- standard serving practice.
PACKAGE = joblib.load("outputs/model_package.joblib")
MODEL = PACKAGE["model"]
FEATURES = PACKAGE["features"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features_expected": FEATURES})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"missing required features: {missing}"}), 400

    row = pd.DataFrame([{f: payload[f] for f in FEATURES}])
    prob = MODEL.predict_proba(row)[0, 1]
    pred = "MULTI-FATALITY RISK" if prob >= 0.5 else "single-fatality risk"

    return jsonify({
        "input": payload,
        "multi_fatality_risk_probability": round(float(prob), 3),
        "prediction": pred,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
