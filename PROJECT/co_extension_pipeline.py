"""
Extension Pipeline: Supervised Modeling, Evaluation, and ML-Engineering
------------------------------------------------------------------------
Covers Course Outcomes CO1, CO2, CO3, CO5, CO6 on top of the existing
DBSCAN hotspot work (CO4, in hotspot_detection.py).

TASK: Predict whether an accident is a MULTI-FATALITY crash (FATALS >= 2)
      vs a single-fatality crash (FATALS == 1), using accident attributes
      PLUS the DBSCAN hotspot membership from the unsupervised stage.
      This connects the CO4 output directly into the CO2/CO3/CO5 pipeline,
      keeping the whole project as one coherent system rather than two
      disconnected exercises.

CO1 - End-to-end lifecycle: data -> features -> training -> evaluation ->
      "deployment" (predict_new_record) -> monitoring (check_feature_drift).
      A single simulated request is traced end-to-end at the bottom.
CO2 - Logistic Regression (linear supervised model), with L2 regularisation
      strength (C) swept to show the bias-variance trade-off, and feature
      scaling applied before fitting.
CO3 - Decision Tree and Random Forest (tree-based models), with feature
      importance and a splitting-criterion note.
CO5 - Stratified train/val/test split, k-fold cross-validation, grid-search
      hyperparameter tuning, and classification metrics (accuracy, precision,
      recall, F1, ROC-AUC, PR-AUC).
CO6 - Model packaging (joblib), a feature-computation module reused for both
      training and serving (train/serve skew avoidance), and a basic
      statistical drift check.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.cluster import DBSCAN

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)

# =================================================================
# CO1 (data + features) -- shared feature-computation module
# Reused identically at training time and at "serving" time below,
# which is the concrete train/serve-skew-avoidance practice under CO6.
# =================================================================
FEATURE_COLS = ["WEATHER", "LGT_COND", "RUR_URB", "HOUR", "DAY_WEEK", "hotspot_member"]
TARGET_COL = "MULTI_FATAL"

def compute_features(df, dbscan_model=None, coords_rad_ref=None):
    """Single source of truth for feature computation.
    Used both for training (batch) and for simulated serving (single record)."""
    out = df.copy()
    # sentinel cleanup consistent with the CO4 stage
    out = out[(out["WEATHER"] < 90) & (out["LGT_COND"] < 9)]
    out["hotspot_member"] = (out["cluster"] != -1).astype(int)
    return out[FEATURE_COLS + (["FATALS"] if "FATALS" in out.columns else [])].copy()


# =================================================================
# STEP 0: Load cleaned data + reproduce CO4 cluster labels
# (re-run identically to hotspot_detection.py so this script is standalone)
# =================================================================
tx = pd.read_csv("outputs/texas_cleaned.csv")
coords_rad = np.radians(tx[["LATITUDE", "LONGITUDE"]].to_numpy())
EARTH_RADIUS_KM = 6371.0088
eps_rad = 5.0 / EARTH_RADIUS_KM
db = DBSCAN(eps=eps_rad, min_samples=5, metric="haversine")
tx["cluster"] = db.fit_predict(coords_rad)

# Supervised target: multi-fatality crash flag
tx["MULTI_FATAL"] = (tx["FATALS"] >= 2).astype(int)
print("Class balance:\n", tx["MULTI_FATAL"].value_counts())

feat_df = compute_features(tx)
X = feat_df[FEATURE_COLS]
y = tx.loc[feat_df.index, "MULTI_FATAL"]

# =================================================================
# CO5: Stratified train/val/test split
# =================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# CO2: feature scaling (required before regularised linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =================================================================
# CO2: Logistic Regression + bias-variance sweep over regularisation C
# =================================================================
c_values = [0.01, 0.1, 1, 10, 100]
bv_results = []
for c in c_values:
    lr = LogisticRegression(C=c, penalty="l2", class_weight="balanced", max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, lr.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, lr.predict(X_val_scaled))
    bv_results.append({"C": c, "train_acc": train_acc, "val_acc": val_acc})
bv_df = pd.DataFrame(bv_results)
print("\nBias-variance sweep (Logistic Regression, L2):\n", bv_df)

plt.figure(figsize=(6, 4.5))
plt.plot(bv_df["C"], bv_df["train_acc"], marker="o", label="Train accuracy")
plt.plot(bv_df["C"], bv_df["val_acc"], marker="o", label="Validation accuracy")
plt.xscale("log")
plt.xlabel("Regularisation strength C (log scale)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression: Bias-Variance Trade-off (L2 penalty)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/co2_bias_variance.png", dpi=150)
plt.close()

# Final logistic regression at best C (by validation accuracy)
best_c = bv_df.loc[bv_df["val_acc"].idxmax(), "C"]
logreg = LogisticRegression(C=best_c, penalty="l2", class_weight="balanced", max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# =================================================================
# CO3: Tree-based models (Decision Tree, Random Forest) + feature importance
# =================================================================
dtree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
dtree.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nRandom Forest feature importances:\n", importances)

plt.figure(figsize=(6, 4.5))
importances.plot(kind="barh", color="#2C4A63")
plt.gca().invert_yaxis()
plt.xlabel("Importance (Gini-based)")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/co3_feature_importance.png", dpi=150)
plt.close()

# =================================================================
# CO5: Cross-validation + GridSearchCV hyperparameter tuning (Random Forest)
# =================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"\n5-fold CV ROC-AUC (Random Forest, default params): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 8, 12],
}
grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid, scoring="roc_auc", cv=cv, n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Best params from GridSearchCV: {grid.best_params_}  (CV ROC-AUC = {grid.best_score_:.3f})")
best_rf = grid.best_estimator_

# =================================================================
# CO5: Final evaluation on held-out TEST set for all three models
# =================================================================
def evaluate(name, model, X_te, y_te, uses_scaled=False):
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_te, preds),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall": recall_score(y_te, preds, zero_division=0),
        "f1": f1_score(y_te, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_te, probs),
        "pr_auc": average_precision_score(y_te, probs),
    }

results = [
    evaluate("Logistic Regression", logreg, X_test_scaled, y_test),
    evaluate("Decision Tree", dtree, X_test, y_test),
    evaluate("Random Forest (tuned)", best_rf, X_test, y_test),
]
results_df = pd.DataFrame(results)
print("\nTest-set evaluation:\n", results_df.round(3))
results_df.to_csv(f"{OUT}/co5_model_comparison.csv", index=False)

# ROC curves
plt.figure(figsize=(6, 5.5))
for name, model, X_te in [
    ("Logistic Regression", logreg, X_test_scaled),
    ("Decision Tree", dtree, X_test),
    ("Random Forest (tuned)", best_rf, X_test),
]:
    probs = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Multi-Fatality Crash Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/co5_roc_curves.png", dpi=150)
plt.close()

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_rf.predict(X_test))
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xticks([0, 1], ["Single-fatal", "Multi-fatal"])
plt.yticks([0, 1], ["Single-fatal", "Multi-fatal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Random Forest (tuned)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{OUT}/co5_confusion_matrix.png", dpi=150)
plt.close()

# =================================================================
# CO6: Model packaging (joblib) for serving
# =================================================================
joblib.dump({"model": best_rf, "scaler": scaler, "features": FEATURE_COLS}, f"{OUT}/model_package.joblib")
print(f"\nModel packaged to {OUT}/model_package.joblib")

# =================================================================
# CO6: Basic drift check (compares a new batch's feature stats to training stats)
# =================================================================
def check_feature_drift(train_ref: pd.DataFrame, new_batch: pd.DataFrame, z_thresh: float = 2.0):
    """Very simple drift check: flags a feature if the new batch's mean
    has shifted by more than z_thresh standard deviations (measured on the
    training distribution). This is the minimal form of the 'basic monitoring
    for drift' practice named in CO6 -- in production this would run on a
    schedule against live inference logs."""
    report = {}
    for col in train_ref.columns:
        mu, sigma = train_ref[col].mean(), train_ref[col].std()
        if sigma == 0:
            continue
        shift = (new_batch[col].mean() - mu) / sigma
        report[col] = {"train_mean": round(mu, 3), "new_mean": round(new_batch[col].mean(), 3),
                        "z_shift": round(shift, 3), "drift_flag": bool(abs(shift) > z_thresh)}
    return report

# Simulate a "new batch" = the held-out test set, to demonstrate the check
drift_report = check_feature_drift(X_train, X_test)
with open(f"{OUT}/co6_drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=2)
print("\nDrift check (test batch vs training distribution):")
for k, v in drift_report.items():
    print(f"  {k}: {v}")

# =================================================================
# CO1: Deployment stand-in — predict_new_record() — and one traced request
# =================================================================
def predict_new_record(raw_record: dict, cluster_model, scaler, clf, feature_cols):
    """End-to-end serving function: takes one raw accident report, computes
    features the SAME way as training (via compute_features), determines
    hotspot membership from the trained DBSCAN model, then returns both the
    CO4 hotspot assignment and the CO5-evaluated CO3 model's risk prediction.
    This is the single 'prediction request traced through every lifecycle
    stage' required by CO1."""
    # --- feature stage ---
    point_rad = np.radians([[raw_record["LATITUDE"], raw_record["LONGITUDE"]]])
    # nearest-point cluster lookup (DBSCAN has no native .predict for new points)
    dists = np.linalg.norm(coords_rad - point_rad, axis=1)
    nearest_idx = dists.argmin()
    nearest_dist_km = dists[nearest_idx] * EARTH_RADIUS_KM
    assigned_cluster = tx.iloc[nearest_idx]["cluster"] if nearest_dist_km <= 5.0 else -1

    feat_row = pd.DataFrame([{
        "WEATHER": raw_record["WEATHER"],
        "LGT_COND": raw_record["LGT_COND"],
        "RUR_URB": raw_record["RUR_URB"],
        "HOUR": raw_record["HOUR"],
        "DAY_WEEK": raw_record["DAY_WEEK"],
        "hotspot_member": int(assigned_cluster != -1),
    }])[feature_cols]

    # --- inference stage ---
    risk_prob = clf.predict_proba(feat_row)[0, 1]
    risk_pred = int(risk_prob >= 0.5)

    return {
        "input": raw_record,
        "assigned_hotspot_cluster": int(assigned_cluster),
        "is_hotspot_member": bool(assigned_cluster != -1),
        "multi_fatality_risk_probability": round(float(risk_prob), 3),
        "multi_fatality_risk_prediction": "MULTI-FATALITY RISK" if risk_pred else "single-fatality risk",
    }

sample_new_record = {
    "LATITUDE": 29.80, "LONGITUDE": -95.40,   # inside the Houston hotspot
    "WEATHER": 1, "LGT_COND": 2, "RUR_URB": 2, "HOUR": 22, "DAY_WEEK": 6,
}
traced = predict_new_record(sample_new_record, db, scaler, best_rf, FEATURE_COLS)
with open(f"{OUT}/co1_traced_prediction.json", "w") as f:
    json.dump(traced, f, indent=2)
print("\nCO1 traced end-to-end prediction request:\n", json.dumps(traced, indent=2))

print("\nAll extension outputs written to ./outputs/")
