"""
GA9 Healthcare IoT Monitoring System
Step 3 (Firestore-connected version): Cloud AI Anomaly Detection Pipeline

This version connects to your REAL Firebase Firestore project instead of
just reading/writing local CSVs. It:
  1. Pushes the simulated readings into Firestore (patients/patient_001/readings)
  2. Trains the personalised Isolation Forest model on the baseline period
  3. Scores live data and writes risk scores back to Firestore
     (patients/patient_001/risk_scores)

Run this from inside the scripts/ folder, same as before.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import time

import firebase_admin
from firebase_admin import credentials, firestore

# ── Firebase connection ─────────────────────────────────────────────────

SERVICE_ACCOUNT_PATH = "C:/Users/byrst/OneDrive/Desktop/Healthcare IoT Monitoring System/healthcare-iot-monitoring-sys-servAcc-key.json"
PATIENT_ID = "patient_001"

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

print("Connected to Firestore.")

# ── Configuration ────────────────────────────────────────────────────────

BASELINE_DAYS = 7
CONTAMINATION = 0.01
ROLLING_WINDOW = 10
RISK_THRESHOLD = 54
BASE_FEATURES = ["heart_rate_bpm", "spo2_percent", "temperature_c"]

DATA_PATH = "C:/Users/byrst/OneDrive/Desktop/Healthcare IoT Monitoring System/data/simulated_patient_vitals.csv"

# Cap how many readings we push to Firestore -- 40,320 individual writes
# would be slow and could exceed free-tier daily write quotas. We push a
# representative subset: the full baseline week + all live-period anomaly
# events + a light sample of normal live readings.
MAX_READINGS_TO_PUSH = 1500


def add_rolling_features(df, features=BASE_FEATURES, window=ROLLING_WINDOW):
    df = df.copy()
    for f in features:
        df[f"{f}_roll_mean"] = df[f].rolling(window, min_periods=1).mean()
        df[f"{f}_roll_std"] = df[f].rolling(window, min_periods=1).std().fillna(0)
    return df


FEATURES = BASE_FEATURES + [f"{f}_roll_mean" for f in BASE_FEATURES] + [f"{f}_roll_std" for f in BASE_FEATURES]


def push_readings_to_firestore(df, patient_id, max_readings=MAX_READINGS_TO_PUSH):
    """
    Pushes a representative subset of readings to Firestore using batched
    writes (much faster and within Firestore's batch limits of 500/batch).
    """
    # Always include every non-normal reading (the events that matter)
    anomalous = df[df["label"] != "normal"]
    # Sample normal readings to fill the remaining budget
    remaining_budget = max(max_readings - len(anomalous), 0)
    normal_sample = df[df["label"] == "normal"].sample(
        n=min(remaining_budget, len(df[df["label"] == "normal"])),
        random_state=1
    )
    to_push = pd.concat([anomalous, normal_sample]).sort_values("timestamp")

    print(f"Pushing {len(to_push)} readings to Firestore (patients/{patient_id}/readings)...")

    readings_ref = db.collection("patients").document(patient_id).collection("readings")

    batch = db.batch()
    count = 0
    for idx, row in to_push.iterrows():
        doc_ref = readings_ref.document(f"reading_{idx:06d}")
        batch.set(doc_ref, {
            "timestamp": row["timestamp"].isoformat(),
            "heart_rate_bpm": float(row["heart_rate_bpm"]),
            "spo2_percent": float(row["spo2_percent"]),
            "temperature_c": float(row["temperature_c"]),
            "tinyml_classification": row["label"],
            "source": "simulated"
        })
        count += 1
        if count % 450 == 0:  # stay under Firestore's 500-operation batch limit
            batch.commit()
            batch = db.batch()
            print(f"  ...committed {count} readings")

    batch.commit()
    print(f"Finished pushing {count} readings to Firestore.")


def set_patient_profile(patient_id, baseline_stats):
    profile_ref = db.collection("patients").document(patient_id)
    profile_ref.set({
        "profile": {
            "name": "Test Patient",
            "age": 58,
            "condition": "post-operative cardiac monitoring",
            "baseline_hr": round(baseline_stats["heart_rate_bpm"]["mean"], 1),
            "baseline_spo2": round(baseline_stats["spo2_percent"]["mean"], 1),
            "baseline_temp": round(baseline_stats["temperature_c"]["mean"], 2),
            "baseline_established": True,
            "caregiver_phone": "+27000000000"
        }
    }, merge=True)
    print(f"Patient profile written to Firestore for {patient_id}.")


def push_risk_scores_to_firestore(scored_df, patient_id, only_flagged=True):
    """
    Writes risk scores back to Firestore. By default only writes FLAGGED
    anomalies (this is what a clinician dashboard actually needs to see --
    not 20,000 'all clear' entries).
    """
    to_write = scored_df[scored_df["anomaly_detected"]] if only_flagged else scored_df
    print(f"Writing {len(to_write)} risk score entries to Firestore (patients/{patient_id}/risk_scores)...")

    scores_ref = db.collection("patients").document(patient_id).collection("risk_scores")
    batch = db.batch()
    count = 0
    for idx, row in to_write.iterrows():
        doc_ref = scores_ref.document(f"score_{idx:06d}")
        batch.set(doc_ref, {
            "timestamp": row["timestamp"].isoformat(),
            "risk_score": float(row["risk_score"]),
            "anomaly_detected": bool(row["anomaly_detected"]),
            "contributing_factors": row["contributing_factors"],
            "model_version": "isolation_forest_v1"
        })
        count += 1
        if count % 450 == 0:
            batch.commit()
            batch = db.batch()
            print(f"  ...committed {count} risk scores")

    batch.commit()
    print(f"Finished writing {count} risk score entries to Firestore.")


def identify_contributing_factors(row, baseline_stats):
    factors = []
    for feature in BASE_FEATURES:
        mean = baseline_stats[feature]["mean"]
        std = baseline_stats[feature]["std"]
        z = abs((row[feature] - mean) / std) if std > 0 else 0
        if z > 2.0:
            factors.append(feature)
    return factors


def main():
    print("Loading simulated dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    print(f"Engineering rolling window features ({ROLLING_WINDOW} samples = 5 min)...")
    df = add_rolling_features(df)

    cutoff = df["timestamp"].min() + pd.Timedelta(days=BASELINE_DAYS)
    baseline_df = df[df["timestamp"] < cutoff].copy()
    live_df = df[df["timestamp"] >= cutoff].copy()

    baseline_stats = {
        feature: {"mean": baseline_df[feature].mean(), "std": baseline_df[feature].std()}
        for feature in BASE_FEATURES
    }

    print("\nTraining personalised Isolation Forest model on baseline period...")
    scaler = StandardScaler()
    X_baseline = scaler.fit_transform(baseline_df[FEATURES])
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42)
    model.fit(X_baseline)

    print("Scoring live (unseen) data...")
    X_live = scaler.transform(live_df[FEATURES])
    raw_scores = model.decision_function(X_live)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    risk_score = 100 * (max_s - raw_scores) / (max_s - min_s + 1e-9)

    scored_df = live_df.copy()
    scored_df["risk_score"] = np.round(risk_score, 1)
    scored_df["anomaly_detected"] = scored_df["risk_score"] >= RISK_THRESHOLD
    scored_df["contributing_factors"] = scored_df.apply(
        lambda row: identify_contributing_factors(row, baseline_stats) if row["anomaly_detected"] else [],
        axis=1
    )

    # Quick local evaluation, same as before, just to confirm the model still performs
    eval_df = scored_df[scored_df["label"] != "watch"]
    y_true = (eval_df["label"] == "alert").astype(int)
    y_pred = eval_df["anomaly_detected"].astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\nModel performance check -- F1: {f1:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}")

    # ── Push everything to real Firestore ──
    print("\n--- Writing to Firestore ---")
    set_patient_profile(PATIENT_ID, baseline_stats)
    push_readings_to_firestore(df, PATIENT_ID)
    push_risk_scores_to_firestore(scored_df, PATIENT_ID, only_flagged=True)

    print("\nDone. Check your Firebase Console -> Firestore Database to see the data.")


if __name__ == "__main__":
    main()