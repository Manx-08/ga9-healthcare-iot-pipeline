"""
GA9 Healthcare IoT Monitoring System
Step 2: Firebase Data Structure Definition

This defines the exact JSON shape that will be used in Firebase Realtime
Database / Firestore. Every other piece of the system (mobile app,
TinyML output, cloud AI pipeline, React dashboard) must read and write
data in this exact structure, so this file is the single source of truth.

Run this script to:
  1. Print the schema documentation
  2. Generate a sample Firebase-ready JSON file from the simulated dataset
     (useful for manually importing into Firebase to test the dashboard
     before the real hardware/app pipeline exists)
"""

import pandas as pd
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────
# FIREBASE STRUCTURE (Firestore-style, collection/document)
# ─────────────────────────────────────────────────────────────────────────
#
# patients/
#   {patient_id}/
#       profile/
#           name: string
#           age: number
#           condition: string              (e.g. "post-operative cardiac")
#           baseline_hr: number             (set after 7-14 day onboarding)
#           baseline_spo2: number
#           baseline_temp: number
#           baseline_established: boolean
#           caregiver_phone: string          (for Twilio SMS)
#
#       readings/                            <- one doc per sensor reading
#           {reading_id}/
#               timestamp: ISO8601 string
#               heart_rate_bpm: number
#               spo2_percent: number
#               temperature_c: number
#               tinyml_classification: "normal" | "watch" | "alert"   <- Stage 2 output (on-device)
#               source: "device" | "simulated"
#
#       risk_scores/                         <- one doc per cloud AI evaluation (Stage 4 output)
#           {score_id}/
#               timestamp: ISO8601 string
#               risk_score: number (0-100)
#               anomaly_detected: boolean
#               contributing_factors: array of strings   (e.g. ["heart_rate_bpm"])
#               model_version: string
#
#       alerts/                              <- one doc per alert sent (Stage 5 output)
#           {alert_id}/
#               timestamp: ISO8601 string
#               risk_score: number
#               severity: "watch" | "alert"
#               sms_sent: boolean
#               sms_sid: string (Twilio message ID, for tracking)
#               acknowledged: boolean
#               acknowledged_by: string
#
# ─────────────────────────────────────────────────────────────────────────

SCHEMA_DOC = {
    "patients": {
        "{patient_id}": {
            "profile": {
                "name": "string",
                "age": "number",
                "condition": "string",
                "baseline_hr": "number",
                "baseline_spo2": "number",
                "baseline_temp": "number",
                "baseline_established": "boolean",
                "caregiver_phone": "string"
            },
            "readings": {
                "{reading_id}": {
                    "timestamp": "ISO8601 string",
                    "heart_rate_bpm": "number",
                    "spo2_percent": "number",
                    "temperature_c": "number",
                    "tinyml_classification": "normal | watch | alert",
                    "source": "device | simulated"
                }
            },
            "risk_scores": {
                "{score_id}": {
                    "timestamp": "ISO8601 string",
                    "risk_score": "number (0-100)",
                    "anomaly_detected": "boolean",
                    "contributing_factors": "array of strings",
                    "model_version": "string"
                }
            },
            "alerts": {
                "{alert_id}": {
                    "timestamp": "ISO8601 string",
                    "risk_score": "number",
                    "severity": "watch | alert",
                    "sms_sent": "boolean",
                    "sms_sid": "string",
                    "acknowledged": "boolean",
                    "acknowledged_by": "string"
                }
            }
        }
    }
}

def reading_to_firebase_doc(row):
    """Convert one row of the simulated dataset into a Firebase 'readings' document."""
    return {
        "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], datetime) else str(row["timestamp"]),
        "heart_rate_bpm": float(row["heart_rate_bpm"]),
        "spo2_percent": float(row["spo2_percent"]),
        "temperature_c": float(row["temperature_c"]),
        "tinyml_classification": row["label"],  # ground-truth label stands in for TinyML output for now
        "source": "simulated"
    }

def build_sample_firebase_export(csv_path, patient_id, n_samples=200, out_path=None):
    """
    Build a small sample Firebase-ready JSON export from the simulated dataset.
    n_samples is kept small (default 200) because this is just for manually
    seeding Firebase to test the dashboard, not a bulk import of all 40,320 rows.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Take an interesting slice: include one full anomaly window (Day 3 tachycardia)
    window = df[(df["timestamp"] >= "2026-06-03 13:50:00") & (df["timestamp"] <= "2026-06-03 14:30:00")]
    sample = window if len(window) <= n_samples else window.sample(n_samples, random_state=1)
    sample = sample.sort_values("timestamp")

    readings_doc = {}
    for idx, row in sample.iterrows():
        reading_id = f"reading_{idx:06d}"
        readings_doc[reading_id] = reading_to_firebase_doc(row)

    profile_doc = {
        "name": "Test Patient",
        "age": 58,
        "condition": "post-operative cardiac monitoring",
        "baseline_hr": 72,
        "baseline_spo2": 97.5,
        "baseline_temp": 36.6,
        "baseline_established": True,
        "caregiver_phone": "+27000000000"
    }

    firebase_export = {
        "patients": {
            patient_id: {
                "profile": profile_doc,
                "readings": readings_doc
            }
        }
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(firebase_export, f, indent=2)
        print(f"Sample Firebase export saved to: {out_path}")
        print(f"Contains {len(readings_doc)} readings for patient '{patient_id}'")

    return firebase_export


if __name__ == "__main__":
    print("=" * 70)
    print("FIREBASE SCHEMA DOCUMENTATION")
    print("=" * 70)
    print(json.dumps(SCHEMA_DOC, indent=2))

    print("\n" + "=" * 70)
    print("BUILDING SAMPLE FIREBASE EXPORT FROM SIMULATED DATA")
    print("=" * 70)
    build_sample_firebase_export(
        csv_path="C:/Users/byrst/OneDrive/Desktop/Healthcare IoT Monitoring System/data/simulated_patient_vitals.csv",
        patient_id="patient_001",
        n_samples=200,
        out_path="C:/Users/byrst/OneDrive/Desktop/Healthcare IoT Monitoring System/data/sample_firebase_export.json"
    )