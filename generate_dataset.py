"""
GA9 Healthcare IoT Monitoring System
Step 1: Simulated Patient Vital Signs Dataset Generator

Generates a realistic 14-day dataset of patient vital signs (HR, SpO2,
temperature) sampled every 30 seconds, with injected anomaly events at
known timestamps. This dataset is used to:
  1. Build and test the Firebase data structure
  2. Train and validate the Isolation Forest anomaly detection model
  3. Provide labelled data for the TinyML Edge Impulse model later

Each patient has their own baseline (resting HR, normal SpO2, normal temp)
so that "abnormal" is defined relative to the INDIVIDUAL, not a fixed
global threshold. This is the core idea behind your personalised AI layer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────

SAMPLE_INTERVAL_SECONDS = 30
DAYS = 14
SAMPLES_PER_DAY = int(24 * 60 * 60 / SAMPLE_INTERVAL_SECONDS)  # 2880 samples/day
TOTAL_SAMPLES = SAMPLES_PER_DAY * DAYS

PATIENT_ID = "patient_001"

# Individual baseline for this patient (this is what makes monitoring "personalised")
BASELINE_HR = 72        # resting heart rate, bpm
BASELINE_SPO2 = 97.5    # blood oxygen saturation, %
BASELINE_TEMP = 36.6    # body temperature, deg C

# ── Anomaly events to inject (day, start_hour, duration_minutes, type) ────
# These simulate real clinical deterioration scenarios for testing
ANOMALY_EVENTS = [
    {"day": 3,  "start_hour": 14, "duration_min": 20, "type": "tachycardia"},     # HR spike
    {"day": 5,  "start_hour": 3,  "duration_min": 35, "type": "desaturation"},    # SpO2 drop
    {"day": 7,  "start_hour": 22, "duration_min": 15, "type": "fever_onset"},     # temp rise
    {"day": 9,  "start_hour": 9,  "duration_min": 25, "type": "bradycardia"},     # HR drop
    {"day": 11, "start_hour": 17, "duration_min": 40, "type": "combined_crisis"}, # HR+SpO2 together
    {"day": 13, "start_hour": 6,  "duration_min": 18, "type": "desaturation"},
]

# ── Generate timestamps ──────────────────────────────────────────────────

start_time = datetime(2026, 6, 1, 0, 0, 0)
timestamps = [start_time + timedelta(seconds=i * SAMPLE_INTERVAL_SECONDS) for i in range(TOTAL_SAMPLES)]

# ── Generate baseline signals with natural variation ───────────────────────

def circadian_modifier(hour):
    """Heart rate and temp naturally dip at night, rise slightly in afternoon."""
    return np.sin((hour - 6) / 24 * 2 * np.pi)

hr = np.zeros(TOTAL_SAMPLES)
spo2 = np.zeros(TOTAL_SAMPLES)
temp = np.zeros(TOTAL_SAMPLES)
labels = ["normal"] * TOTAL_SAMPLES  # ground truth label for later model training

for i, ts in enumerate(timestamps):
    hour = ts.hour + ts.minute / 60

    # Natural circadian variation + small random noise
    hr[i] = BASELINE_HR + 4 * circadian_modifier(hour) + np.random.normal(0, 2.0)
    spo2[i] = BASELINE_SPO2 + np.random.normal(0, 0.5)
    temp[i] = BASELINE_TEMP + 0.3 * circadian_modifier(hour) + np.random.normal(0, 0.1)

# Clip to physiologically plausible ranges
hr = np.clip(hr, 45, 180)
spo2 = np.clip(spo2, 80, 100)
temp = np.clip(temp, 34.5, 41.0)

# ── Inject anomaly events ───────────────────────────────────────────────

def day_hour_to_index(day, hour, minute=0):
    elapsed_seconds = (day - 1) * 86400 + hour * 3600 + minute * 60
    return int(elapsed_seconds / SAMPLE_INTERVAL_SECONDS)

for event in ANOMALY_EVENTS:
    start_idx = day_hour_to_index(event["day"], event["start_hour"])
    duration_samples = int(event["duration_min"] * 60 / SAMPLE_INTERVAL_SECONDS)
    end_idx = min(start_idx + duration_samples, TOTAL_SAMPLES)

    for i in range(start_idx, end_idx):
        # Ramp-up/ramp-down profile so the anomaly doesn't look like a sharp glitch
        progress = (i - start_idx) / max(duration_samples, 1)
        ramp = np.sin(progress * np.pi)  # 0 -> 1 -> 0 shape

        if event["type"] == "tachycardia":
            hr[i] += 45 * ramp + np.random.normal(0, 3)
            labels[i] = "alert"
        elif event["type"] == "bradycardia":
            hr[i] -= 25 * ramp + np.random.normal(0, 2)
            labels[i] = "alert"
        elif event["type"] == "desaturation":
            spo2[i] -= 12 * ramp + np.random.normal(0, 1)
            labels[i] = "alert"
        elif event["type"] == "fever_onset":
            temp[i] += 2.2 * ramp + np.random.normal(0, 0.15)
            labels[i] = "watch" if ramp < 0.6 else "alert"
        elif event["type"] == "combined_crisis":
            hr[i] += 35 * ramp
            spo2[i] -= 10 * ramp
            labels[i] = "alert"

    # Mark a short "watch" buffer zone just before/after the alert window
    buffer_samples = int(5 * 60 / SAMPLE_INTERVAL_SECONDS)  # 5 min
    for i in range(max(0, start_idx - buffer_samples), start_idx):
        if labels[i] == "normal":
            labels[i] = "watch"
    for i in range(end_idx, min(TOTAL_SAMPLES, end_idx + buffer_samples)):
        if labels[i] == "normal":
            labels[i] = "watch"

# Re-clip after anomaly injection
hr = np.clip(hr, 40, 200)
spo2 = np.clip(spo2, 70, 100)
temp = np.clip(temp, 34.0, 42.0)

# ── Assemble DataFrame ──────────────────────────────────────────────────

df = pd.DataFrame({
    "patient_id": PATIENT_ID,
    "timestamp": timestamps,
    "heart_rate_bpm": np.round(hr, 1),
    "spo2_percent": np.round(spo2, 1),
    "temperature_c": np.round(temp, 2),
    "label": labels
})

# ── Save dataset ─────────────────────────────────────────────────────────

output_path = "C:/Users/byrst/OneDrive\Desktop/Healthcare IoT Monitoring System/data/simulated_patient_vitals.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} samples over {DAYS} days")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nSaved to: {output_path}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nSample anomaly window (Day 3 tachycardia event):")
print(df[(df['timestamp'] >= datetime(2026,6,3,13,55)) & (df['timestamp'] <= datetime(2026,6,3,14,25))].head(10))