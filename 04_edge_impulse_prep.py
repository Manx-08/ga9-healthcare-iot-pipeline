"""
GA9 Healthcare IoT Monitoring System
Edge Impulse Data Preparation Script

Converts the simulated patient vitals dataset into the CSV format
that Edge Impulse expects for uploading training data.

Edge Impulse CSV format requirements:
- One file per sample window (we use 10-reading = 5-minute windows)
- Columns: timestamp (ms), then sensor values
- Label is set during upload or via filename convention

This script produces:
  - edge_impulse_normal.csv   (normal windows for training)
  - edge_impulse_anomaly.csv  (alert windows for training)
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = r"C:\Users\byrst\OneDrive\Desktop\Healthcare IoT Monitoring System\data\simulated_patient_vitals.csv"
OUTPUT_DIR = r"C:\Users\byrst\OneDrive\Desktop\Healthcare IoT Monitoring System\data"

WINDOW_SIZE = 10       # 10 readings per sample (5 minutes at 30s intervals)
STEP_SIZE = 5          # 50% overlap between windows

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

print(f"Loaded {len(df)} readings")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

normal_rows = []
anomaly_rows = []

for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
    window = df.iloc[start:start + WINDOW_SIZE]

    # Label the window by its dominant label
    label_counts = window["label"].value_counts()
    dominant_label = label_counts.index[0]

    # Build one flat row: timestamp_0, hr_0, spo2_0, temp_0, timestamp_1, hr_1...
    row = {}
    for i, (_, reading) in enumerate(window.iterrows()):
        row[f"timestamp_{i}"] = i * 30000  # convert to milliseconds offset
        row[f"heart_rate_{i}"] = reading["heart_rate_bpm"]
        row[f"spo2_{i}"] = reading["spo2_percent"]
        row[f"temperature_{i}"] = reading["temperature_c"]

    if dominant_label == "normal":
        normal_rows.append(row)
    elif dominant_label in ["alert", "watch"]:
        anomaly_rows.append(row)

normal_df = pd.DataFrame(normal_rows)
anomaly_df = pd.DataFrame(anomaly_rows)

# Balance the dataset: downsample normal to 3x the anomaly count
# (too many normals vs anomalies causes the model to just predict "normal" always)
max_normal = min(len(normal_df), len(anomaly_df) * 3)
normal_df = normal_df.sample(n=max_normal, random_state=42)

normal_path = os.path.join(OUTPUT_DIR, "edge_impulse_normal.csv")
anomaly_path = os.path.join(OUTPUT_DIR, "edge_impulse_anomaly.csv")

normal_df.to_csv(normal_path, index=False)
anomaly_df.to_csv(anomaly_path, index=False)

print(f"Normal windows: {len(normal_df)} -> saved to {normal_path}")
print(f"Anomaly windows: {len(anomaly_df)} -> saved to {anomaly_path}")
print(f"\nReady to upload to Edge Impulse.")
print(f"Upload edge_impulse_normal.csv with label: 'normal'")
print(f"Upload edge_impulse_anomaly.csv with label: 'anomaly'")