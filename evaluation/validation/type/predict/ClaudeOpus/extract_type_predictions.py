"""
Extract per-IP type predictions from baseline ClaudeOpus predict_{dev}.json files
and save as type_{dev}.csv in this directory.

Columns: ip, true_type, predicted_type
"""

import json
import csv
import os

BASELINE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "baseline", "ClaudeOpus"
)
OUT_DIR = os.path.dirname(__file__)

TYPES = [
    "Camera", "Printer", "Scada", "Router", "NAS",
    "NVR", "Power_meter", "Building_Automation", "Medical"
]

for dev in TYPES:
    src = os.path.join(BASELINE_DIR, f"predict_{dev}.json")
    if not os.path.exists(src):
        print(f"  [SKIP] {src} not found")
        continue

    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)

    out_path = os.path.join(OUT_DIR, f"type_{dev}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ip", "true_type", "predicted_type"])
        for rec in records:
            writer.writerow([rec["ip"], dev, rec["type"]])

    print(f"  Saved type_{dev}.csv  ({len(records)} records)")

print("Done.")
