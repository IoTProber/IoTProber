#!/usr/bin/env python3
"""
Recompute vendor_metrics.csv using only the vendor testset IPs.

- Ground truth: ../../../vendor_testset/test_{TYPE}_1_label.csv  (ip, vendor)
- Predictions : predict_{dev}.json                                (ip, type, top3_vendor)
- Output      : vendor_metrics.csv  (overwrites existing)

Vendor matching is case-insensitive and strips non-alphanumeric characters.
"""

import os
import re
import json
import csv
from collections import Counter

PREDICT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.abspath(os.path.join(PREDICT_DIR, "..", "..", "..", ".."))
VENDOR_TS_DIR = os.path.join(BASE_DIR, "evaluation", "validation", "vendor_testset")

TYPES = [
    "Camera", "Printer", "Scada", "Router", "NAS",
    "NVR", "Power_meter", "Building_Automation", "Medical",
]

LABEL_FILE = {
    "Camera":              "test_CAMERA_1_label.csv",
    "Printer":             "test_PRINTER_1_label.csv",
    "Scada":               "test_SCADA_1_label.csv",
    "Router":              "test_ROUTER_1_label.csv",
    "NAS":                 "test_NAS_1_label.csv",
    "NVR":                 "test_NVR_1_label.csv",
    "Power_meter":         "test_POWER_METER_1_label.csv",
    "Building_Automation": "test_BUILDING_AUTOMATION_1_label.csv",
    "Medical":             "test_MEDICAL_1_label.csv",
}


def normalize_vendor(name):
    if not name or str(name).strip() == "":
        return "unknown"
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return s if s else "unknown"


def load_labels(label_path):
    """Return dict {ip: true_vendor} from label CSV."""
    labels = {}
    with open(label_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ip = row["ip"].strip()
            vendor = row["vendor"].strip()
            if ip:
                labels[ip] = vendor
    return labels


def load_predictions(pred_path):
    """Return dict {ip: top3_vendor_list} from predict JSON."""
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        for entry in json.load(f):
            ip = str(entry.get("ip", "")).strip()
            top3 = entry.get("top3_vendor", [])
            if ip:
                preds[ip] = top3 if isinstance(top3, list) else [top3]
    return preds


def compute_metrics(labels, preds):
    """
    For IPs present in both labels and preds, compute:
      n_ips, identification_rate, top3_accuracy, total_correct,
      top-5 vendor counts among identified IPs.
    """
    matched_ips = [ip for ip in labels if ip in preds]
    n = len(matched_ips)
    if n == 0:
        return None

    n_identified = 0
    n_top3_correct = 0
    top1_counter = Counter()

    for ip in matched_ips:
        true_vendor = labels[ip]
        top3 = preds[ip]
        top1 = top3[0] if top3 else "unknown"

        top1_norm = normalize_vendor(top1)
        identified = top1_norm != "unknown"

        if identified:
            n_identified += 1
            top1_counter[top1] += 1

        true_norm = normalize_vendor(true_vendor)
        top3_norms = [normalize_vendor(v) for v in top3]
        if true_norm in top3_norms:
            n_top3_correct += 1

    identification_rate = round(n_identified / n * 100, 2)
    top3_accuracy = round(n_top3_correct / n * 100, 2)

    top5 = top1_counter.most_common(5)

    return {
        "n_ips":               n,
        "identification_rate": identification_rate,
        "top3_accuracy":       top3_accuracy,
        "total_correct":       n_top3_correct,
        "top5":                top5,
        "n_identified":        n_identified,
    }


def main():
    out_path = os.path.join(PREDICT_DIR, "vendor_metrics.csv")

    header = [
        "device", "n_ips", "identification_rate(%)", "top3_accuracy(%)", "total_correct",
        "top1_vendor", "top1_count", "top1_proportion(%)",
        "top2_vendor", "top2_count", "top2_proportion(%)",
        "top3_vendor", "top3_count", "top3_proportion(%)",
        "top4_vendor", "top4_count", "top4_proportion(%)",
        "top5_vendor", "top5_count", "top5_proportion(%)",
    ]

    rows = []
    for dev in TYPES:
        label_path = os.path.join(VENDOR_TS_DIR, LABEL_FILE[dev])
        pred_path  = os.path.join(PREDICT_DIR, f"predict_{dev}.json")

        if not os.path.exists(label_path):
            print(f"[SKIP] Label file not found: {label_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"[SKIP] Prediction file not found: {pred_path}")
            continue

        labels = load_labels(label_path)
        preds  = load_predictions(pred_path)

        m = compute_metrics(labels, preds)
        if m is None:
            print(f"[WARN] No matching IPs for {dev}")
            continue

        row = [
            dev,
            m["n_ips"],
            m["identification_rate"],
            m["top3_accuracy"],
            m["total_correct"],
        ]
        for rank in range(5):
            if rank < len(m["top5"]):
                vname, vcnt = m["top5"][rank]
                vprop = round(vcnt / m["n_identified"] * 100, 2) if m["n_identified"] else 0
                row += [vname, vcnt, vprop]
            else:
                row += ["", "", 0]

        rows.append(row)
        print(
            f"[{dev}]  n={m['n_ips']}  ident={m['identification_rate']}%  "
            f"top3acc={m['top3_accuracy']}%  correct={m['total_correct']}"
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
