#!/usr/bin/env python3
"""
1. Split ClaudeOpus baseline JSON predictions into per-device CSV files
   (vendor_<TYPE>.csv) in evaluation/validation/vendor/predict/ClaudeOpus/.
2. Print a comparison table of vendor metrics across ClaudeOpus, IoTProber, Mahmood.
"""

import os
import json
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
CLAUDE_JSON_DIR = os.path.abspath(
    os.path.join(BASE, "..", "..", "baseline", "ClaudeOpus")
)
CLAUDE_OUT_DIR = os.path.join(BASE, "ClaudeOpus")

TYPE_MAP = {
    "Building_Automation": "BUILDING_AUTOMATION",
    "Camera":              "CAMERA",
    "Medical":             "MEDICAL",
    "NAS":                 "NAS",
    "NVR":                 "NVR",
    "Power_meter":         "POWER_METER",
    "Printer":             "PRINTER",
    "Router":              "ROUTER",
    "Scada":               "SCADA",
}

DISPLAY_ORDER = [
    "CAMERA", "PRINTER", "SCADA", "ROUTER",
    "NAS", "NVR", "POWER_METER", "BUILDING_AUTOMATION", "MEDICAL",
]

METRICS_FILES = {
    "ClaudeOpus":  os.path.join(CLAUDE_JSON_DIR, "vendor_metrics.csv"),
    "IoTProber":   os.path.join(BASE, "IoTProber", "metrics.csv"),
    "Mahmood":     os.path.join(BASE, "Mahmood",   "metrics.csv"),
}

CLAUDE_DEVICE_NORM = {
    "Camera": "CAMERA", "Printer": "PRINTER", "Scada": "SCADA",
    "Router": "ROUTER", "NAS": "NAS", "NVR": "NVR",
    "Power_meter": "POWER_METER", "Building_Automation": "BUILDING_AUTOMATION",
    "Medical": "MEDICAL",
}


def split_claude_predictions():
    """Convert ClaudeOpus JSON predict files → vendor_<TYPE>.csv."""
    os.makedirs(CLAUDE_OUT_DIR, exist_ok=True)
    written = {}

    for dev_name, type_key in TYPE_MAP.items():
        json_path = os.path.join(CLAUDE_JSON_DIR, f"predict_{dev_name}.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {json_path} not found")
            continue

        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)

        out_path = os.path.join(CLAUDE_OUT_DIR, f"vendor_{type_key}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ip", "vendor1", "vendor2", "vendor3"])
            for rec in records:
                ip = str(rec.get("ip", "")).strip()
                top3 = rec.get("top3_vendor", [])
                if not isinstance(top3, list):
                    top3 = [top3]
                while len(top3) < 3:
                    top3.append("")
                w.writerow([ip] + top3[:3])

        written[type_key] = (len(records), out_path)
        print(f"[OK] {dev_name} → {out_path}  ({len(records)} rows)")

    return written


def load_metrics(csv_path, method_name):
    """Return dict {TYPE_KEY: {identification_rate, top3_accuracy, n_ips}}."""
    result = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dev_raw = row["device"].strip()
            type_key = CLAUDE_DEVICE_NORM.get(dev_raw, dev_raw.upper())
            result[type_key] = {
                "identification_rate": float(row["identification_rate(%)"]),
                "top3_accuracy":       float(row["top3_accuracy(%)"]),
                "n_ips":               int(row["n_ips"]),
            }
    return result


def print_comparison(all_metrics):
    """Print a comparison table and compute overall stats."""
    methods = ["ClaudeOpus", "IoTProber", "Mahmood"]

    col_w = 10
    dev_w = 22

    def hline(char="-"):
        return char * (dev_w + 3 + len(methods) * (col_w * 2 + 3))

    header1 = f"{'Device':<{dev_w}} | "
    for m in methods:
        header1 += f"{'--- ' + m + ' ---':^{col_w*2+1}} | "

    header2 = f"{'':^{dev_w}} | "
    for _ in methods:
        header2 += f"{'Ident%':>{col_w}} {'Top3%':>{col_w}} | "

    print("\n" + "=" * len(header1))
    print("  Vendor Metrics Comparison: ClaudeOpus vs IoTProber vs Mahmood")
    print("=" * len(header1))
    print(header1)
    print(header2)
    print(hline())

    totals = {m: {"sum_ips": 0, "sum_weighted_top3": 0.0, "sum_id_rate": 0.0, "cnt": 0}
              for m in methods}

    for dev in DISPLAY_ORDER:
        row_str = f"{dev:<{dev_w}} | "
        for m in methods:
            d = all_metrics[m].get(dev)
            if d:
                row_str += f"{d['identification_rate']:>{col_w}.2f} {d['top3_accuracy']:>{col_w}.2f} | "
                totals[m]["sum_ips"]            += d["n_ips"]
                totals[m]["sum_weighted_top3"]  += d["top3_accuracy"] * d["n_ips"]
                totals[m]["sum_id_rate"]        += d["identification_rate"]
                totals[m]["cnt"]                += 1
            else:
                row_str += f"{'N/A':>{col_w}} {'N/A':>{col_w}} | "
        print(row_str)

    print(hline("="))

    avg_row   = f"{'Avg Identification Rate':<{dev_w}} | "
    total_row = f"{'Overall Top-3 Accuracy':<{dev_w}} | "
    for m in methods:
        t = totals[m]
        avg_id   = t["sum_id_rate"] / t["cnt"] if t["cnt"] else 0
        w_top3   = t["sum_weighted_top3"] / t["sum_ips"] if t["sum_ips"] else 0
        avg_row   += f"{avg_id:>{col_w}.2f} {'':>{col_w}} | "
        total_row += f"{'':>{col_w}} {w_top3:>{col_w}.2f} | "

    print(avg_row)
    print(total_row)
    print("=" * len(header1))

    print("\n[Summary]")
    for m in methods:
        t = totals[m]
        avg_id = t["sum_id_rate"] / t["cnt"] if t["cnt"] else 0
        w_top3 = t["sum_weighted_top3"] / t["sum_ips"] if t["sum_ips"] else 0
        print(f"  {m:12s}  avg_identification_rate={avg_id:.2f}%  "
              f"overall_top3_accuracy={w_top3:.2f}%")
    print()


def main():
    print("\n=== Step 1: Split ClaudeOpus predictions to per-device CSVs ===\n")
    split_claude_predictions()

    print("\n=== Step 2: Load metrics for all three methods ===\n")
    all_metrics = {}
    for method, path in METRICS_FILES.items():
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            all_metrics[method] = {}
        else:
            all_metrics[method] = load_metrics(path, method)
            print(f"[OK] Loaded {method} ({len(all_metrics[method])} device types)")

    print("\n=== Step 3: Comparison Table ===")
    print_comparison(all_metrics)


if __name__ == "__main__":
    main()
