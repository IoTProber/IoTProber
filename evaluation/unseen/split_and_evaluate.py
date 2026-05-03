"""
Split prediction results into type / vendor CSVs and compute metrics.

Sources
-------
IoTProber  : predict/IoTProber/predict_{dev}.json
Mahmood    : predict/baseline/Mahmood/predict_{dev}.json
ClaudeOpus : predict/baseline/ClaudeOpus/all_result_type_{dev}.csv

Outputs (written under evaluation/unseen/)
------------------------------------------
type/
  type_predictions_{method}_{dev}.csv   — ip, true_type, pred_type
vendor/
  vendor_predictions_{method}_{dev}.csv — ip, true_vendor, pred_top1, pred_top2, pred_top3
metrics.csv   — one row per (method, device_type) with all six metric values
"""

import os, re, json
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/home/iot-classification/evaluation/unseen"
IPRAW_DIR  = "/home/iot-classification/platform_data/csv"
VENDOR_DIR = os.path.join(BASE, "vendor", "dataset")
PREDICT    = os.path.join(BASE, "predict")

TYPE_OUT   = os.path.join(BASE, "type")
VENDOR_OUT = os.path.join(BASE, "vendor")

os.makedirs(TYPE_OUT,   exist_ok=True)
os.makedirs(VENDOR_OUT, exist_ok=True)

DEVICE_TYPES = ["MEDIA_SERVER", "VPN"]

METHODS = {
    "IoTProber": {
        "kind": "json",
        "path": lambda dev: os.path.join(PREDICT, "IoTProber", f"predict_{dev}.json"),
    },
    "Mahmood": {
        "kind": "json",
        "path": lambda dev: os.path.join(PREDICT, "baseline", "Mahmood", f"predict_{dev}.json"),
    },
    "ClaudeOpus": {
        "kind": "csv",
        "path": lambda dev: os.path.join(
            PREDICT, "baseline", "ClaudeOpus", f"all_result_type_{dev}.csv"
        ),
    },
}

# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_vendor(name):
    if pd.isna(name) or str(name).strip() == "":
        return "unknown"
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return s if s else "unknown"


def load_vendor_labels(dev):
    path = os.path.join(VENDOR_DIR, f"test_{dev}_label.csv")
    df = pd.read_csv(path)
    df["ip"] = df["ip"].astype(str)
    return dict(zip(df["ip"], df["vendor"]))


def load_all_ips(dev):
    """All IPs in ipraw CSV (= full type test set)."""
    path = os.path.join(IPRAW_DIR, f"ipraw_{dev}.csv")
    df = pd.read_csv(path, usecols=["ip"])
    return df["ip"].astype(str).tolist()


def load_predictions_json(dev, path):
    """
    Returns (type_df, vendor_df).
    type_df  : ip, true_type, pred_type
    vendor_df: ip, pred_top1, pred_top2, pred_top3
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows_t, rows_v = [], []
    for r in records:
        ip = str(r["ip"])
        rows_t.append({"ip": ip, "true_type": dev, "pred_type": r["predicted_type"]})
        top3 = r.get("top3_vendors", [None, None, None])
        rows_v.append({
            "ip": ip,
            "pred_top1": top3[0] if len(top3) > 0 else None,
            "pred_top2": top3[1] if len(top3) > 1 else None,
            "pred_top3": top3[2] if len(top3) > 2 else None,
        })

    return pd.DataFrame(rows_t), pd.DataFrame(rows_v)


def load_predictions_csv(dev, path):
    """
    Returns (type_df, vendor_df) from ClaudeOpus all_result_type CSV.
    type_df  : ip, true_type, pred_type
    vendor_df: ip, pred_top1, pred_top2, pred_top3
    """
    df = pd.read_csv(path)
    df["ip"] = df["ip"].astype(str)

    type_df = df[["ip", "true_type", "pred_type"]].copy()

    vendor_df = df[["ip"]].copy()
    vendor_df["pred_top1"] = df.get("vendor_top1", pd.Series(dtype=str))
    vendor_df["pred_top2"] = df.get("vendor_top2", pd.Series(dtype=str))
    vendor_df["pred_top3"] = df.get("vendor_top3", pd.Series(dtype=str))

    return type_df, vendor_df


def compute_type_metrics(type_df, dev):
    """Binary evaluation: all test IPs belong to `dev`, so FP=0, precision=1.0.
    recall = accuracy = TP / (TP+FN)
    F1 = 2 * precision * recall / (precision + recall) = 2*acc / (1+acc)
    """
    n = len(type_df)
    n_correct = (type_df["pred_type"] == dev).sum()
    acc = n_correct / n
    macro_f1 = 2 * acc / (1 + acc) if acc > 0 else 0.0
    return round(acc, 4), round(macro_f1, 4)


def compute_vendor_metrics(vendor_df, ip2vendor):
    """Fuzzy-normalized top-1 / top-3 accuracy over labelled IPs."""
    df = vendor_df[vendor_df["ip"].isin(ip2vendor)].copy()
    if df.empty:
        return 0.0, 0.0

    df["true_v_n"] = df["ip"].map(ip2vendor).apply(normalize_vendor)
    df["top1_n"]   = df["pred_top1"].apply(normalize_vendor)
    df["top2_n"]   = df["pred_top2"].apply(normalize_vendor)
    df["top3_n"]   = df["pred_top3"].apply(normalize_vendor)

    n = len(df)
    top1 = (df["true_v_n"] == df["top1_n"]).sum() / n
    top3 = (
        (df["true_v_n"] == df["top1_n"]) |
        (df["true_v_n"] == df["top2_n"]) |
        (df["true_v_n"] == df["top3_n"])
    ).sum() / n

    return round(top1, 4), round(top3, 4)


# ── main loop ─────────────────────────────────────────────────────────────────

metrics_rows = []

for method, cfg in METHODS.items():
    for dev in DEVICE_TYPES:
        path = cfg["path"](dev)
        if not os.path.exists(path):
            print(f"[SKIP] Missing: {path}")
            continue

        print(f"Processing {method} / {dev} …")

        # load
        if cfg["kind"] == "json":
            type_df, vendor_df = load_predictions_json(dev, path)
        else:
            type_df, vendor_df = load_predictions_csv(dev, path)

        # ── type: add true_vendor for reference where available ───────────────
        ip2vendor = load_vendor_labels(dev)
        vendor_df["true_vendor"] = vendor_df["ip"].map(ip2vendor)

        # reorder vendor columns
        vendor_df = vendor_df[["ip", "true_vendor", "pred_top1", "pred_top2", "pred_top3"]]

        # ── save type CSV ──────────────────────────────────────────────────────
        type_out = os.path.join(TYPE_OUT, f"type_predictions_{method}_{dev}.csv")
        type_df.to_csv(type_out, index=False)

        # ── save vendor CSV ────────────────────────────────────────────────────
        vendor_out = os.path.join(VENDOR_OUT, f"vendor_predictions_{method}_{dev}.csv")
        vendor_df.to_csv(vendor_out, index=False)

        # ── compute metrics ────────────────────────────────────────────────────
        acc, f1 = compute_type_metrics(type_df, dev)
        top1, top3 = compute_vendor_metrics(vendor_df, ip2vendor)

        print(f"  type  acc={acc:.4f}  macro-F1={f1:.4f}")
        print(f"  vendor top-1={top1:.4f}  top-3={top3:.4f}")

        metrics_rows.append({
            "method":         method,
            "device_type":    dev,
            "type_macro_f1":  f1,
            "type_accuracy":  acc,
            "vendor_top1_acc": top1,
            "vendor_top3_acc": top3,
        })

# ── save metrics.csv ──────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_rows, columns=[
    "method", "device_type",
    "type_macro_f1", "type_accuracy",
    "vendor_top1_acc", "vendor_top3_acc",
])
metrics_path = os.path.join(BASE, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nSaved metrics → {metrics_path}")
print(metrics_df.to_string(index=False))
