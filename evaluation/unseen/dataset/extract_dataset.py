import pandas as pd
import os

LABEL_DIR = "/home/iot-classification/platform_data/csv/label/known"
IPRAW_DIR = "/home/iot-classification/platform_data/csv"
OUT_DIR = "/home/iot-classification/evaluation/unseen/dataset"

DEVICES = ["MEDIA_SERVER", "VPN"]

for dev in DEVICES:
    label_path = os.path.join(LABEL_DIR, f"label_nonempty_{dev}.csv")
    ipraw_path = os.path.join(IPRAW_DIR, f"ipraw_{dev}.csv")
    out_path = os.path.join(OUT_DIR, f"dataset_{dev}.csv")

    print(f"[{dev}] Loading label file...")
    labels = pd.read_csv(label_path)
    ip_set = set(labels["ip"].astype(str))
    print(f"[{dev}] {len(ip_set)} IPs in label file.")

    print(f"[{dev}] Scanning ipraw file in chunks...")
    chunks = []
    total_matched = 0
    for chunk in pd.read_csv(ipraw_path, chunksize=10000, low_memory=False):
        matched = chunk[chunk["ip"].astype(str).isin(ip_set)]
        if not matched.empty:
            chunks.append(matched)
            total_matched += len(matched)

    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        result = result.merge(labels, on="ip", how="left")
        result.to_csv(out_path, index=False)
        print(f"[{dev}] Saved {len(result)} rows to {out_path}")
    else:
        print(f"[{dev}] No matching rows found.")
