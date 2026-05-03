"""
Data preparation script for LLaMA 3.1 vendor classification fine-tuning.
Merges ipraw features with vendor labels, filters Unknown vendors,
and saves processed data for fine-tuning.

Usage:
    python prepare_data.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from util import load_all_dev_labels

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
IPRAW_DIR = os.path.join(BASE_PATH, "platform_data", "csv", "local", "1")
LABEL_DIR = os.path.join(BASE_PATH, "platform_data", "csv", "label")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dev_list = load_all_dev_labels()
    if not dev_list:
        print("No device labels found!")
        return

    total_samples = 0
    for dev in dev_list:
        label_path = os.path.join(LABEL_DIR, f"label_{dev}.csv")
        ipraw_path = os.path.join(IPRAW_DIR, f"ipraw_{dev}.csv")
        output_path = os.path.join(OUTPUT_DIR, f"finetune_data_{dev}.csv")

        if not os.path.exists(label_path):
            print(f"[SKIP] label_{dev}.csv not found, skipping {dev}.")
            continue

        if not os.path.exists(ipraw_path):
            print(f"[SKIP] ipraw_{dev}.csv not found, skipping {dev}.")
            continue

        print(f"Processing {dev} ...")

        label_df = pd.read_csv(label_path, dtype=str)
        ipraw_df = pd.read_csv(ipraw_path, dtype=str, low_memory=False)

        merged = ipraw_df.merge(label_df[["ip", "vendor"]], on="ip", how="inner")

        merged = merged[merged["vendor"].str.strip() != "Unknown"]

        print(f"  {dev}: {len(merged)} samples after filtering Unknown vendors")

        if len(merged) > 0:
            merged.to_csv(output_path, index=False)
            total_samples += len(merged)
            print(f"  Saved to {output_path}")
        else:
            print(f"  No valid samples for {dev}, skipping save.")

    print(f"\nData preparation complete! Total samples: {total_samples}")


if __name__ == "__main__":
    prepare_data()
