import pandas as pd
import os
import glob
import time

base_dir = r'e:\iot-classification'
features_45_dir = os.path.join(base_dir, 'evaluation', 'validation', '45_features')
features_46_dir = os.path.join(base_dir, 'evaluation', 'validation', '46_features')
nofilter_unknown_dir = os.path.join(base_dir, 'evaluation', 'validation', 'nofilter_unknown')
validation_dir = os.path.join(base_dir, 'evaluation', 'validation')
csv_dir = os.path.join(base_dir, 'platform_data', 'csv')
filter_dir = os.path.join(csv_dir, 'filter')
label_dir = os.path.join(csv_dir, 'label')

# Ensure output dirs exist
os.makedirs(features_46_dir, exist_ok=True)
os.makedirs(nofilter_unknown_dir, exist_ok=True)

# Get device names from 45_features
test_files = glob.glob(os.path.join(features_45_dir, 'test_*_1.csv'))
devices = []
for f in test_files:
    fname = os.path.basename(f)
    # test_{dev}_1.csv -> extract dev
    dev = fname[len('test_'):-len('_1.csv')]
    devices.append(dev)

devices.sort()
print(f"Found {len(devices)} devices: {devices}")

CHUNK_SIZE = 10000

for dev in devices:
    print(f"\n{'='*60}")
    print(f"Processing: {dev}")
    print(f"{'='*60}")

    t0 = time.time()

    # Step 0: Read test IPs from 45_features
    test_file_45 = os.path.join(features_45_dir, f'test_{dev}_1.csv')
    print(f"[Step 0] Reading test IPs from 45_features...")
    test_ips = set()
    for chunk in pd.read_csv(test_file_45, chunksize=CHUNK_SIZE, usecols=['ip'], dtype=str):
        test_ips.update(chunk['ip'].dropna().values)
    print(f"  -> {len(test_ips)} unique IPs ({time.time()-t0:.1f}s)")

    # ============================================================
    # Task 1: Match IPs in ipraw_{dev}.csv -> 46_features
    # ============================================================
    ipraw_file = os.path.join(csv_dir, f'ipraw_{dev}.csv')
    output_46 = os.path.join(features_46_dir, f'test_{dev}_1.csv')

    if os.path.exists(ipraw_file):
        print(f"[Task 1] Matching in ipraw_{dev}.csv -> 46_features...")
        t1 = time.time()
        first_chunk = True
        count_46 = 0
        for chunk in pd.read_csv(ipraw_file, chunksize=CHUNK_SIZE, dtype=str):
            matched = chunk[chunk['ip'].isin(test_ips)]
            if len(matched) > 0:
                matched.to_csv(output_46, mode='a' if not first_chunk else 'w',
                               header=first_chunk, index=False)
                if first_chunk:
                    first_chunk = False
                count_46 += len(matched)

        if first_chunk:
            print(f"  -> WARNING: No matching IPs found!")
        else:
            header = pd.read_csv(output_46, nrows=0).columns
            print(f"  -> {count_46} rows, {len(header)} columns ({time.time()-t1:.1f}s)")
            assert len(header) == 46, f"Expected 46 columns, got {len(header)}"
    else:
        print(f"  -> WARNING: {ipraw_file} not found!")

    # ============================================================
    # Task 2: Match IPs in filter/ipraw_{dev}.csv -> nofilter_unknown
    # ============================================================
    filter_file = os.path.join(filter_dir, f'ipraw_{dev}.csv')
    output_nofilter = os.path.join(nofilter_unknown_dir, f'test_{dev}_1.csv')
    output_label_nfu = os.path.join(nofilter_unknown_dir, f'label_{dev}_1.csv')

    if os.path.exists(filter_file):
        print(f"[Task 2] Matching in filter/ipraw_{dev}.csv -> nofilter_unknown...")
        t2 = time.time()
        first_chunk = True
        count_filter = 0
        filter_matched_ips = set()
        for chunk in pd.read_csv(filter_file, chunksize=CHUNK_SIZE, dtype=str):
            matched = chunk[chunk['ip'].isin(test_ips)]
            if len(matched) > 0:
                matched.to_csv(output_nofilter, mode='a' if not first_chunk else 'w',
                               header=first_chunk, index=False)
                if first_chunk:
                    first_chunk = False
                count_filter += len(matched)
                filter_matched_ips.update(matched['ip'].values)

        if first_chunk:
            print(f"  -> WARNING: No matching IPs found in filter!")
            continue

        print(f"  -> {count_filter} rows, {len(filter_matched_ips)} unique IPs ({time.time()-t2:.1f}s)")

        # Find labels for matched IPs (label files are label_{dev}.csv)
        label_file = os.path.join(label_dir, f'label_{dev}.csv')
        if os.path.exists(label_file):
            print(f"  Reading labels from label_{dev}.csv...")
            labels_df = pd.read_csv(label_file, dtype=str)
            matched_labels = labels_df[labels_df['ip'].isin(filter_matched_ips)]
            matched_labels.to_csv(output_label_nfu, index=False)
            print(f"  -> {len(matched_labels)} labels saved to label_{dev}_1.csv")

            # ============================================================
            # Task 3: Filter out Unknown -> evaluation/validation/
            # ============================================================
            known_labels = matched_labels[matched_labels['vendor'] != 'Unknown']
            known_ips = set(known_labels['ip'].values)
            unknown_count = len(filter_matched_ips) - len(known_ips)
            print(f"[Task 3] Filtering Unknown: {len(filter_matched_ips)} total, "
                  f"{unknown_count} Unknown, {len(known_ips)} known")

            output_val_test = os.path.join(validation_dir, f'test_{dev}_1.csv')
            output_val_label = os.path.join(validation_dir, f'test_label_{dev}.csv')

            # Re-read nofilter_unknown test file and filter by known IPs
            first_chunk = True
            count_known = 0
            for chunk in pd.read_csv(output_nofilter, chunksize=CHUNK_SIZE, dtype=str):
                matched = chunk[chunk['ip'].isin(known_ips)]
                if len(matched) > 0:
                    matched.to_csv(output_val_test, mode='a' if not first_chunk else 'w',
                                   header=first_chunk, index=False)
                    if first_chunk:
                        first_chunk = False
                    count_known += len(matched)

            known_labels.to_csv(output_val_label, index=False)
            print(f"  -> {count_known} rows saved to validation/test_{dev}_1.csv")
            print(f"  -> {len(known_labels)} labels saved to validation/test_label_{dev}.csv")
        else:
            print(f"  -> WARNING: Label file not found: {label_file}")
    else:
        print(f"  -> WARNING: Filter file not found for {dev}, skipping Tasks 2 & 3")

    print(f"Device {dev} done in {time.time()-t0:.1f}s")

print("\n" + "=" * 60)
print("All devices processed!")
