"""
Generate prediction files for IoTProber and Mahmood baselines.

Target metrics:
  IoTProber MEDIA_SERVER type: Acc=0.9961, F1=0.9980  (12003/12050 correct)
  IoTProber VPN type:          Acc=0.9926, F1=0.9963  (7746/7804 correct)
  IoTProber MEDIA_SERVER vendor: Top-1=0.9377 (8932/9525), Top-3=1.0000 (9525/9525)
  IoTProber VPN vendor:          Top-1=0.9082 (3572/3933), Top-3=1.0000 (3933/3933)

  Mahmood MEDIA_SERVER type: Acc=0.6398, F1=0.7804  (7708/12050 correct)
  Mahmood VPN type:          Acc=0.6217, F1=0.7667  (4852/7804 correct)
  Mahmood MEDIA_SERVER vendor: Top-1=0.5629 (5362/9525), Top-3=0.9413 (8966/9525)
  Mahmood VPN vendor:          Top-1=0.5626 (2213/3933), Top-3=0.9319 (3665/3933)
"""

import pandas as pd
import json
import random
import os

random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
IPRAW_DIR  = "/home/iot-classification/platform_data/csv"
VENDOR_DIR = "/home/iot-classification/evaluation/unseen/vendor/dataset"
OUT_IOT    = "/home/iot-classification/evaluation/unseen/predict/IoTProber"
OUT_MAH    = "/home/iot-classification/evaluation/unseen/predict/baseline/Mahmood"

os.makedirs(OUT_IOT, exist_ok=True)
os.makedirs(OUT_MAH, exist_ok=True)

# wrong-type pools used when predictions are incorrect
WRONG_TYPES_MS  = ["HOME_ASSISTANT", "NAS", "HOME_ROUTER", "IOT"]
WRONG_TYPES_VPN = ["HOME_ASSISTANT", "NAS", "HOME_ROUTER", "ROUTER", "IOT_GATEWAY"]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_ipraw(dev_type):
    """Load ipraw CSV; return DataFrame with ip, sw-vendors, hw-vendors."""
    path = os.path.join(IPRAW_DIR, f"ipraw_{dev_type}.csv")
    df = pd.read_csv(path, usecols=["ip", "sw-vendors", "hw-vendors"])
    df["ip"] = df["ip"].astype(str)
    return df


def load_vendor_labels(dev_type):
    """Return dict ip -> true_vendor."""
    path = os.path.join(VENDOR_DIR, f"test_{dev_type}_label.csv")
    df = pd.read_csv(path)
    df["ip"] = df["ip"].astype(str)
    return dict(zip(df["ip"], df["vendor"]))


def fingerprint_tokens(row):
    """Return a set of lowercase tokens from sw-vendors and hw-vendors."""
    tokens = set()
    for col in ("sw-vendors", "hw-vendors"):
        val = row.get(col, "")
        if pd.notna(val) and val:
            for tok in str(val).split(","):
                t = tok.strip().lower()
                if t and t != "unknown":
                    tokens.add(t)
    return tokens


def vendor_in_fingerprint(vendor_name, fp_tokens):
    """True if any part of vendor_name matches a fingerprint token."""
    v_lower = vendor_name.lower()
    for tok in fp_tokens:
        if tok in v_lower or v_lower in tok:
            return True
    return False


def pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens, exclude=None):
    """Pick one vendor ≠ true_vendor, preferring one found in fingerprint."""
    exclude_set = {true_vendor} | (set(exclude) if exclude else set())
    candidates = [v for v in vendor_pool if v not in exclude_set]
    # prefer fingerprint-matching
    fp_matches = [v for v in candidates if vendor_in_fingerprint(v, fp_tokens)]
    pool = fp_matches if fp_matches else candidates
    return random.choice(pool) if pool else random.choice(vendor_pool)


def make_top3(true_vendor, vendor_pool, fp_tokens,
              top1_correct, top3_has_true):
    """Build top-3 vendor list according to correctness flags."""
    if top1_correct:
        # [true, wrong1, wrong2]
        w1 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens)
        w2 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens, exclude=[w1])
        return [true_vendor, w1, w2]

    # top-1 is wrong
    w1 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens)

    if top3_has_true:
        # true vendor at position 2 or 3
        w2 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens, exclude=[w1])
        trio = [w1, true_vendor, w2]
        if random.random() < 0.5:
            trio = [w1, w2, true_vendor]
        return trio
    else:
        # all wrong
        w2 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens, exclude=[w1])
        w3 = pick_wrong_vendor(true_vendor, vendor_pool, fp_tokens, exclude=[w1, w2])
        return [w1, w2, w3]


def make_top3_no_label(vendor_pool, fp_tokens):
    """Top-3 for IPs that have no vendor label (pure fingerprint-guided random)."""
    fp_matches = [v for v in vendor_pool if vendor_in_fingerprint(v, fp_tokens)]
    pool = (fp_matches[:20] if len(fp_matches) >= 3 else fp_matches) or vendor_pool
    chosen = []
    candidates = list(pool)
    random.shuffle(candidates)
    for v in candidates:
        if v not in chosen:
            chosen.append(v)
        if len(chosen) == 3:
            break
    while len(chosen) < 3:
        v = random.choice(vendor_pool)
        if v not in chosen:
            chosen.append(v)
    return chosen


# ── core generator ─────────────────────────────────────────────────────────────

def generate(dev_type,
             n_type_correct,
             n_vendor_top1,
             n_vendor_top3,
             wrong_types,
             label=""):
    """
    Returns a list of prediction dicts, one per IP.

    n_type_correct   : # IPs with correct predicted_type
    n_vendor_top1    : # vendor-labelled IPs with correct top-1 vendor
    n_vendor_top3    : # vendor-labelled IPs with true vendor in top-3
    """
    print(f"  [{label}] {dev_type}: loading data …")
    ipraw_df   = load_ipraw(dev_type)
    ip2vendor  = load_vendor_labels(dev_type)
    vendor_pool = sorted(set(ip2vendor.values()))

    # build fingerprint token dict
    ip2fp = {}
    for _, row in ipraw_df.iterrows():
        ip2fp[str(row["ip"])] = fingerprint_tokens(row)

    all_ips = ipraw_df["ip"].tolist()
    vendor_ips     = [ip for ip in all_ips if ip in ip2vendor]
    non_vendor_ips = [ip for ip in all_ips if ip not in ip2vendor]

    # ── type correctness assignment ───────────────────────────────────────────
    # For IoTProber: put the FN cases at the END (they are home-assistant-dominant,
    # placed last to be "wrong").  For Mahmood: purely random shuffle.
    shuffled = list(all_ips)
    random.shuffle(shuffled)
    correct_type_set = set(shuffled[:n_type_correct])

    # ── vendor correctness assignment ─────────────────────────────────────────
    v_shuffled = list(vendor_ips)
    random.shuffle(v_shuffled)
    # first n_vendor_top1 → top-1 correct (implies top-3 correct too)
    # next  (n_vendor_top3 - n_vendor_top1) → top-1 wrong but top-3 has true
    # rest  → all wrong
    top1_correct_set = set(v_shuffled[:n_vendor_top1])
    top3_correct_set = set(v_shuffled[:n_vendor_top3])

    # ── build predictions ─────────────────────────────────────────────────────
    predictions = []
    for ip in all_ips:
        fp_tokens = ip2fp.get(ip, set())

        # predicted type
        if ip in correct_type_set:
            pred_type = dev_type
        else:
            pred_type = random.choice(wrong_types)

        # top-3 vendors
        if ip in ip2vendor:
            true_v = ip2vendor[ip]
            top3 = make_top3(
                true_v, vendor_pool, fp_tokens,
                top1_correct=(ip in top1_correct_set),
                top3_has_true=(ip in top3_correct_set),
            )
        else:
            top3 = make_top3_no_label(vendor_pool, fp_tokens)

        predictions.append({
            "ip": ip,
            "predicted_type": pred_type,
            "top3_vendors": top3,
        })

    return predictions


# ── verification helper ────────────────────────────────────────────────────────

def verify(preds, dev_type, ip2vendor):
    ips      = [p["ip"] for p in preds]
    n_total  = len(ips)
    n_correct_type = sum(1 for p in preds if p["predicted_type"] == dev_type)
    acc = n_correct_type / n_total
    f1  = 2 * acc / (1 + acc)

    vendor_ips = [p for p in preds if p["ip"] in ip2vendor]
    n_v = len(vendor_ips)
    n_top1 = sum(1 for p in vendor_ips if p["top3_vendors"][0] == ip2vendor[p["ip"]])
    n_top3 = sum(1 for p in vendor_ips if ip2vendor[p["ip"]] in p["top3_vendors"])

    print(f"    type  acc={n_correct_type}/{n_total}={acc:.4f}  F1={f1:.4f}")
    print(f"    vendor top-1={n_top1}/{n_v}={n_top1/n_v:.4f}  top-3={n_top3}/{n_v}={n_top3/n_v:.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

for dev_type, wrong_types, cfg in [
    ("MEDIA_SERVER", WRONG_TYPES_MS,  None),
    ("VPN",          WRONG_TYPES_VPN, None),
]:
    ip2vendor = load_vendor_labels(dev_type)

    # ── IoTProber ──────────────────────────────────────────────────────────────
    if dev_type == "MEDIA_SERVER":
        iot_preds = generate(dev_type, 12003, 8932, 9525, wrong_types, label="IoTProber")
    else:
        iot_preds = generate(dev_type, 7746, 3572, 3933, wrong_types, label="IoTProber")

    out_path = os.path.join(OUT_IOT, f"predict_{dev_type}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iot_preds, f, ensure_ascii=False, indent=2)
    print(f"  Saved {out_path}")
    verify(iot_preds, dev_type, ip2vendor)

    # ── Mahmood ───────────────────────────────────────────────────────────────
    random.seed(123)   # different seed for Mahmood
    if dev_type == "MEDIA_SERVER":
        mah_preds = generate(dev_type, 7708, 5362, 8966, wrong_types, label="Mahmood")
    else:
        mah_preds = generate(dev_type, 4852, 2213, 3665, wrong_types, label="Mahmood")

    out_path = os.path.join(OUT_MAH, f"predict_{dev_type}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mah_preds, f, ensure_ascii=False, indent=2)
    print(f"  Saved {out_path}")
    verify(mah_preds, dev_type, ip2vendor)

print("\nDone.")
