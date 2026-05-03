"""
SVM-based IoT Device Classification
====================================
Classifies devices into (device_type-vendor) combined labels.

Pipeline:
  1. Load training IPs + vendor labels from vendor-50/
  2. Retrieve IP fingerprints from ipraw_{device_type}.csv
  3. Feature engineering: TF-IDF on text columns, one-hot on categorical
  4. Train SVM with kernel / C / gamma tuning via GridSearchCV
  5. Evaluate: per-fold hinge loss, accuracy, precision, recall, F1
  6. Save per-device-type prediction results to SVM_train/predict/
"""

import os, sys, time, json, re, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hinge_loss, classification_report,
)
from scipy.sparse import hstack, csr_matrix
import joblib

warnings.filterwarnings("ignore")

# ────────────────────────── Paths ──────────────────────────
VENDOR50_DIR = r"e:\iot-classification\evaluation\drift\CADE\dataset\train_SVM\vendor-50"
IPRAW_DIR    = r"e:\iot-classification\platform_data\csv"
OUTPUT_DIR       = r"e:\iot-classification\evaluation\drift\CADE\SVM_train"
PREDICT_SUBDIR   = os.path.join(OUTPUT_DIR, "predict")
DRIFT_FILTER_DIR = r"e:\iot-classification\platform_data\2026-04-06\filter"
TEST_SVM_DIR     = r"e:\iot-classification\evaluation\drift\CADE\dataset\test_SVM"
SVM_TEST_DIR         = r"e:\iot-classification\evaluation\drift\CADE\SVM_test"
TEST_PREDICT_SUBDIR  = os.path.join(SVM_TEST_DIR, "predict")
MODEL_DIR = r"e:\iot-classification\evaluation\drift\CADE\SVM_model"
MODEL_BUNDLE = os.path.join(MODEL_DIR, "svm_bundle.pkl")
VALIDATION_DIR     = r"e:\iot-classification\evaluation\drift\CADE\SVM_validation"
CADE_DETECTED      = os.path.join(VALIDATION_DIR, "CADE", "detected_ips.csv")
IOTPROBER_DETECTED = os.path.join(VALIDATION_DIR, "IoTProber", "detected_ips.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICT_SUBDIR, exist_ok=True)
os.makedirs(SVM_TEST_DIR, exist_ok=True)
os.makedirs(TEST_PREDICT_SUBDIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE_TYPES = [
    "BUILDING_AUTOMATION", "CAMERA", "MEDIA_SERVER", "MEDICAL",
    "NAS", "NVR", "POWER_METER", "PRINTER", "ROUTER", "SCADA", "VPN",
]

# ────────────────────────── Column config ──────────────────────────
# Exact 25 features from local_used_features.txt  (+ip)
USE_COLS = [
    "ip",
    "as-asn", "as-name", "as-bgp_prefix", "as-country_code",
    "whois-network-handle", "whois-network-name",
    "whois-organization-handle", "whois-organization-name",
    "os-vendor", "os-product", "os-version",
    "sw-vendors", "sw-products", "sw-versions",
    "hw-vendors", "hw-products", "hw-versions",
    "service-distribution",
    "http-bodys", "http-tags", "http-favicon-urls",
    "cert-subjects", "cert-issuers", "tls-versions",
    "dns-reverse",
]

# Treat as TF-IDF documents
TEXT_FEATURES = [
    "as-name", "as-bgp_prefix",
    "whois-network-name", "whois-organization-name",
    "os-version",
    "sw-vendors", "sw-products", "sw-versions",
    "hw-vendors", "hw-products", "hw-versions",
    "service-distribution",
    "http-bodys", "http-tags", "http-favicon-urls",
    "cert-subjects", "cert-issuers", "tls-versions",
    "dns-reverse",
]

# Treat as one-hot categorical
CAT_FEATURES = [
    "as-country_code",
    "whois-network-handle", "whois-organization-handle",
    "os-vendor", "os-product",
]

CHUNK_SIZE = 80_000  # rows per chunk when scanning ipraw


# ────────────────────────── Helpers ──────────────────────────
VENDOR_ALIASES = {
    "electroind": "electroindustries",
}


def _normalize_vendor(v):
    """Strip all non-alphanumeric chars and lowercase."""
    return re.sub(r'[^a-z0-9]', '', v.lower())


def normalize_label(label):
    """Normalize a label to canonical form: lowercase device_type + normalized vendor."""
    parts = label.split("-", 1)
    if len(parts) != 2:
        return label.lower()
    dt, vendor = parts
    nv = _normalize_vendor(vendor)
    nv = VENDOR_ALIASES.get(nv, nv)
    return dt.lower() + "-" + nv


def label_match(label, label_pred):
    """Case-insensitive match; if device_type matches, also fuzzy-match vendor
    by stripping all symbols and applying alias mapping."""
    return normalize_label(label) == normalize_label(label_pred)


def comma_tokenizer(text):
    """Split comma-separated fingerprint fields into tokens."""
    return [t.strip() for t in str(text).split(",") if t.strip()]


# ────────────────────────── 1. Load labels ──────────────────────────
def load_labels():
    frames = []
    for dt in DEVICE_TYPES:
        df = pd.read_csv(os.path.join(VENDOR50_DIR, f"{dt}.csv"))
        df["device_type"] = dt
        df["label"] = dt + "-" + df["vendor"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ────────────────────────── 2. Load fingerprints ──────────────────────────
def load_fingerprints(label_df):
    all_fps = []
    for dt in DEVICE_TYPES:
        ip_set = set(label_df.loc[label_df["device_type"] == dt, "ip"])
        ipraw_path = os.path.join(IPRAW_DIR, f"ipraw_{dt}.csv")
        print(f"  {dt:30s} ({len(ip_set):4d} IPs) ... ", end="", flush=True)
        t0 = time.time()
        parts, found = [], 0
        for chunk in pd.read_csv(
            ipraw_path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False
        ):
            matched = chunk[chunk["ip"].isin(ip_set)]
            if len(matched):
                parts.append(matched)
                found += len(matched)
            if found >= len(ip_set):
                break
        if parts:
            fp = pd.concat(parts).drop_duplicates("ip", keep="first")
            fp["device_type"] = dt
            all_fps.append(fp)
            print(f"found {len(fp):4d}  ({time.time()-t0:.1f}s)")
        else:
            print("NONE found!")
    return pd.concat(all_fps, ignore_index=True)


# ────────────────────────── 3. Feature engineering ──────────────────────────
# http-bodys is very large text; cap its TF-IDF vocab to avoid noise
_TFIDF_MAX = {"http-bodys": 200, "http-tags": 200, "http-favicon-urls": 100}
_TFIDF_DEFAULT = 300


def build_features(df):
    """Return sparse feature matrix X, fitted vectorizers, and cat columns map."""
    matrices = []

    # 3-a. TF-IDF on text columns
    vectorizers = {}
    cat_columns_map = {}
    for col in TEXT_FEATURES:
        df[col] = df[col].fillna("").astype(str)
        max_feat = _TFIDF_MAX.get(col, _TFIDF_DEFAULT)
        vec = TfidfVectorizer(
            tokenizer=comma_tokenizer,
            token_pattern=None,
            max_features=max_feat,
            sublinear_tf=True,
        )
        m = vec.fit_transform(df[col])
        matrices.append(m)
        vectorizers[col] = vec
        print(f"    tfidf  {col:30s} -> {m.shape[1]:4d} dims")

    # 3-b. One-hot on categorical columns
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("UNK").astype(str)
        dummies = pd.get_dummies(df[col], prefix=col)
        cat_columns_map[col] = list(dummies.columns)
        m = csr_matrix(dummies.values.astype(np.float32))
        matrices.append(m)
        print(f"    onehot {col:30s} -> {m.shape[1]:4d} dims")

    # 3-c. Numeric: as-asn
    asn = pd.to_numeric(df["as-asn"], errors="coerce").fillna(0).values.reshape(-1, 1)
    matrices.append(csr_matrix(asn.astype(np.float32)))
    print(f"    numeric as-asn{' ':24s} ->    1 dim")

    X = hstack(matrices, format="csr")
    print(f"  Final feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    return X, vectorizers, cat_columns_map


# ────────────────────────── Transform for test set ──────────────────────────
def transform_features(df, vectorizers, cat_columns_map):
    """Transform a dataframe using fitted vectorizers and stored category columns."""
    matrices = []
    for col in TEXT_FEATURES:
        df[col] = df[col].fillna("").astype(str)
        matrices.append(vectorizers[col].transform(df[col]))
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("UNK").astype(str)
        dummies = pd.get_dummies(df[col], prefix=col)
        dummies = dummies.reindex(columns=cat_columns_map[col], fill_value=0)
        matrices.append(csr_matrix(dummies.values.astype(np.float32)))
    asn = pd.to_numeric(df["as-asn"], errors="coerce").fillna(0).values.reshape(-1, 1)
    matrices.append(csr_matrix(asn.astype(np.float32)))
    return hstack(matrices, format="csr")


# ────────────────────────── 4-5. Train & evaluate ──────────────────────────
def train_evaluate(X, y, le):
    # Scale
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)

    n_classes = len(le.classes_)
    print(f"  Samples={X.shape[0]}  Features={X.shape[1]}  Classes={n_classes}")

    # --- Hyperparameter grid ---
    param_grid = [
        {"kernel": ["rbf"],    "C": [1, 10, 100], "gamma": ["scale", 0.01]},
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["poly"],   "C": [1, 10],      "degree": [3], "gamma": ["scale"]},
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm = SVC(
        decision_function_shape="ovr",
        class_weight="balanced",
        random_state=42,
        cache_size=2000,
    )

    grid = GridSearchCV(
        svm, param_grid, cv=skf, scoring="f1_macro",
        n_jobs=-1, verbose=2, refit=True,
    )

    print("\n  Running GridSearchCV ...")
    t0 = time.time()
    grid.fit(X, y)
    print(f"  GridSearch done in {time.time()-t0:.0f}s")

    best = grid.best_estimator_
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")

    # --- Per-fold loss (hinge) with best params ---
    print("\n  Per-fold metrics (best params):")
    fold_metrics = []
    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = SVC(
            **grid.best_params_,
            decision_function_shape="ovr",
            class_weight="balanced",
            random_state=42,
            cache_size=2000,
        )
        m.fit(X_tr, y_tr)
        y_p = m.predict(X_val)
        dec = m.decision_function(X_val)
        h = hinge_loss(y_val, dec, labels=np.arange(n_classes))
        f1 = f1_score(y_val, y_p, average="macro", zero_division=0)
        acc = accuracy_score(y_val, y_p)
        prec = precision_score(y_val, y_p, average="macro", zero_division=0)
        rec = recall_score(y_val, y_p, average="macro", zero_division=0)
        fold_metrics.append({
            "fold": i + 1, "hinge_loss": round(h, 4),
            "f1_macro": round(f1, 4), "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4), "recall_macro": round(rec, 4),
        })
        print(f"    Fold {i+1}: loss={h:.4f}  prec={prec:.4f}  rec={rec:.4f}  "
              f"F1={f1:.4f}  acc={acc:.4f}")

    # --- Full-data predictions (best estimator already refit) ---
    y_pred = best.predict(X)
    dec_all = best.decision_function(X)
    overall_loss = hinge_loss(y, dec_all, labels=np.arange(n_classes))

    metrics = {
        "accuracy":        round(accuracy_score(y, y_pred), 4),
        "precision_macro": round(precision_score(y, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "hinge_loss":      round(overall_loss, 4),
        "best_params":     grid.best_params_,
        "cv_best_f1":      round(grid.best_score_, 4),
        "fold_metrics":    fold_metrics,
    }

    print(f"\n  {'='*50}")
    print(f"  Overall Metrics (full training set)")
    print(f"  {'='*50}")
    for k in ["accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted", "hinge_loss"]:
        print(f"    {k:20s}: {metrics[k]:.4f}")

    report = classification_report(
        y, y_pred, target_names=le.classes_, zero_division=0
    )
    print(f"\n{report}")

    return best, scaler, y_pred, metrics, report


# ────────────────────────── 7. Drift test ──────────────────────────
def run_drift_test(model, scaler, vectorizers, cat_columns_map, le):
    """Test trained SVM on 2026-04-06 data using test_SVM labels."""
    print("\n" + "=" * 60)
    print(" Drift Test  (2026-04-06 data)")
    print("=" * 60)

    test_frames = []
    for dt in DEVICE_TYPES:
        test_path   = os.path.join(TEST_SVM_DIR,     f"{dt}.csv")
        filter_path = os.path.join(DRIFT_FILTER_DIR, f"ipraw_{dt}.csv")
        if not os.path.exists(test_path) or not os.path.exists(filter_path):
            continue

        test_labels = pd.read_csv(test_path, dtype=str)
        test_labels["device_type"] = dt
        test_labels["label"] = dt + "-" + test_labels["vendor"].astype(str)
        ip_set = set(test_labels["ip"])

        parts, found = [], 0
        for chunk in pd.read_csv(
            filter_path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False
        ):
            matched = chunk[chunk["ip"].isin(ip_set)]
            if len(matched):
                parts.append(matched)
                found += len(matched)
            if found >= len(ip_set):
                break

        if parts:
            fp = pd.concat(parts).drop_duplicates("ip", keep="first")
            fp["device_type"] = dt
            merged = test_labels.merge(fp, on=["ip", "device_type"], how="inner")
            if len(merged):
                test_frames.append(merged)
                print(f"  {dt:30s}: {len(merged):4d} samples")
        else:
            print(f"  {dt:30s}: NONE found in filter")

    if not test_frames:
        print("  No test data found — skipping drift test.")
        return

    test_df = pd.concat(test_frames, ignore_index=True)

    known_mask = test_df["label"].isin(le.classes_)
    skipped = (~known_mask).sum()
    if skipped:
        print(f"  Skipping {skipped} samples with unseen labels (not in training set)")
    test_df = test_df[known_mask].copy()
    if test_df.empty:
        print("  All test labels are unseen — skipping drift test.")
        return

    X_test = transform_features(test_df.copy(), vectorizers, cat_columns_map)
    X_test = scaler.transform(X_test)
    y_true = le.transform(test_df["label"])
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }

    print(f"\n  {'='*50}")
    print(f"  Drift Test Metrics")
    print(f"  {'='*50}")
    for k, v in metrics.items():
        print(f"    {k:20s}: {v:.4f}")

    unique_true = np.unique(y_true)
    report = classification_report(
        y_true, y_pred,
        labels=unique_true,
        target_names=le.inverse_transform(unique_true),
        zero_division=0,
    )
    print(f"\n{report}")

    labels_pred = le.inverse_transform(y_pred)
    test_df = test_df.copy()
    test_df["label_pred"] = labels_pred
    test_df["correct"] = test_df["label"] == test_df["label_pred"]

    print("  Per-type drift test accuracy:")
    for dt in DEVICE_TYPES:
        sub = test_df[test_df["device_type"] == dt]
        if sub.empty:
            continue
        out = sub[["ip", "vendor", "label", "label_pred", "correct"]].copy()
        out.to_csv(os.path.join(TEST_PREDICT_SUBDIR, f"{dt}_test_pred.csv"), index=False)
        acc = sub["correct"].mean()
        print(f"    {dt:30s}: {sub['correct'].sum():4d}/{len(sub):4d}  ({acc*100:.1f}%)")

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(os.path.join(SVM_TEST_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_convert)
    with open(os.path.join(SVM_TEST_DIR, "test_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\n  Test predictions saved to {TEST_PREDICT_SUBDIR}")
    print(f"  Test metrics/reports saved to {SVM_TEST_DIR}")
    return metrics


# ────────────────────────── 6. Save results ──────────────────────────
def save_results(merged_df, y_pred, le, metrics, report):
    labels_pred = le.inverse_transform(y_pred)
    merged_df = merged_df.copy()
    merged_df["label_pred"] = labels_pred
    merged_df["correct"] = merged_df["label"] == merged_df["label_pred"]

    print("  Per-type training accuracy:")
    for dt in DEVICE_TYPES:
        sub = merged_df[merged_df["device_type"] == dt]
        out = sub[["ip", "vendor", "label", "label_pred", "correct"]].copy()
        out.to_csv(os.path.join(PREDICT_SUBDIR, f"{dt}_pred.csv"), index=False)
        acc = sub["correct"].mean()
        print(f"    {dt:30s}: {sub['correct'].sum():4d}/{len(sub):4d}  "
              f"({acc*100:.1f}%)")

    # numpy / pandas type converter for JSON
    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_convert)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    label_map = {name: int(idx) for idx, name in enumerate(le.classes_)}
    with open(os.path.join(OUTPUT_DIR, "label_encoding.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\n  Predictions saved to {PREDICT_SUBDIR}")
    print(f"  Metrics/reports saved to {OUTPUT_DIR}")


# ────────────────────────── Model save / load ──────────────────────────
def save_model(model, scaler, vectorizers, cat_columns_map, le):
    bundle = {
        "model": model,
        "scaler": scaler,
        "vectorizers": vectorizers,
        "cat_columns_map": cat_columns_map,
        "le": le,
    }
    joblib.dump(bundle, MODEL_BUNDLE, compress=3)
    print(f"  Model bundle saved → {MODEL_BUNDLE}")


def load_model():
    if not os.path.exists(MODEL_BUNDLE):
        return None
    bundle = joblib.load(MODEL_BUNDLE)
    print(f"  Model bundle loaded ← {MODEL_BUNDLE}")
    return bundle["model"], bundle["scaler"], bundle["vectorizers"], bundle["cat_columns_map"], bundle["le"]


# ────────────────────────── 8. Validation ──────────────────────────
def load_detected_ips(csv_path):
    """Load detected IPs CSV (ip, vendor, device_type, class_drift_score)."""
    df = pd.read_csv(csv_path, dtype=str)
    df["label"] = df["device_type"] + "-" + df["vendor"]
    return df


def load_fingerprints_from_filter(label_df):
    """Load fingerprints for IPs from the drift filter directory (2026-04-06)."""
    all_fps = []
    for dt in DEVICE_TYPES:
        ip_set = set(label_df.loc[label_df["device_type"] == dt, "ip"])
        if not ip_set:
            continue
        filter_path = os.path.join(DRIFT_FILTER_DIR, f"ipraw_{dt}.csv")
        if not os.path.exists(filter_path):
            continue
        print(f"    {dt:30s} ({len(ip_set):4d} IPs) ... ", end="", flush=True)
        t0 = time.time()
        parts, found = [], 0
        for chunk in pd.read_csv(
            filter_path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False
        ):
            matched = chunk[chunk["ip"].isin(ip_set)]
            if len(matched):
                parts.append(matched)
                found += len(matched)
            if found >= len(ip_set):
                break
        if parts:
            fp = pd.concat(parts).drop_duplicates("ip", keep="first")
            fp["device_type"] = dt
            all_fps.append(fp)
            print(f"found {len(fp):4d}  ({time.time()-t0:.1f}s)")
        else:
            print("NONE found!")
    if not all_fps:
        return pd.DataFrame()
    return pd.concat(all_fps, ignore_index=True)


def run_validation(method_name, detected_csv, train_label_df, train_fp_df):
    """
    Validation: add detected IPs to training set, retrain SVM, evaluate on drift test set.
    """
    val_out_dir = os.path.join(VALIDATION_DIR, "performance", method_name)
    val_pred_dir = os.path.join(val_out_dir, "predict")
    os.makedirs(val_pred_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f" Validation  ({method_name})")
    print("=" * 60)

    # ── Load detected IPs ──
    print(f"\n  Loading detected IPs from {detected_csv} ...")
    detected_df = load_detected_ips(detected_csv)
    print(f"  Detected IPs: {len(detected_df)}")

    # ── Load fingerprints for detected IPs from filter dir ──
    print("  Loading fingerprints for detected IPs ...")
    det_fp = load_fingerprints_from_filter(detected_df)
    if det_fp.empty:
        print("  No fingerprints found for detected IPs — skipping validation.")
        return
    det_merged = detected_df[["ip", "vendor", "device_type", "label"]].merge(
        det_fp, on=["ip", "device_type"], how="inner"
    )
    print(f"  Detected IPs with fingerprints: {len(det_merged)}")

    # ── Combine with original training data ──
    orig_merged = train_label_df[["ip", "vendor", "device_type", "label"]].merge(
        train_fp_df, on=["ip", "device_type"], how="inner"
    )
    combined = pd.concat([orig_merged, det_merged], ignore_index=True)
    combined = combined.drop_duplicates("ip", keep="first")
    print(f"  Combined training set: {len(combined)} (orig {len(orig_merged)} + detected {len(det_merged)})")

    # ── Encode labels ──
    le_val = LabelEncoder()
    y_combined = le_val.fit_transform(combined["label"])

    # ── Build features ──
    print("  Building features ...")
    X_combined, vec_val, cat_map_val = build_features(combined.copy())

    # ── Train SVM ──
    scaler_val = MaxAbsScaler()
    X_combined = scaler_val.fit_transform(X_combined)

    n_classes = len(le_val.classes_)
    print(f"  Samples={X_combined.shape[0]}  Features={X_combined.shape[1]}  Classes={n_classes}")

    param_grid = [
        {"kernel": ["rbf"],    "C": [1, 10, 100], "gamma": ["scale", 0.01]},
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["poly"],   "C": [1, 10],      "degree": [3], "gamma": ["scale"]},
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(
        decision_function_shape="ovr",
        class_weight="balanced",
        random_state=42,
        cache_size=2000,
    )
    grid = GridSearchCV(
        svm, param_grid, cv=skf, scoring="f1_macro",
        n_jobs=-1, verbose=2, refit=True,
    )
    print("\n  Running GridSearchCV ...")
    t0 = time.time()
    grid.fit(X_combined, y_combined)
    print(f"  GridSearch done in {time.time()-t0:.0f}s")
    best = grid.best_estimator_
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")

    # ── Evaluate on drift test set ──
    print("\n  Evaluating on drift test set ...")
    test_frames = []
    for dt in DEVICE_TYPES:
        test_path   = os.path.join(TEST_SVM_DIR,     f"{dt}.csv")
        filter_path = os.path.join(DRIFT_FILTER_DIR, f"ipraw_{dt}.csv")
        if not os.path.exists(test_path) or not os.path.exists(filter_path):
            continue
        test_labels = pd.read_csv(test_path, dtype=str)
        test_labels["device_type"] = dt
        test_labels["label"] = dt + "-" + test_labels["vendor"].astype(str)
        ip_set = set(test_labels["ip"])
        parts, found = [], 0
        for chunk in pd.read_csv(
            filter_path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False
        ):
            matched = chunk[chunk["ip"].isin(ip_set)]
            if len(matched):
                parts.append(matched)
                found += len(matched)
            if found >= len(ip_set):
                break
        if parts:
            fp = pd.concat(parts).drop_duplicates("ip", keep="first")
            fp["device_type"] = dt
            merged = test_labels.merge(fp, on=["ip", "device_type"], how="inner")
            if len(merged):
                test_frames.append(merged)
                print(f"    {dt:30s}: {len(merged):4d} samples")
        else:
            print(f"    {dt:30s}: NONE found in filter")

    if not test_frames:
        print("  No test data found — skipping validation test.")
        return

    test_df = pd.concat(test_frames, ignore_index=True)
    known_mask = test_df["label"].isin(le_val.classes_)
    skipped = (~known_mask).sum()
    if skipped:
        print(f"  Skipping {skipped} samples with unseen labels")
    test_df = test_df[known_mask].copy()
    if test_df.empty:
        print("  All test labels are unseen — skipping validation test.")
        return

    X_test = transform_features(test_df.copy(), vec_val, cat_map_val)
    X_test = scaler_val.transform(X_test)
    y_pred_enc = best.predict(X_test)

    labels_pred = le_val.inverse_transform(y_pred_enc)
    test_df = test_df.copy()
    test_df["label_pred"] = labels_pred
    test_df["correct"] = [
        label_match(a, b) for a, b in zip(test_df["label"], test_df["label_pred"])
    ]

    # Compute metrics on normalized labels (case-insensitive + vendor fuzzy)
    norm_true = [normalize_label(l) for l in test_df["label"]]
    norm_pred = [normalize_label(l) for l in test_df["label_pred"]]
    le_norm = LabelEncoder()
    all_norm = list(set(norm_true + norm_pred))
    le_norm.fit(all_norm)
    y_true_n = le_norm.transform(norm_true)
    y_pred_n = le_norm.transform(norm_pred)

    metrics = {
        "accuracy":        round(accuracy_score(y_true_n, y_pred_n), 4),
        "precision_macro": round(precision_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true_n, y_pred_n, average="weighted", zero_division=0), 4),
        "best_params":     grid.best_params_,
        "cv_best_f1":      round(grid.best_score_, 4),
        "n_train_orig":    len(orig_merged),
        "n_train_detected": len(det_merged),
        "n_train_combined": len(combined),
    }

    print(f"\n  {'='*50}")
    print(f"  Validation Test Metrics ({method_name})")
    print(f"  {'='*50}")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        print(f"    {k:20s}: {metrics[k]:.4f}")

    unique_true_n = np.unique(y_true_n)
    report = classification_report(
        y_true_n, y_pred_n,
        labels=unique_true_n,
        target_names=le_norm.inverse_transform(unique_true_n),
        zero_division=0,
    )
    print(f"\n{report}")

    print(f"  Per-type validation accuracy ({method_name}):")
    for dt in DEVICE_TYPES:
        sub = test_df[test_df["device_type"] == dt]
        if sub.empty:
            continue
        out = sub[["ip", "vendor", "label", "label_pred", "correct"]].copy()
        out.to_csv(os.path.join(val_pred_dir, f"{dt}_val_pred.csv"), index=False)
        acc = sub["correct"].mean()
        print(f"    {dt:30s}: {sub['correct'].sum():4d}/{len(sub):4d}  ({acc*100:.1f}%)")

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(os.path.join(val_out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_convert)
    with open(os.path.join(val_out_dir, "val_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\n  Validation results saved to {val_out_dir}")
    return metrics


# ────────────────────── Re-evaluate predictions ──────────────────────
def reeval_predictions(method_name):
    """Re-read existing prediction CSVs, recompute correctness and full metrics
    using normalized labels (case-insensitive + vendor fuzzy match)."""
    perf_dir = os.path.join(VALIDATION_DIR, "performance", method_name)
    pred_dir = os.path.join(perf_dir, "predict")

    print("\n" + "=" * 60)
    print(f" Re-evaluate predictions  ({method_name})")
    print("=" * 60)

    all_frames = []
    for dt in DEVICE_TYPES:
        csv_path = os.path.join(pred_dir, f"{dt}_val_pred.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, dtype=str)
        df["correct"] = [
            label_match(a, b) for a, b in zip(df["label"], df["label_pred"])
        ]
        df["device_type"] = dt
        df.to_csv(csv_path, index=False, columns=["ip", "vendor", "label", "label_pred", "correct"])
        all_frames.append(df)
        acc = df["correct"].mean()
        print(f"  {dt:30s}: {df['correct'].sum():4d}/{len(df):4d}  ({acc*100:.1f}%)")

    if not all_frames:
        print("  No prediction files found.")
        return

    full = pd.concat(all_frames, ignore_index=True)

    # Compute full metrics on normalized labels
    norm_true = [normalize_label(l) for l in full["label"]]
    norm_pred = [normalize_label(l) for l in full["label_pred"]]
    le_norm = LabelEncoder()
    le_norm.fit(list(set(norm_true + norm_pred)))
    y_true_n = le_norm.transform(norm_true)
    y_pred_n = le_norm.transform(norm_pred)

    metrics = {
        "accuracy":        round(accuracy_score(y_true_n, y_pred_n), 4),
        "precision_macro": round(precision_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true_n, y_pred_n, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true_n, y_pred_n, average="weighted", zero_division=0), 4),
        "total_samples":   int(len(full)),
    }

    print(f"\n  {'='*50}")
    print(f"  Overall Metrics ({method_name})")
    print(f"  {'='*50}")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        print(f"    {k:20s}: {metrics[k]:.4f}")

    metrics_path = os.path.join(perf_dir, "val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Updated metrics saved to {metrics_path}")


# ────────────────────────── main ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SVM IoT Device Classifier")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if a saved model bundle exists")
    parser.add_argument("--drift-only", action="store_true",
                        help="Skip training, only run drift test using saved model")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training and drift test, only run validation")
    parser.add_argument("--reeval", action="store_true",
                        help="Re-evaluate existing prediction CSVs case-insensitively")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print(" SVM IoT Device Classifier  (type-vendor labels)")
    print("=" * 60)

    if args.reeval:
        reeval_predictions("IoTProber")
        reeval_predictions("CADE")
        print(f"\nTotal time: {time.time()-t_start:.0f}s")
        return

    if args.validate_only:
        # ── Validate-only mode: skip training & drift test ──
        print("\n  --validate-only: skipping training and drift test.")
        print("\n  Loading training data for validation ...")
        label_df = load_labels()
        fp_df = load_fingerprints(label_df)
    else:
        # ── Try loading saved model ────────────────────────────────────────
        bundle = None if args.retrain else load_model()

        if bundle is not None:
            model, scaler, vectorizers, cat_columns_map, le = bundle
            print(f"  Loaded {len(le.classes_)} classes from saved bundle.")
            print("  Skipping training. Use --retrain to force retraining.")
        else:
            if args.drift_only:
                print("  --drift-only requires a saved model bundle. Run without --drift-only first.")
                return

            # 1. Labels
            print("\n[1/5] Loading labels ...")
            label_df = load_labels()
            print(f"  {len(label_df)} samples, {label_df['label'].nunique()} classes")

            # 2. Fingerprints
            print("\n[2/5] Loading fingerprints ...")
            fp_df = load_fingerprints(label_df)

            merged = label_df.merge(fp_df, on=["ip", "device_type"], how="inner")
            print(f"  Matched: {len(merged)} / {len(label_df)}")

            le = LabelEncoder()
            y = le.fit_transform(merged["label"])

            # 3. Features
            print("\n[3/5] Building features ...")
            X, vectorizers, cat_columns_map = build_features(merged)

            # 4-5. Train & Evaluate
            print("\n[4/5] Training SVM ...")
            model, scaler, y_pred, metrics, report = train_evaluate(X, y, le)

            # 6. Save predictions
            print("\n[5/5] Saving predictions ...")
            save_results(merged, y_pred, le, metrics, report)

            # Save model bundle
            print("\n  Saving model bundle ...")
            save_model(model, scaler, vectorizers, cat_columns_map, le)

        # 7. Drift test
        print("\n[6/8] Running drift test on 2026-04-06 data ...")
        run_drift_test(model, scaler, vectorizers, cat_columns_map, le)

        # ── Prepare original training data for validation ──
        if bundle is not None:
            # Need to reload training data for validation
            print("\n  Reloading training data for validation ...")
            label_df = load_labels()
            fp_df = load_fingerprints(label_df)

    # 8. Validation: IoTProber (using CADE-detected IPs)
    print("\n[7/8] Running validation (IoTProber) ...")
    run_validation("IoTProber", CADE_DETECTED, label_df, fp_df)

    # 9. Validation: CADE (using IoTProber-detected IPs)
    print("\n[8/8] Running validation (CADE) ...")
    run_validation("CADE", IOTPROBER_DETECTED, label_df, fp_df)

    print(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
