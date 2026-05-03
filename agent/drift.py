"""
Perspective-Aware Contrastive Autoencoder (PACA) for In-Class Concept Drift Detection
======================================================================================
Detects in-class concept drift where the (vendor + device_type) label is unchanged
but the feature space has shifted.  Non-critical perspectives (whois, as, dns, sd,
htags, hfavicons, body) are weighted **higher** in the loss because their fluctuation
signals genuine in-class drift, while critical perspectives (sw, hw, os, certificate)
are weighted **lower** because their changes more likely indicate a new vendor / device.

Loss = L_recon + λ · L_contrast

L_recon   = Σ_{p∈P} α_p · (1/|C_p|) · Σ_{k∈C_p} (x_k − x̂_k)²

L_contrast = (1/N) Σ_{(i,j)} [ y_ij · D_w(i,j)²
             + (1−y_ij) · max(0, m − D_w(i,j))² ]

D_w(i,j)  = sqrt( Σ_{p∈P} α_p · ‖r_i^p − r_j^p‖² + ε )

where  r_i^p = x_i^p − x̂_i^p  is the per-perspective reconstruction residual.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import median_abs_deviation
from collections import OrderedDict
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────── Paths ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
IPRAW_DIR = os.path.join(ROOT_DIR, "platform_data", "csv")
FEATURES_FILE = os.path.join(ROOT_DIR, "local_used_features.txt")
PERSPECTIVE_FILE = os.path.join(ROOT_DIR, "perspective_info.json")
DRIFT_OUTPUT_DIR = os.path.join(ROOT_DIR, "drift_data", "autoencoder_drift")

DEVICE_TYPES = [
    "BUILDING_AUTOMATION", "CAMERA", "MEDIA_SERVER", "MEDICAL",
    "NAS", "NVR", "POWER_METER", "PRINTER", "ROUTER", "SCADA", "VPN",
]

EXCLUDED_PERSPECTIVES = {"http", "hpart", "overall"}

# ── Drift-sensitivity weights α_p ─────────────────────────
# High α → fluctuation counts strongly toward in-class drift
# Low  α → fluctuation is suppressed (likely new device/vendor)
DRIFT_SENSITIVITY = {
    "whois":       1.0,
    "as":          1.0,
    "dns":         0.9,
    "sd":          0.9,
    "htags":       0.8,
    "hfavicons":   0.8,
    "body":        0.7,
    "os":          0.3,
    "sw":          0.2,
    "hw":          0.2,
    "certificate": 0.3,
}

# ── Column lists (25 features from local_used_features.txt) ──
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

CAT_FEATURES = [
    "as-country_code",
    "whois-network-handle", "whois-organization-handle",
    "os-vendor", "os-product",
]

_TFIDF_MAX = {"http-bodys": 200, "http-tags": 200, "http-favicon-urls": 100}
_TFIDF_DEFAULT = 300


# ═══════════════════════════════════════════════════════════
# 1. Perspective Configuration
# ═══════════════════════════════════════════════════════════

def load_perspectives():
    with open(PERSPECTIVE_FILE, "r", encoding="utf-8") as f:
        pinfo = json.load(f)
    perspectives = OrderedDict()
    for pname, pdata in pinfo.items():
        if pname in EXCLUDED_PERSPECTIVES:
            continue
        perspectives[pname] = {
            "cols": pdata["cols"],
            "weight": DRIFT_SENSITIVITY.get(pname, 0.5),
        }
    return perspectives


# ═══════════════════════════════════════════════════════════
# 2. Data Loading & Feature Engineering
# ═══════════════════════════════════════════════════════════

def _comma_tokenizer(text):
    return [t.strip() for t in str(text).split(",") if t.strip()]


def load_ipraw(device_type, ip_filter=None, ipraw_dir=IPRAW_DIR):
    path = os.path.join(ipraw_dir, f"ipraw_{device_type}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    parts = []
    for chunk in pd.read_csv(path, usecols=USE_COLS, chunksize=50_000, low_memory=False):
        if ip_filter is not None:
            chunk = chunk[chunk["ip"].isin(ip_filter)]
        if len(chunk):
            parts.append(chunk)
    return pd.concat(parts).drop_duplicates("ip", keep="first") if parts else pd.DataFrame()


def build_feature_matrix(df, vectorizers=None, cat_columns_map=None, fit=True):
    """Return (X_dense, perspective_indices, vectorizers, cat_columns_map)."""
    if vectorizers is None:
        vectorizers = {}
    if cat_columns_map is None:
        cat_columns_map = {}

    matrices, col_ranges, cur = [], {}, 0

    # TF-IDF
    for col in TEXT_FEATURES:
        df[col] = df[col].fillna("").astype(str)
        mx = _TFIDF_MAX.get(col, _TFIDF_DEFAULT)
        if fit:
            vec = TfidfVectorizer(tokenizer=_comma_tokenizer, token_pattern=None,
                                  max_features=mx, sublinear_tf=True)
            m = vec.fit_transform(df[col]).toarray()
            vectorizers[col] = vec
        else:
            m = vectorizers[col].transform(df[col]).toarray()
        col_ranges[col] = (cur, cur + m.shape[1]); cur += m.shape[1]
        matrices.append(m)

    # One-hot categorical
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("UNK").astype(str)
        dum = pd.get_dummies(df[col], prefix=col)
        if fit:
            cat_columns_map[col] = list(dum.columns)
        else:
            dum = dum.reindex(columns=cat_columns_map[col], fill_value=0)
        m = dum.values.astype(np.float32)
        col_ranges[col] = (cur, cur + m.shape[1]); cur += m.shape[1]
        matrices.append(m)

    # Numeric: as-asn
    asn = pd.to_numeric(df["as-asn"], errors="coerce").fillna(0).values.reshape(-1, 1).astype(np.float32)
    col_ranges["as-asn"] = (cur, cur + 1); cur += 1
    matrices.append(asn)

    X = np.hstack(matrices)

    # Map perspective → column indices
    perspectives = load_perspectives()
    p_idx = {}
    for pname, pdata in perspectives.items():
        idx = []
        for c in pdata["cols"]:
            if c in col_ranges:
                s, e = col_ranges[c]
                idx.extend(range(s, e))
        if idx:
            p_idx[pname] = idx

    return X, p_idx, vectorizers, cat_columns_map


# ═══════════════════════════════════════════════════════════
# 3. Model
# ═══════════════════════════════════════════════════════════

class PerspectiveAwareCAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), latent_dim=16):
        super().__init__()
        # Encoder
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*layers)
        # Decoder (symmetric)
        layers = []
        d = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
            d = h
        layers.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ═══════════════════════════════════════════════════════════
# 4. Loss
# ═══════════════════════════════════════════════════════════

class PerspectiveWeightedLoss(nn.Module):
    def __init__(self, p_idx, sensitivity, lam=1.0, margin=2.0):
        super().__init__()
        self.p_idx = p_idx
        self.sens = sensitivity
        self.lam = lam
        self.margin = margin
        # pre-build index tensors (moved to device on first call)
        self._idx_tensors = {p: torch.tensor(idx, dtype=torch.long) for p, idx in p_idx.items()}
        self._device_set = False

    def _ensure_device(self, device):
        if not self._device_set:
            self._idx_tensors = {p: t.to(device) for p, t in self._idx_tensors.items()}
            self._device_set = True

    def recon_loss(self, x, x_hat):
        self._ensure_device(x.device)
        loss = torch.tensor(0.0, device=x.device)
        for p, idx_t in self._idx_tensors.items():
            alpha = self.sens.get(p, 0.5)
            loss = loss + alpha * F.mse_loss(x_hat[:, idx_t], x[:, idx_t])
        return loss

    def contrastive_loss(self, x, x_hat, labels):
        bs = x.size(0)
        half = bs // 2
        if half < 1:
            return torch.tensor(0.0, device=x.device)

        self._ensure_device(x.device)
        is_same = (labels[:half] == labels[half: 2 * half]).float()
        resid = x - x_hat

        w_dist_sq = torch.zeros(half, device=x.device)
        for p, idx_t in self._idx_tensors.items():
            alpha = self.sens.get(p, 0.5)
            rp = resid[:, idx_t]
            diff = rp[:half] - rp[half: 2 * half]
            w_dist_sq = w_dist_sq + alpha * (diff ** 2).sum(dim=1)

        dist = torch.sqrt(w_dist_sq + 1e-10)
        l_same = is_same * dist.pow(2)
        l_diff = (1 - is_same) * F.relu(self.margin - dist).pow(2)
        return (l_same + l_diff).mean()

    def forward(self, x, x_hat, z, labels):
        lr = self.recon_loss(x, x_hat)
        lc = self.contrastive_loss(x, x_hat, labels)
        return lr + self.lam * lc, lr, lc


# ═══════════════════════════════════════════════════════════
# 5. Dataset (pair-based)
# ═══════════════════════════════════════════════════════════

class PairDataset(Dataset):
    def __init__(self, X, y, similar_ratio=0.25):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = np.asarray(y)
        self.sr = similar_ratio
        self.n = len(X)
        self.lab2idx = {}
        for i, l in enumerate(self.y):
            self.lab2idx.setdefault(int(l), []).append(i)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        y1 = int(self.y[i])
        if np.random.random() < self.sr:
            cands = self.lab2idx[y1]
        else:
            others = [l for l in self.lab2idx if l != y1]
            cands = self.lab2idx[others[np.random.randint(len(others))]] if others else list(range(self.n))
        j = cands[np.random.randint(len(cands))]
        return self.X[i], y1, self.X[j], int(self.y[j])


# ═══════════════════════════════════════════════════════════
# 6. Training
# ═══════════════════════════════════════════════════════════

def train_model(X, y, p_idx, hidden_dims=(128, 64), latent_dim=16,
                lam=1.0, margin=2.0, lr=1e-3, bs=64, epochs=100,
                sim_ratio=0.25, device="cpu", save_path=None):
    model = PerspectiveAwareCAE(X.shape[1], hidden_dims, latent_dim).to(device)
    crit = PerspectiveWeightedLoss(p_idx, DRIFT_SENSITIVITY, lam, margin)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loader = DataLoader(PairDataset(X, y, sim_ratio), batch_size=bs, shuffle=True, drop_last=True)

    best = float("inf")
    for ep in range(epochs):
        model.train()
        tot_l, tot_r, tot_c, nb = 0, 0, 0, 0
        for x1, y1, x2, y2 in loader:
            xb = torch.cat([x1, x2]).to(device)
            yb = torch.cat([torch.tensor(y1), torch.tensor(y2)]).to(device)
            xh, z = model(xb)
            loss, lr_, lc_ = crit(xb, xh, z, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_l += loss.item(); tot_r += lr_.item(); tot_c += lc_.item(); nb += 1

        avg = tot_l / max(nb, 1)
        sched.step(avg)
        if (ep + 1) % 10 == 0:
            logger.info(f"Epoch {ep+1}/{epochs}  loss={avg:.4f}  recon={tot_r/max(nb,1):.4f}  contrast={tot_c/max(nb,1):.4f}")
        if avg < best:
            best = avg
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"state": model.state_dict(), "dim": X.shape[1],
                            "hidden": hidden_dims, "latent": latent_dim, "p_idx": p_idx}, save_path)
    logger.info(f"Training done. Best loss={best:.4f}")
    return model


# ═══════════════════════════════════════════════════════════
# 7. Drift Scoring & Detection
# ═══════════════════════════════════════════════════════════

def compute_drift_scores(model, X, p_idx, device="cpu"):
    """Return (total_scores, per_perspective_scores)."""
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        xh, _ = model(xt)
    err = (xt.cpu().numpy() - xh.cpu().numpy()) ** 2

    p_scores, total = {}, np.zeros(len(X))
    for p, idx in p_idx.items():
        a = DRIFT_SENSITIVITY.get(p, 0.5)
        ps = err[:, idx].mean(axis=1)
        p_scores[p] = ps
        total += a * ps
    return total, p_scores


def detect_drift(ref_scores, test_scores, mad_thr=3.5):
    med = np.median(ref_scores)
    mad = median_abs_deviation(ref_scores)
    mad = max(mad, 1e-10)
    thr = med + mad_thr * mad
    return test_scores > thr, thr, med, mad


def explain_drift(ref_ps, test_ps, p_idx, mad_thr=3.5):
    expl = {}
    for p in p_idx:
        ref, test = ref_ps[p], test_ps[p]
        med = np.median(ref)
        mad = max(median_abs_deviation(ref), 1e-10)
        zs = (test - med) / mad
        a = DRIFT_SENSITIVITY.get(p, 0.5)
        expl[p] = {
            "median_ref": float(med), "mad_ref": float(mad),
            "mean_test": float(test.mean()), "mean_z": float(zs.mean()),
            "drift_ratio": float((zs > mad_thr).mean()),
            "alpha": a, "contribution": float(a * zs.mean()),
        }
    return dict(sorted(expl.items(), key=lambda x: x[1]["contribution"], reverse=True))


# ═══════════════════════════════════════════════════════════
# 8. Full Pipeline
# ═══════════════════════════════════════════════════════════

def run_drift_detection(
    ref_csv_dir, test_csv_dir,
    device_types=None, vendor_col="vendor", type_col="device_type",
    hidden_dims=(128, 64), latent_dim=16,
    lam=1.0, margin=2.0, lr=1e-3, bs=64, epochs=100,
    sim_ratio=0.25, mad_thr=3.5,
    output_dir=DRIFT_OUTPUT_DIR, torch_device="cpu",
):
    if device_types is None:
        device_types = DEVICE_TYPES
    os.makedirs(output_dir, exist_ok=True)

    # ── Load reference ──
    logger.info("Loading reference data ...")
    ref_frames = []
    for dt in device_types:
        df = load_ipraw(dt, ipraw_dir=ref_csv_dir)
        if not df.empty:
            df["device_type"] = dt; ref_frames.append(df)
            logger.info(f"  {dt}: {len(df)}")
    if not ref_frames:
        logger.error("No reference data."); return
    ref_df = pd.concat(ref_frames, ignore_index=True)
    if vendor_col in ref_df.columns:
        ref_df["class_label"] = ref_df[type_col] + "-" + ref_df[vendor_col].fillna("UNK").astype(str)
    else:
        ref_df["class_label"] = ref_df[type_col]

    # ── Feature matrix ──
    logger.info("Building features ...")
    X_ref, p_idx, vecs, cmap = build_feature_matrix(ref_df.copy(), fit=True)
    le = LabelEncoder(); y_ref = le.fit_transform(ref_df["class_label"])
    scaler = StandardScaler(); X_ref_s = scaler.fit_transform(X_ref)
    logger.info(f"Ref: {X_ref_s.shape}  classes={len(le.classes_)}")

    # ── Train ──
    mpath = os.path.join(output_dir, "paca_model.pt")
    model = train_model(X_ref_s, y_ref, p_idx, hidden_dims, latent_dim,
                        lam, margin, lr, bs, epochs, sim_ratio, torch_device, mpath)

    # ── Reference scores ──
    ref_scores, ref_ps = compute_drift_scores(model, X_ref_s, p_idx, torch_device)

    # ── Load test ──
    logger.info("Loading test data ...")
    test_frames = []
    for dt in device_types:
        df = load_ipraw(dt, ipraw_dir=test_csv_dir)
        if not df.empty:
            df["device_type"] = dt; test_frames.append(df)
            logger.info(f"  {dt}: {len(df)}")
    if not test_frames:
        logger.error("No test data."); return
    test_df = pd.concat(test_frames, ignore_index=True)
    if vendor_col in test_df.columns:
        test_df["class_label"] = test_df[type_col] + "-" + test_df[vendor_col].fillna("UNK").astype(str)
    else:
        test_df["class_label"] = test_df[type_col]

    X_test, _, _, _ = build_feature_matrix(test_df.copy(), vecs, cmap, fit=False)
    X_test_s = scaler.transform(X_test)

    # ── Detect ──
    test_scores, test_ps = compute_drift_scores(model, X_test_s, p_idx, torch_device)
    is_drift, thr, med, mad = detect_drift(ref_scores, test_scores, mad_thr)
    nd = int(is_drift.sum())
    logger.info(f"Threshold={thr:.4f}  drifted={nd}/{len(test_scores)} ({100*nd/len(test_scores):.1f}%)")

    # ── Explain ──
    expl = explain_drift(ref_ps, test_ps, p_idx, mad_thr)

    # ── Save ──
    res = test_df[["ip", "device_type"]].copy()
    res["drift_score"] = test_scores; res["is_drift"] = is_drift.astype(int)
    for p in p_idx:
        res[f"score_{p}"] = test_ps[p]
    res.to_csv(os.path.join(output_dir, "drift_results.csv"), index=False)

    with open(os.path.join(output_dir, "drift_explanations.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": float(thr), "median": float(med), "mad": float(mad),
                    "n_drift": nd, "n_total": len(test_scores), "perspectives": expl},
                   f, indent=2, ensure_ascii=False)

    csm = res.groupby("device_type").agg(total=("is_drift", "count"), drifted=("is_drift", "sum"),
                                          mean_score=("drift_score", "mean")).reset_index()
    csm["drift_ratio"] = csm["drifted"] / csm["total"]
    csm.to_csv(os.path.join(output_dir, "class_drift_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("Perspective Drift Explanations (by weighted contribution):")
    print("=" * 60)
    for p, e in expl.items():
        print(f"  {p:15s}  α={e['alpha']:.1f}  z={e['mean_z']:.2f}  "
              f"drift={e['drift_ratio']:.1%}  contrib={e['contribution']:.3f}")

    logger.info("Done.")
    return model, res, expl


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PACA In-Class Drift Detection")
    ap.add_argument("--ref_dir",   default=IPRAW_DIR)
    ap.add_argument("--test_dir",  default=IPRAW_DIR)
    ap.add_argument("--output_dir", default=DRIFT_OUTPUT_DIR)
    ap.add_argument("--hidden",    type=int, nargs="+", default=[128, 64])
    ap.add_argument("--latent",    type=int, default=16)
    ap.add_argument("--lam",       type=float, default=1.0)
    ap.add_argument("--margin",    type=float, default=2.0)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--bs",        type=int, default=64)
    ap.add_argument("--epochs",    type=int, default=100)
    ap.add_argument("--sim_ratio", type=float, default=0.25)
    ap.add_argument("--mad_thr",   type=float, default=3.5)
    ap.add_argument("--device",    default="cpu")
    a = ap.parse_args()
    run_drift_detection(a.ref_dir, a.test_dir, hidden_dims=a.hidden, latent_dim=a.latent,
                        lam=a.lam, margin=a.margin, lr=a.lr, bs=a.bs, epochs=a.epochs,
                        sim_ratio=a.sim_ratio, mad_thr=a.mad_thr,
                        output_dir=a.output_dir, torch_device=a.device)
