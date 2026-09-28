#!/usr/bin/env python
"""
Select the best-identifiable demo fingerprints for each of the 11 RAG device
types (config/rag_devices.json) and save them under demo/{TYPE}/.

"Best" := among candidate fingerprints of the type, the ones the v2 adapter
classifies CORRECTLY with the highest type confidence, using the exact
production inference chain (build_unseen_type_vendor_classification_prompt →
_generate_classification → _classification_novelty_result). Each selected
case also gets a PACA drift score (detect_query_device) and, for CAMERA only,
local-retrieval neighbours from the built npz store.

Two phases (8-GPU sharded, mirrors rerun_calibration_retrieval):
  --rank R --world_size 8   generate predictions for this rank's candidates
  --merge                   rank candidates per type, write demo/{TYPE}/cases
"""

import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../iotprober
LLAMA3 = os.path.join(REPO, "evaluation", "unseen", "llama3")
ADAPTER = os.path.join(LLAMA3, "results_v2", "final_model")
ORIG_FT = os.path.join(LLAMA3, "log", "original_run", "fine-tune.py")
IPRAW_DIR = "/dev/shm/ipraw"
WORK = "/dev/shm/demo_select"
DEMO_DIR = os.path.join(REPO, "demo")
CANDIDATES_PER_TYPE = 40
TOP_K = 3

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "agent"))

from path_config import LOCAL_VECTOR_DB_DIR  # noqa: E402

spec = importlib.util.spec_from_file_location("finetune_original", ORIG_FT)
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from util import (  # noqa: E402
    build_fingerprint_info_text,
    build_unseen_type_vendor_classification_prompt,
)


# Generic cloud/VM hosts are labelled as consumer device types in parts of the
# corpus (e.g. POWER_METER is ~85% AWS EC2 boxes) — they demonstrate hosted
# software, not the device family, so they are excluded unless the pool would
# otherwise be empty (recorded in selection_summary.json).
CLOUD_HOST_PAT = (r"compute\.amazonaws|compute\.internal|googleusercontent|cloudapp\.azure"
                  r"|digitalocean|linode|vultr|hetzner|ovh\.net|contabo|oraclecloud|scaleway"
                  r"|amazon-0[0-9]|amazon-aes|amazon data services|amazon technologies"
                  r"|google cloud|microsoft azure|alibaba cloud|tencent cloud")


def _require_files(paths, what, hint):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"missing {what} ({len(missing)}):\n  " + "\n  ".join(missing[:6]) +
            (f"\n  ... and {len(missing) - 6} more" if len(missing) > 6 else "") + f"\n{hint}")


def _atomic_json_dump(path, value, indent=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _heldout_ips():
    """IPs used by the adapter training/validation — a demo case must not be one
    of them (otherwise the showcase demonstrates memorisation, not inference)."""
    ips = set()
    for name in ("unseen_sft_v2.jsonl", "validation_sft.jsonl", "unseen_sft.jsonl"):
        p = os.path.join(LLAMA3, "dataset", name)
        if not os.path.exists(p):
            continue
        with open(p, errors="ignore") as fh:
            for line in fh:
                try:
                    m = json.loads(line).get("metadata") or {}
                except Exception:
                    continue
                if m.get("ip"):
                    ips.add(str(m["ip"]))
    return ips


def _ambiguous_ips(types):
    """IPs that appear in more than one type file — the corpus label is then
    contradictory, so they cannot serve as a 'correct' demo case."""
    seen, dup = {}, set()
    for t in types:
        p = os.path.join(IPRAW_DIR, f"ipraw_{t}.csv")
        for chunk in pd.read_csv(p, usecols=["ip"], chunksize=500_000, low_memory=False):
            for ip in chunk["ip"].astype(str):
                if ip in seen and seen[ip] != t:
                    dup.add(ip)
                else:
                    seen.setdefault(ip, t)
    return dup


def build_candidates():
    """Rich, held-out fingerprints per type from the raw ipraw CSVs (deterministic)."""
    types = json.load(open(os.path.join(REPO, "config", "rag_devices.json")))["IoT"]
    os.makedirs(WORK, exist_ok=True)

    ipraw_paths = [os.path.join(IPRAW_DIR, f"ipraw_{t}.csv") for t in types]
    _require_files(ipraw_paths, "ipraw fingerprint CSVs",
                   "  hint: they live on tmpfs (cleared on reboot). Rebuild with:\n"
                   "    hf_hub_download('IoTProber/raw_dataset', 'platform_data/rag/ipraw_files.tar.gz',"
                   " repo_type='dataset')  ->  tar -xzf into /dev/shm/ipraw")

    heldout = _heldout_ips()
    ambiguous = _ambiguous_ips(types)
    print(f"excluded: {len(heldout)} training/validation IPs, {len(ambiguous)} multi-type IPs")

    rng = np.random.default_rng(7)
    cand, meta = {}, {}
    for t in types:
        df = pd.read_csv(os.path.join(IPRAW_DIR, f"ipraw_{t}.csv"), low_memory=False)
        n_raw = len(df)
        # drop IPs the adapter was trained on, ambiguous labels, and generic cloud hosts
        df = df[~df["ip"].astype(str).isin(heldout | ambiguous)]
        cloud_cols = [c for c in ("dns-reverse", "as-name", "as-info", "whois-info") if c in df.columns]
        cloud = pd.Series(False, index=df.index)
        for c in cloud_cols:
            cloud |= df[c].astype(str).str.contains(CLOUD_HOST_PAT, case=False, na=False, regex=True)
        n_cloud = int(cloud.sum())
        df_nocloud = df[~cloud]
        cloud_filtered = True
        if len(df_nocloud) < 5:          # keep the class usable, but say so
            df_nocloud, cloud_filtered = df, False
        df = df_nocloud
        rich_cols = ["cert-subjects", "hw-vendors", "sw-vendors", "http-bodys", "os-vendor"]
        have = [c for c in rich_cols if c in df.columns]
        mask = pd.Series(True, index=df.index)
        for c in have:
            mask &= df[c].notna() & (df[c].astype(str).str.len() > 3)
        sub = df[mask]
        fallback = False
        if len(sub) < CANDIDATES_PER_TYPE:
            sub = df            # thin pool (e.g. CONTROLLER) — recorded, not silent
            fallback = True
        idx = rng.choice(len(sub), size=min(CANDIDATES_PER_TYPE, len(sub)), replace=False)
        rows = sub.iloc[idx]
        cand[t] = [
            {"ip": r["ip"], "fingerprint": {k: (None if pd.isna(v) else v) for k, v in r.items()}}
            for _, r in rows.iterrows()
        ]
        meta[t] = {"rows_total": n_raw, "rows_after_exclusions": len(df),
                   "rows_dropped_as_cloud_hosts": n_cloud,
                   "cloud_filter_skipped": not cloud_filtered,
                   "rich_pool": int(len(sub)), "rich_filter_bypassed": fallback,
                   "candidates": len(cand[t])}
        print(f"{t:22} pool {len(sub):6}/{len(df):6} (raw {n_raw:6}, −{n_cloud} cloud)"
              f"{'  [rich filter bypassed]' if fallback else ''}"
              f"{'  [cloud filter skipped: pool too small]' if not cloud_filtered else ''}"
              f" -> {len(cand[t])} candidates")
    json.dump(cand, open(os.path.join(WORK, "candidates.json"), "w"))
    json.dump(meta, open(os.path.join(WORK, "candidate_meta.json"), "w"), indent=1)
    return cand


def run_rank(rank, world_size):
    cand_path = os.path.join(WORK, "candidates.json")
    if not os.path.exists(cand_path):
        raise SystemExit(f"{cand_path} not found — run --candidates first (and note {WORK} is tmpfs)")
    cand = json.load(open(cand_path))
    flat = [(t, c) for t in sorted(cand) for c in cand[t]]
    mine = flat[rank::world_size]
    print(f"[rank {rank}] {len(mine)}/{len(flat)} candidates", flush=True)

    from unseen import UnseenDeviceDetector
    det = UnseenDeviceDetector(adapter_path=ADAPTER, gpu=0, load_in_4bit=True)
    budget = int(det.classification_metadata.get("fingerprint_token_budget", 25454))
    out = {}
    for i, (t, c) in enumerate(mine):
        fp_text = build_fingerprint_info_text(c["fingerprint"]) or "(no fingerprint info)"
        if len(fp_text) > 80000:
            ids = det.tokenizer(fp_text, truncation=True, max_length=budget - 512).input_ids
            fp_text = det.tokenizer.decode(ids, skip_special_tokens=True)
        prompt = build_unseen_type_vendor_classification_prompt(
            fp_text, det.rag_devices, det.known_vendors_by_type)
        try:
            text, conf = det._generate_classification(prompt)
            parsed = det._parse_response(text)
            nov = det._classification_novelty_result(parsed, conf)
            out[f"{t}|{c['ip']}"] = {
                "type": t, "ip": c["ip"],
                "classified_type": nov["classified_type"],
                "classified_vendor": nov["classified_vendor"],
                "type_confidence": float(nov["type_confidence"]),
                "vendor_confidence": float(nov["vendor_confidence"]),
                "new_type_probability": float(nov["new_type_probability"]),
                "new_vendor_probability": float(nov["new_vendor_probability"]),
                "is_unseen": bool(nov["is_unseen"]),
            }
        except Exception as exc:
            out[f"{t}|{c['ip']}"] = {"type": t, "ip": c["ip"], "error": str(exc)[:150]}
        if (i + 1) % 20 == 0:
            print(f"[rank {rank}] {i+1}/{len(mine)}", flush=True)
    _atomic_json_dump(os.path.join(WORK, f"preds_{rank}.json"), out)
    print(f"[rank {rank}] wrote {len(out)}", flush=True)


def materialize_validation_cases():
    """Persist every fixed-seed random candidate without showcase ranking."""
    cand_path = os.path.join(WORK, "candidates.json")
    if not os.path.exists(cand_path):
        raise SystemExit(f"{cand_path} not found — run --candidates first")
    cand = json.load(open(cand_path))
    predictions = {}
    for name in os.listdir(WORK):
        if name.startswith("preds_") and name.endswith(".json"):
            predictions.update(json.load(open(os.path.join(WORK, name))))

    for device_type, candidates in cand.items():
        cases = []
        for candidate in candidates:
            row = predictions.get(f"{device_type}|{candidate['ip']}")
            if row is None:
                adapter_result = {"error": "adapter_prediction_missing"}
            elif "error" in row:
                adapter_result = {"error": row["error"]}
            else:
                adapter_result = {key: row[key] for key in (
                    "classified_type", "classified_vendor", "type_confidence",
                    "vendor_confidence", "new_type_probability",
                    "new_vendor_probability",
                )}
            cases.append({
                "ip": candidate["ip"],
                "device_type": device_type,
                "fingerprint": candidate["fingerprint"],
                "result": adapter_result,
                "selection": "fixed_seed_random_validation",
            })
        out_path = os.path.join(DEMO_DIR, device_type, "validation_cases.json")
        _atomic_json_dump(out_path, cases)
        print(f"{device_type:22} validation={len(cases):3} -> {out_path}")


def run_merge():
    cand_path = os.path.join(WORK, "candidates.json")
    if not os.path.exists(cand_path):
        raise SystemExit(f"{cand_path} not found — run --candidates then --rank first")
    cand = json.load(open(cand_path))
    cand_meta = {}
    mp = os.path.join(WORK, "candidate_meta.json")
    if os.path.exists(mp):
        cand_meta = json.load(open(mp))
    P = {}
    for f in os.listdir(WORK):
        if f.startswith("preds_"):
            P.update(json.load(open(os.path.join(WORK, f))))

    # drift detector (fixed artifacts) for supplementary scoring
    from drift import DriftDetector
    from path_config import DRIFT_OUTPUT_DIR
    det_drift = DriftDetector(model_dir=DRIFT_OUTPUT_DIR)

    # CAMERA-only local retrieval neighbours (production MultiLevelRetrieval
    # over the built npz store; other types have no npz store on this box)
    neighbours = {}
    npz_dir = os.path.join(LOCAL_VECTOR_DB_DIR, "local_npz")
    if os.path.exists(os.path.join(npz_dir, "CAMERA_embeddings.npy")):
        try:
            from retrieval import MultiLevelRetrieval
            mr = MultiLevelRetrieval(llm="DEEPSEEK", gpu=0, whether_milvus=False)
            for c in cand.get("CAMERA", []):
                try:
                    lr = mr.local_retrieval(c["fingerprint"], top_k=5)
                    neighbours[c["ip"]] = [
                        {"ip": d.get("ip"), "device_type": d.get("device_type"),
                         "similarity": round(float(d.get("similarity_score", 0)), 4)}
                        for d in lr.get("similar_devices", [])]
                except Exception as exc:
                    print("neighbour fail", c["ip"], str(exc)[:80])
        except Exception as exc:
            print("retrieval unavailable:", str(exc)[:120])

    summary = {}
    for t in sorted(cand):
        rows = [P[f"{t}|{c['ip']}"] for c in cand[t] if f"{t}|{c['ip']}" in P]
        cand_by_ip = {str(c["ip"]): c for c in cand[t]}
        # The fixed-seed random candidate pool is the validation set. Persist all
        # cases before any correctness/confidence ranking so fullflow evaluation
        # cannot silently become a curated showcase.
        validation_cases = []
        for candidate in cand[t]:
            row = P.get(f"{t}|{candidate['ip']}")
            if row is None:
                adapter_result = {"error": "adapter_prediction_missing"}
            elif "error" in row:
                adapter_result = {"error": row["error"]}
            else:
                adapter_result = {key: row[key] for key in (
                    "classified_type", "classified_vendor", "type_confidence",
                    "vendor_confidence", "new_type_probability",
                    "new_vendor_probability",
                )}
            validation_cases.append({
                "ip": candidate["ip"],
                "device_type": t,
                "fingerprint": candidate["fingerprint"],
                "result": adapter_result,
                "selection": "fixed_seed_random_validation",
            })

        ok = [r for r in rows if "error" not in r and r["classified_type"].strip().upper() == t]
        ok.sort(key=lambda r: -r["type_confidence"])
        # prefer vendor diversity in the showcase (three instances of one vendor
        # demonstrate less than three different ones), falling back to confidence
        chosen, seen_v = [], set()
        for r in ok:
            v = (r["classified_vendor"] or "").strip().lower()
            if v and v not in seen_v:
                chosen.append(r)
                seen_v.add(v)
            if len(chosen) >= TOP_K:
                break
        if len(chosen) < TOP_K:
            chosen += [r for r in ok if r not in chosen][:TOP_K - len(chosen)]
        tdir = os.path.join(DEMO_DIR, t)
        os.makedirs(tdir, exist_ok=True)
        _atomic_json_dump(os.path.join(tdir, "validation_cases.json"), validation_cases)
        cases = []
        for r in chosen:
            c = cand_by_ip[str(r["ip"])]
            drift = {}
            try:
                dd = det_drift.detect_query_device(c["fingerprint"])
                drift = {"drift_score": round(float(dd.get("drift_score", 0)), 4),
                         "is_drift": bool(dd.get("is_drift")),
                         # keep the full attribution; the UI slices for display
                         "drifted_perspectives": list(dd.get("drifted_perspectives", []))}
            except Exception as exc:
                drift = {"error": str(exc)[:100]}
            case = {"ip": r["ip"], "device_type": t,
                    "fingerprint": c["fingerprint"],
                    "result": {k: r[k] for k in ("classified_type", "classified_vendor",
                                                 "type_confidence", "vendor_confidence",
                                                 "new_type_probability", "new_vendor_probability")},
                    "drift": drift}
            if r["ip"] in neighbours:
                case["local_retrieval_neighbours"] = neighbours[r["ip"]]
            cases.append(case)
        _atomic_json_dump(os.path.join(tdir, "cases.json"), cases)
        summary[t] = {"candidates": len(cand[t]), "adapter_predictions": len(rows),
                      "correct": len(ok),
                      "top_conf": round(chosen[0]["type_confidence"], 4) if chosen else None,
                      "cases": len(cases), **cand_meta.get(t, {})}
        print(f"{t:22} cand={len(rows):3} correct={len(ok):3} "
              f"top_conf={summary[t]['top_conf']} saved={len(cases)}")
    _atomic_json_dump(os.path.join(DEMO_DIR, "selection_summary.json"), summary)
    print("\nwrote", DEMO_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int)
    p.add_argument("--world_size", type=int, default=8)
    p.add_argument("--candidates", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--materialize-validation", action="store_true")
    a = p.parse_args()
    if a.materialize_validation:
        materialize_validation_cases()
    elif a.candidates:
        build_candidates()
    elif a.merge:
        run_merge()
    elif a.rank is not None:
        run_rank(a.rank, a.world_size)
