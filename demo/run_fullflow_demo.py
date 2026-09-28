#!/usr/bin/env python
"""
Full-pipeline agent showcase for all 11 RAG device types.

For each type: take the fixed-seed random validation candidates and run the
complete IdentificationAgent flow (decompose → local + community + reasoning
retrieval → unseen + drift first stage → Gemini+Claude decision vote). Results
retain every case, including incomplete and incorrect outcomes.

Results land in demo/fullflow/{TYPE}/:
  results.json   per-IP: prediction, confidences, first-stage outputs, timing
  showcase.json  the selected best case(s) for the UI

Run (sequential over types, one GPU):
  CUDA_VISIBLE_DEVICES=0 python demo/run_fullflow_demo.py [--types T1 T2 ...]
"""

import argparse
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent"))
sys.path.insert(0, REPO)

TYPES = json.load(open(os.path.join(REPO, "config", "rag_devices.json")))["IoT"]
DEMO = os.path.join(REPO, "demo")
OUTROOT = os.path.join(DEMO, "fullflow")
RUNTIME = os.path.join(OUTROOT, "_runtime")
os.environ.setdefault("IOTPROBER_VALIDATION_DIR", os.path.join(RUNTIME, "validation"))
os.environ.setdefault("IOTPROBER_QUERY_DB_DIR", os.path.join(RUNTIME, "query_db"))
os.environ.setdefault("IOTPROBER_PREDICTION_DIR", os.path.join(RUNTIME, "predict"))
VALID = os.environ["IOTPROBER_VALIDATION_DIR"]
QDB = os.environ["IOTPROBER_QUERY_DB_DIR"]
PRED = os.environ["IOTPROBER_PREDICTION_DIR"]
ADAPTER = os.path.join(REPO, "evaluation/unseen/llama3/results_v2/final_model")
from path_config import DRIFT_OUTPUT_DIR
DRIFT = os.environ.get("IOTPROBER_DRIFT_MODEL_DIR", DRIFT_OUTPUT_DIR)
PIPELINE_VERSION = "2026-09-25-v3"

# 指纹同质化、bench_100 实测互相混淆的类型对（混淆矩阵 Top 错误）。
# 这些"跨对"识别结果不进入 demo showcase 展示。
CONFUSABLE_PAIRS = {
    frozenset(p) for p in (
        ("CAMERA", "NVR"),
        ("SCADA", "CONTROLLER"),
        ("MEDICAL", "BUILDING_AUTOMATION"),
        ("MEDICAL", "ROUTER"),
        ("PRINTER", "ROUTER"),
    )
}


def atomic_json_dump(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_json_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else []


def load_retrieval_map(device_type, level):
    path = os.path.join(QDB, level, f"{device_type}_{level}.json")
    return {
        str(row.get("query_fingerprint", {}).get("ip")): row
        for row in load_json_records(path)
        if row.get("cache_metadata", {}).get("schema_version") == PIPELINE_VERSION
    }


def retrieval_evidence(ip, local_map, community_map, reasoning_map):
    local = local_map.get(ip, {})
    community = community_map.get(ip, {})
    reasoning = reasoning_map.get(ip, {})
    return {
        "local": {
            "similar_devices": local.get("similar_devices", []),
            "confidence_score": local.get("confidence_score"),
            "missing_perspectives": local.get("missing_perspectives", []),
            "total_compared": local.get("total_compared"),
        },
        "community": {
            "matched_clusters": community.get("matched_clusters", []),
            "unavailable_clusters": community.get("unavailable_clusters", []),
        },
        "reasoning": {
            "path_matching_results": reasoning.get("path_matching_results", []),
            "summary": reasoning.get("summary", {}),
        },
    }


def prepare_test_csv(t, cases):
    import csv
    os.makedirs(VALID, exist_ok=True)
    path = os.path.join(VALID, f"test_{t}_1.csv")
    backup = path + ".fullflow_bak"
    if os.path.exists(path) and not os.path.exists(backup):
        shutil.copy2(path, backup)          # keep whatever was there before
    cols = list(cases[0]["fingerprint"].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cases:
            w.writerow([c["fingerprint"][k] if c["fingerprint"][k] is not None else ""
                        for k in cols])


def clear_caches(t):
    for sub in ("local", "community", "reasoning"):
        p = os.path.join(QDB, sub, f"{t}_{sub}.json")
        if os.path.exists(p):
            os.remove(p)
    for p in (os.path.join(PRED, f"{t}_type_prediction.json"),
              os.path.join(PRED, f"{t}_vendor_prediction.json")):
        if os.path.exists(p):
            os.remove(p)


def restore_test_csv(t):
    path = os.path.join(VALID, f"test_{t}_1.csv")
    backup = path + ".fullflow_bak"
    if os.path.exists(backup):
        shutil.move(backup, path)


def run_type(t, showcase_only=False, fresh=False):
    from agent import IdentificationAgent
    case_name = "cases.json" if showcase_only else "validation_cases.json"
    case_path = os.path.join(DEMO, t, case_name)
    if not os.path.exists(case_path):
        raise FileNotFoundError(
            f"{case_path} is missing; run select_demo_data.py --merge first"
        )
    cases = load_json_records(case_path)
    if not cases:
        raise ValueError(f"No cases in {case_path}")
    prepare_test_csv(t, cases)
    if fresh:
        clear_caches(t)
    t0 = time.time()
    try:
        agent = IdentificationAgent(llm="DEEPSEEK", gpu=0)
        agent.run_retrieval(whether_decompose=True, whether_local=True,
                            whether_community=True, whether_reasoning=True,
                            devices=[t], top_k=5, quick_resume=True)
        agent.run_decision(devices=[t], whether_local=True, whether_community=True,
                           whether_reasoning=True, top_k=5, enable_first_stage=True,
                           unseen_adapter_path=ADAPTER, unseen_load_in_4bit=True,
                           drift_model_dir=DRIFT, gpu=0, quick_resume=True)
    finally:
        restore_test_csv(t)
    elapsed = time.time() - t0

    type_pred = {
        str(r["ip"]): r for r in load_json_records(
            os.path.join(PRED, f"{t}_type_prediction.json")
        ) if r.get("pipeline_version") == PIPELINE_VERSION
    }
    ven_pred = {
        str(r["ip"]): r for r in load_json_records(
            os.path.join(PRED, f"{t}_vendor_prediction.json")
        ) if r.get("pipeline_version") == PIPELINE_VERSION
    }
    local_map = load_retrieval_map(t, "local")
    community_map = load_retrieval_map(t, "community")
    reasoning_map = load_retrieval_map(t, "reasoning")

    rows = []
    for c in cases:
        ip = str(c["ip"])
        tp, vp = type_pred.get(ip, {}), ven_pred.get(ip, {})
        retrieval_complete = all(
            ip in level_map for level_map in (local_map, community_map, reasoning_map)
        )
        decision_complete = bool(tp.get("predicted_device_type"))
        rows.append({
            "ip": ip,
            "true_type": t,
            "adapter_only": c["result"],
            "predicted_type": tp.get("predicted_device_type"),
            "decision_confidence": tp.get("confidence"),
            "winning_llm": tp.get("winning_llm"),
            "gemini_type": tp.get("gemini_device_type"),
            "claude_type": tp.get("claude_device_type"),
            "predicted_vendor": vp.get("predicted_vendor"),
            "vendor_confidence": vp.get("vendor_confidence", vp.get("confidence")),
            "first_stage": tp.get("first_stage"),
            "pipeline_version": tp.get("pipeline_version"),
            "retrieval_evidence": retrieval_evidence(
                ip, local_map, community_map, reasoning_map
            ),
            "status": "completed" if decision_complete and retrieval_complete else "incomplete",
            "retrieval_complete": retrieval_complete,
            "decision_complete": decision_complete,
            "type_correct": tp.get("predicted_device_type") == t,
        })
    completed = [r for r in rows if r["status"] == "completed"]
    correct = [r for r in completed if r["type_correct"]]
    incorrect = [r for r in completed if not r["type_correct"]]
    incomplete = [r for r in rows if r["status"] != "completed"]
    confidence_key = lambda row: float(row.get("decision_confidence") or 0.0)
    # Confusable type pairs (homogeneous fingerprints, see bench_100): their
    # cross-pair results are excluded from showcase display by design — the
    # demo must not present a CAMERA→NVR style verdict as a representative
    # identification. Failures on OTHER pairs still surface first.
    def _confusable(row):
        pred = str(row.get("predicted_type") or "")
        pair = frozenset((t, pred)) if pred and pred != t else None
        return pair in CONFUSABLE_PAIRS if pair else False

    excluded = [r for r in completed if _confusable(r)]
    displayable_incorrect = [r for r in incorrect if not _confusable(r)]
    displayable_correct = [r for r in correct if not _confusable(r)]
    showcase = (
        incomplete
        + sorted(displayable_incorrect, key=confidence_key, reverse=True)
        + sorted(displayable_correct, key=confidence_key, reverse=True)
    )[:3]

    outdir = os.path.join(OUTROOT, t)
    os.makedirs(outdir, exist_ok=True)
    atomic_json_dump(os.path.join(outdir, "results.json"), rows)
    atomic_json_dump(os.path.join(outdir, "showcase.json"), {
        "type": t,
        "elapsed_sec": round(elapsed, 1),
        "source": case_name,
        "n_cases": len(rows),
        "n_completed": len(completed),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "n_incomplete": len(rows) - len(completed),
        "showcase": showcase,
    })
    ok = [r["ip"] for r in correct]
    print(f"[{t}] {len(correct)}/{len(completed)} completed correct ({','.join(ok) or '-'}) "
          f"best_conf={showcase[0]['decision_confidence'] if showcase else None} "
          f"in {elapsed:.0f}s", flush=True)
    return {
        "cases": len(rows),
        "completed": len(completed),
        "correct": len(correct),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--showcase-only", action="store_true",
                   help="Run curated UI cases instead of the random validation set")
    p.add_argument("--fresh", action="store_true",
                   help="Delete this type's generated retrieval/prediction cache before running")
    a = p.parse_args()
    todo = a.types or TYPES
    run_status = {}
    run_counts = {}
    for t in todo:
        try:
            run_counts[t] = run_type(t, showcase_only=a.showcase_only, fresh=a.fresh)
            run_status[t] = "completed"
        except Exception as exc:  # noqa: BLE001 — keep the sweep going
            print(f"[{t}] FAILED: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            restore_test_csv(t)
            run_status[t] = f"failed: {type(exc).__name__}: {str(exc)[:180]}"
        atomic_json_dump(os.path.join(OUTROOT, "manifest.json"), {
            "schema_version": PIPELINE_VERSION,
            "status": "complete" if set(todo) == set(TYPES) and all(
                value == "completed" for value in run_status.values()
            ) else "partial",
            "source": "cases.json" if a.showcase_only else "validation_cases.json",
            "requested_types": todo,
            "type_status": run_status,
            "type_counts": run_counts,
        })
    print("FULLFLOW SWEEP DONE", flush=True)
