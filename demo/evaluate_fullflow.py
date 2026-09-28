#!/usr/bin/env python
"""Evaluate only integrity-versioned fullflow outputs."""

import argparse
import json
import math
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULLFLOW = os.path.join(ROOT, "demo", "fullflow")
SCHEMA_VERSION = "2026-09-25-v3"


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def atomic_dump(path, value):
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def expected_calibration_error(rows, bins=10):
    samples = [
        (float(row["decision_confidence"]), bool(row["type_correct"]))
        for row in rows
        if row.get("decision_confidence") is not None
    ]
    if not samples:
        return None
    total = len(samples)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            sample for sample in samples
            if lower <= sample[0] <= upper and (index == bins - 1 or sample[0] < upper)
        ]
        if not bucket:
            continue
        confidence = sum(sample[0] for sample in bucket) / len(bucket)
        accuracy = sum(sample[1] for sample in bucket) / len(bucket)
        value += len(bucket) / total * abs(accuracy - confidence)
    return round(value, 6)


def summarize(rows):
    completed = [row for row in rows if row.get("status") == "completed"]
    local_available = []
    community_available = []
    reasoning_available = []
    self_matches = 0
    score_violations = 0

    for row in completed:
        true_type = row.get("true_type")
        evidence = row.get("retrieval_evidence") or {}
        neighbors = (evidence.get("local") or {}).get("similar_devices") or []
        if neighbors:
            local_available.append((true_type, neighbors))
        if any(str(item.get("ip")) == str(row.get("ip")) for item in neighbors):
            self_matches += 1
        score_violations += sum(
            1 for item in neighbors
            if not -1.0 <= float(item.get("similarity_score", math.inf)) <= 1.0
        )

        clusters = (evidence.get("community") or {}).get("matched_clusters") or []
        if clusters:
            community_available.append((true_type, clusters[0]))
        paths = (evidence.get("reasoning") or {}).get("path_matching_results") or []
        if paths:
            reasoning_available.append((true_type, paths[0]))

    correct = sum(bool(row.get("type_correct")) for row in completed)
    local_recall = sum(
        any(item.get("device_type") == true_type for item in neighbors)
        for true_type, neighbors in local_available
    )
    local_top1 = sum(
        neighbors[0].get("device_type") == true_type
        for true_type, neighbors in local_available
    )
    community_top1 = sum(
        cluster.get("device_type") == true_type
        for true_type, cluster in community_available
    )
    reasoning_top1 = sum(
        (path.get("cluster_info") or {}).get("device_type") == true_type
        for true_type, path in reasoning_available
    )

    def ratio(numerator, denominator):
        return round(numerator / denominator, 6) if denominator else None

    return {
        "cases": len(rows),
        "completed": len(completed),
        "completion_rate": ratio(len(completed), len(rows)),
        "final_accuracy": ratio(correct, len(completed)),
        "end_to_end_accuracy": ratio(correct, len(rows)),
        "local_recall_at_k": ratio(local_recall, len(completed)),
        "local_top1_accuracy": ratio(local_top1, len(completed)),
        "community_top1_accuracy": ratio(community_top1, len(completed)),
        "reasoning_top1_accuracy": ratio(reasoning_top1, len(completed)),
        "local_available": len(local_available),
        "local_availability": ratio(len(local_available), len(completed)),
        "community_available": len(community_available),
        "community_availability": ratio(len(community_available), len(completed)),
        "reasoning_available": len(reasoning_available),
        "reasoning_availability": ratio(len(reasoning_available), len(completed)),
        "self_match_violations": self_matches,
        "score_range_violations": score_violations,
        "high_confidence_errors": sum(
            not row.get("type_correct") and float(row.get("decision_confidence") or 0) >= 0.9
            for row in completed
        ),
        "expected_calibration_error_10_bins": expected_calibration_error(completed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest = load_json(os.path.join(FULLFLOW, "manifest.json"), {})
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("fullflow outputs do not use the current retrieval schema")
    if manifest.get("status") != "complete" and not args.allow_partial:
        raise SystemExit("fullflow run is not complete; pass --allow-partial for diagnostics")

    by_type = {}
    all_rows = []
    type_status = manifest.get("type_status", {})
    for device_type in manifest.get("requested_types", []):
        if type_status.get(device_type) != "completed":
            continue
        rows = load_json(os.path.join(FULLFLOW, device_type, "results.json"), [])
        rows = [row for row in rows if row.get("pipeline_version") == SCHEMA_VERSION]
        by_type[device_type] = summarize(rows)
        all_rows.extend(rows)

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_status": manifest.get("status"),
        "aggregate": summarize(all_rows),
        "by_type": by_type,
    }
    atomic_dump(os.path.join(FULLFLOW, "evaluation_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
