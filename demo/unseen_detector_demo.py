#!/usr/bin/env python3
"""Single-file runnable demo: unseen detector on an ALARM device.

This is the artifact-evaluation example for the paper's first-stage unseen
detector. For one fixed-seed validation device it shows, end to end:

  1. the recorded three-layer retrieval evidence — LOCAL neighbours,
     COMMUNITY cluster matches and REASONING-path matches — all non-empty
     (schema 2026-09-25-v3 caches of the reference run);
  2. a live run of UnseenDeviceDetector (Llama-3.1-8B-Instruct + the released
     LoRA adapter, 4-bit) on exactly that evidence;
  3. a RANKED top-3 vendor prediction (detector label first, then
     evidence-scored candidates from the fingerprint and the community
     cluster report, lexicon = the adapter's known_vendors.json contract),
     plus a top-3 check against a reference vendor label.

No LLM API keys are needed. With --skip-model the detector call is skipped
(and rank 1 comes from evidence only) so the evidence part runs on any CPU.

Usage (from the repository root):
  python demo/unseen_detector_demo.py                          # ALARM showcase
  python demo/unseen_detector_demo.py --ip 107.89.86.78
  python demo/unseen_detector_demo.py --skip-model             # no GPU needed
  python demo/unseen_detector_demo.py --true-vendor Bosch      # override label

Expected resources (see evaluation/REBUTTAL_ADDENDUM.md §1.3 on the dataset mirror):
  demo/evidence/ALARM_{local,community,reasoning}.json  (v3 evidence cache, shipped)
  demo/evidence/known_vendors.json                      (vendor lexicon, shipped)
  demo/ALARM/validation_cases.json                      (fixed-seed pool, shipped)
  ./Meta-Llama-3.1-8B-Instruct/                         (gated HF download)
  evaluation/unseen/llama3/results_v2/final_model/      (released adapter, HF dataset)

`--skip-model` therefore runs from a fresh clone with no further downloads.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent")]

# Evidence caches + vendor lexicon ship with the repository under demo/evidence/;
# a full local run layout (agent/query_db/, released adapter dir) is used as fallback.
EVIDENCE_DIR = os.path.join(ROOT, "demo", "evidence")
QUERY_DB = os.path.join(ROOT, "agent", "query_db")
DEFAULT_BASE = os.path.join(ROOT, "Meta-Llama-3.1-8B-Instruct")
DEFAULT_ADAPTER = os.path.join(ROOT, "evaluation", "unseen", "llama3",
                               "results_v2", "final_model")
KNOWN_VENDORS_CANDIDATES = [
    os.path.join(EVIDENCE_DIR, "known_vendors.json"),
    os.path.join(DEFAULT_ADAPTER, "known_vendors.json"),
]

# Vendor-bearing fingerprint/report fields used for evidence scoring.
VENDOR_FIELDS = ["cert-issuers", "cert-subjects", "cert-info", "hw-products",
                 "hw-vendor", "http-tags", "http-info", "http-titles",
                 "sw-products", "sw-vendor"]

# Paper Table 3 dominant vendor of ALARM — the DEFAULT top-3 reference label.
# (The demo pool carries no per-IP vendor label; override with --true-vendor.)
DEVICE = "ALARM"
DOMINANT_VENDOR = "Bosch"


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def fingerprint_id(fp):
    return hashlib.sha256(canon(fp).encode()).hexdigest()[:16]


def load_cases():
    path = os.path.join(ROOT, "demo", DEVICE, "validation_cases.json")
    return json.load(open(path))


def load_evidence():
    """Return {fingerprint_id: entry} for each non-empty retrieval level.

    demo/evidence/ (shipped with the repository) takes precedence; the
    agent/query_db caches of a full local run are the fallback.
    """
    ev = {}
    for lvl in ("local", "community", "reasoning"):
        name = f"{DEVICE}_{lvl}.json"
        candidates = [os.path.join(EVIDENCE_DIR, name),
                      os.path.join(QUERY_DB, lvl, name)]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            sys.exit(f"missing evidence cache: {name} (looked in "
                     f"{', '.join(candidates)}) — re-run "
                     f"`python demo/run_fullflow_demo.py --types {DEVICE}` to "
                     f"rebuild it")
        entries = json.load(open(path))
        ev[lvl] = {fingerprint_id(e.get("query_fingerprint", {})): e for e in entries}
    return ev


def load_known_vendors():
    path = next((p for p in KNOWN_VENDORS_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        sys.exit("missing known_vendors.json (looked in "
                 + ", ".join(KNOWN_VENDORS_CANDIDATES) + ")")
    return json.load(open(path))["known_vendors_by_type"].get(DEVICE, [])


def _hint_regen():
    return ("(no cached entry for this case — regenerate v3 evidence with "
            "`python demo/run_fullflow_demo.py --types ALARM`)")


def show_evidence(case, ev):
    fid = fingerprint_id(case["fingerprint"])
    ok = True

    loc = ev["local"].get(fid)
    print("\n== 1. LOCAL retrieval evidence (cache schema "
          f"{loc['cache_metadata']['schema_version'] if loc else '?'}) ==")
    if loc and loc.get("similar_devices"):
        for n in loc["similar_devices"][:3]:
            print(f"   neighbour {n['ip']:<16} type={n['device_type']:<8} "
                  f"sim={n['similarity_score']:.3f}")
    else:
        ok = False
        print("   ", _hint_regen())

    com = ev["community"].get(fid)
    print(f"\n== 2. COMMUNITY retrieval evidence "
          f"({com['cache_metadata']['schema_version'] if com else 'none'}) ==")
    if com and com.get("matched_clusters"):
        for c in com["matched_clusters"][:2]:
            print(f"   cluster {c.get('cluster_id')} type={c.get('device_type')} "
                  f"sim={c.get('similarity_score')}")
            patterns = c.get("report", {}).get("common_patterns") or {}
            shown = 0
            for field in list(VENDOR_FIELDS) + ["as-name", "os-product", "http-titles"]:
                values = patterns.get(field)
                if values and shown < 4:
                    print(f"     {field}: " + "; ".join(
                        f"{v.get('value')} x{v.get('count')}" for v in values[:2]))
                    shown += 1
    else:
        ok = False
        print("   (empty)")

    rea = ev["reasoning"].get(fid)
    print(f"\n== 3. REASONING-path evidence "
          f"({rea['cache_metadata']['schema_version'] if rea else 'none'}) ==")
    if rea and rea.get("path_matching_results"):
        for p in rea["path_matching_results"][:2]:
            info = p.get("cluster_info", {})
            print(f"   path {p.get('cluster_key')} sim={info.get('similarity_score')} "
                  f"type={info.get('device_type')}")
            gist = str((info.get("report") or {}).get("common_patterns", ""))[:160]
            if gist:
                print(f"     gist: {gist}")
        top = (rea.get("summary") or {}).get("top_cluster") or {}
        print(f"   summary top cluster: {top.get('cluster_key')} "
              f"({(top.get('cluster_info') or {}).get('similarity_score')})")
    else:
        ok = False
        print("   (empty)")

    print(f"\n   evidence non-empty: {'YES' if ok else 'NO — pick another --index'}")
    return ok, (loc, com, rea)


def run_detector(case, loc, com, rea, args):
    print(f"\n== 4. UnseenDeviceDetector (live, "
          f"{'4-bit' if not args.gpu < 0 else 'cpu'}) ==")
    from unseen import UnseenDeviceDetector  # noqa: E402  (agent/unseen.py)
    det = UnseenDeviceDetector(
        model_path=args.base_model,
        adapter_path=DEFAULT_ADAPTER,
        gpu=args.gpu,
        load_in_4bit=args.gpu >= 0,
        max_new_tokens=512,
    )
    t0 = time.time()
    out = det.detect_unseen(rea, loc, com, allow_web_search=False)
    print(f"   new_type_probability  = {out['new_type_probability']}")
    print(f"   new_vendor_probability= {out['new_vendor_probability']}")
    print(f"   is_unseen             = {out['is_unseen']}")
    if out.get("classified_vendor") is not None:
        print(f"   classified_type       = {out.get('classified_type')} "
              f"(conf {out.get('type_confidence')}, "
              f"registry match: {out.get('matched_rag_type')})")
        print(f"   classified_vendor     = {out.get('classified_vendor')} "
              f"(conf {out.get('vendor_confidence')}, "
              f"registry match: {out.get('matched_known_vendor')})")
    else:
        print(f"   detector vendor       = {out.get('predicted_vendor')}")
    print(f"   latency               = {time.time() - t0:.1f}s "
          f"(model {out.get('generation_time_sec', float('nan'))}s)")
    return out


def evidence_vendor_scores(case, com):
    """Count lexicon-vendor mentions across fingerprint + community report."""
    lex = load_known_vendors()
    lex = sorted(lex, key=len, reverse=True)  # longest match first
    scores, sources = {}, {}

    def scan(text, weight, where):
        low = str(text).lower()
        for v in lex:
            if v.lower() in low:
                scores[v] = scores.get(v, 0.0) + weight
                sources.setdefault(v, []).append(where)

    for f in VENDOR_FIELDS:
        if case["fingerprint"].get(f):
            scan(case["fingerprint"][f], 3.0, f"fingerprint:{f}")
    if com and com.get("matched_clusters"):
        for c in com["matched_clusters"][:3]:
            sim = float(c.get("similarity_score") or 0)
            for field, values in (c.get("report", {}).get("common_patterns") or {}).items():
                if field in VENDOR_FIELDS:
                    for v in values[:5]:
                        scan(v.get("value", ""), 1.0 * sim, f"community:{field}")
    return scores, sources


def ranked_top3(detector_out, case, com):
    scores, sources = evidence_vendor_scores(case, com)
    ranking = []
    if detector_out:
        # For a known vendor the novelty fields are "none" by design; the
        # classifier's classified_vendor is the detector's top-1 vendor label.
        v = detector_out.get("classified_vendor")
        score = detector_out.get("vendor_confidence")
        src = "detector (classified_vendor, conf)"
        if not v:  # genuinely unseen vendor
            v = detector_out.get("predicted_vendor")
            score = detector_out.get("new_vendor_probability")
            src = "detector (novel vendor, prob)"
        if v and v not in ("none", "None", "UNKNOWN", ""):
            ranking.append((v, float(score or 0), src))
    for v, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        if len(ranking) >= 3:
            break
        if any(v == r[0] for r in ranking):
            continue
        ranking.append((v, round(s, 2), "evidence: " + ",".join(
            sorted(set(sources[v]))[:3])))
    while len(ranking) < 3:
        ranking.append(("-", 0.0, "no further evidence"))
    return ranking


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=32,
                    help="case index in the pool (default 32: the case with the "
                         "strongest ALARM community match)")
    ap.add_argument("--ip", help="select the case by IP instead of --index")
    ap.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--skip-model", action="store_true",
                    help="evidence + ranking only, no GPU / model needed")
    ap.add_argument("--true-vendor", default=None,
                    help="reference vendor for the top-3 check "
                         f"(default: Table-3 dominant vendor "
                         f"{DOMINANT_VENDOR})")
    args = ap.parse_args()

    cases = load_cases()
    case = next((c for c in cases if c.get("ip") == args.ip), None) if args.ip \
        else cases[args.index]
    ref = args.true_vendor or DOMINANT_VENDOR
    ref_note = "override" if args.true_vendor else (
        "dominant ALARM vendor, paper Table 3; "
        "the pool carries no per-IP vendor label")
    print(f"device={DEVICE}  ip={case['ip']}  "
          f"fingerprint_id={fingerprint_id(case['fingerprint'])}")
    print(f"top-3 reference label: {ref!r} ({ref_note})")

    ev = load_evidence()
    ok, (loc, com, rea) = show_evidence(case, ev)

    det_out = None
    if args.skip_model:
        print("\n== 4. UnseenDeviceDetector: SKIPPED (--skip-model) ==")
    elif rea is None:
        print("\n== 4. UnseenDeviceDetector: SKIPPED (no reasoning evidence for "
              "this case — the detector consumes reasoning-path output; "
              "see the hint in section 3) ==")
    else:
        det_out = run_detector(case, loc, com, rea, args)

    ranking = ranked_top3(det_out, case, com)
    print("\n== 5. RANKED vendor predictions (top-3) ==")
    for i, (v, s, src) in enumerate(ranking, 1):
        print(f"   {i}. {v:<20} score={s:<8} source={src}")

    hit = next((i for i, (v, _, _) in enumerate(ranking, 1) if v == ref), None)
    verdict = f"HIT (rank {hit})" if hit else "MISS (not in top-3)"
    print(f"\n   TOP-3 CHECK: reference {ref!r} -> {verdict}")


if __name__ == "__main__":
    main()
