#!/usr/bin/env python3
"""
LLM-based device type and Top-3 vendor identification using Claude Opus (CLAUDEBaseline).

Pipeline:
  For each device type in TYPES:
    1. Read IP fingerprints from evaluation/validation/46_features/test_{TYPE}_1.csv
    2. Build concise fingerprint text per IP
    3. Call chat_with_llm("CLAUDEBaseline", ...) to get device type + Top-3 vendors
    4. Save predict_{dev}.json  (format: [{ip, type, top3_vendor}, ...])

Resume support: if predict_{dev}.json already exists, processed IPs are skipped.
"""

import os
import sys
import json
import time
import csv

# ── Path setup ─────────────────────────────────────────────────────────────────
PREDICT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.abspath(os.path.join(PREDICT_DIR, "..", "..", "..", ".."))
FEAT_DIR     = os.path.join(BASE_DIR, "evaluation", "validation", "46_features")
sys.path.insert(0, BASE_DIR)

from llm import LLM

# ── Config ─────────────────────────────────────────────────────────────────────
LLM_KEY = "CLAUDEBaseline"

TYPES = [
    "Camera", "Printer", "Scada", "Router", "NAS",
    "NVR", "Power_meter", "Building_Automation", "Medical",
]

FEAT_FILE = {
    "Camera":              "test_CAMERA_1.csv",
    "Printer":             "test_PRINTER_1.csv",
    "Scada":               "test_SCADA_1.csv",
    "Router":              "test_ROUTER_1.csv",
    "NAS":                 "test_NAS_1.csv",
    "NVR":                 "test_NVR_1.csv",
    "Power_meter":         "test_POWER_METER_1.csv",
    "Building_Automation": "test_BUILDING_AUTOMATION_1.csv",
    "Medical":             "test_MEDICAL_1.csv",
}

# LLM type label → display format used in predict_{dev}.json
TYPE_NORM = {
    "CAMERA":              "Camera",
    "PRINTER":             "Printer",
    "SCADA":               "Scada",
    "ROUTER":              "Router",
    "NAS":                 "NAS",
    "NVR":                 "NVR",
    "POWER_METER":         "Power_meter",
    "BUILDING_AUTOMATION": "Building_Automation",
    "MEDICAL":             "Medical",
    "UNKNOWN":             "UNKNOWN",
}

ALL_DEVICE_TYPES = list(TYPE_NORM.keys())

SAVE_INTERVAL = 20   # save checkpoint every N processed IPs
REQUEST_DELAY = 0.5  # seconds between API calls

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an expert in IoT device fingerprinting, classification, and vendor identification.
Given a device fingerprint derived from network scanning data, you must:
1. Identify the **device type** — choose exactly one label from the candidates below.
2. Identify the **vendor** (manufacturer / software brand) — return up to 3 candidates.

**Device type candidates**: {', '.join(ALL_DEVICE_TYPES)}
- Use "UNKNOWN" only if you cannot determine the type.

**Vendor identification rules**:
- Return 1–3 vendors clearly supported by the evidence.
- If no vendor can be determined, return [{"vendor": "unknown"}].
- Do NOT pad to 3 vendors if fewer are clearly supported by the evidence.

Respond ONLY in valid JSON with this exact structure:
{{
  "device_type": "<TYPE or UNKNOWN>",
  "vendor_top3": [
    {{"vendor": "<name>"}},
    ...
  ]
}}
Write all text in English."""


def build_fingerprint_text(fp: dict) -> str:
    """Select key fields from a 46-feature row and format for the LLM prompt."""
    fields = [
        ("Service Distribution",     "service-distribution"),
        ("Software Vendors",         "sw-vendors"),
        ("Software Products",        "sw-products"),
        ("Software Versions",        "sw-versions"),
        ("Software Info",            "sw-info"),
        ("Hardware Vendors",         "hw-vendors"),
        ("Hardware Products",        "hw-products"),
        ("Hardware Info",            "hw-info"),
        ("OS Vendor",                "os-vendor"),
        ("OS Product",               "os-product"),
        ("OS Version",               "os-version"),
        ("TLS Certificate Subjects", "cert-subjects"),
        ("TLS Certificate Issuers",  "cert-issuers"),
        ("TLS Certificate Info",     "cert-info"),
        ("TLS Versions",             "tls-versions"),
        ("HTTP Tags",                "http-tags"),
        ("HTTP Favicon Hashes",      "http-favicon-hashes"),
        ("DNS Reverse",              "dns-reverse"),
        ("AS Name",                  "as-name"),
        ("WHOIS Organization",       "whois-organization-name"),
        ("Country",                  "loc-country"),
    ]
    lines = []
    for label, key in fields:
        val = (fp.get(key) or "").strip()
        if val:
            if len(val) > 500:
                val = val[:500] + "...[truncated]"
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "(No fingerprint data available)"


def build_user_prompt(ip: str, fp_text: str) -> str:
    return (
        f"Analyze the following IoT device fingerprint and identify its type and vendor.\n\n"
        f"**Device IP**: {ip}\n\n"
        f"**Device Fingerprint**:\n{fp_text}\n\n"
        f"Return your analysis as JSON."
    )


def normalise_type(raw: str) -> str:
    """Map LLM-returned type string to canonical display format."""
    key = raw.upper().strip().replace(" ", "_")
    return TYPE_NORM.get(key, raw)


def extract_vendor_list(vendor_top3) -> list:
    """Convert vendor_top3 (list of {vendor}) to plain name list (top 3)."""
    names = []
    if isinstance(vendor_top3, list):
        for v in vendor_top3:
            if isinstance(v, dict):
                name = v.get("vendor", "unknown")
                if name.lower() != "unknown":
                    names.append(name)
    if not names:
        names = ["unknown"]
    return names[:3]


# ── Main ───────────────────────────────────────────────────────────────────────

print("Initializing LLM client...")
llm_client = LLM()

for dev in TYPES:
    feat_path = os.path.join(FEAT_DIR, FEAT_FILE[dev])
    out_path  = os.path.join(PREDICT_DIR, f"predict_{dev}.json")

    if not os.path.exists(feat_path):
        print(f"[SKIP] {feat_path} not found")
        continue

    # ── Load feature CSV ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Processing [{dev}]  ←  {FEAT_FILE[dev]}")
    print("=" * 60)

    rows = []
    with open(feat_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows)} rows")

    # ── Resume: load existing results ─────────────────────────────────────────
    existing: dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                for entry in json.load(f):
                    if "type" in entry:
                        existing[entry["ip"]] = entry
            print(f"  Resuming: {len(existing)} IPs already done")
        except Exception as exc:
            print(f"  [WARN] Could not load existing results: {exc}")

    results: list[dict] = list(existing.values())

    # ── Inference loop ─────────────────────────────────────────────────────────
    total   = len(rows)
    new_cnt = 0

    for idx, row in enumerate(rows, 1):
        ip = str(row.get("ip", "")).strip()
        if not ip:
            continue

        if ip in existing:
            print(f"  [{idx}/{total}] Cached  {ip}")
            continue

        fp_text  = build_fingerprint_text(row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(ip, fp_text)},
        ]

        try:
            response = llm_client.chat_with_llm(LLM_KEY, messages, whether_json=True)

            pred_type = normalise_type(response.get("device_type", "UNKNOWN"))
            vendors   = extract_vendor_list(response.get("vendor_top3", []))

            entry = {
                "ip":          ip,
                "type":        pred_type,
                "top3_vendor": vendors,
            }
            print(f"  [{idx}/{total}] OK     {ip}  type={pred_type}  vendors={vendors}")

        except Exception as exc:
            entry = {
                "ip":          ip,
                "type":        "UNKNOWN",
                "top3_vendor": ["unknown"],
            }
            print(f"  [{idx}/{total}] ERROR  {ip}  {exc}")

        results.append(entry)
        existing[ip] = entry
        new_cnt += 1

        # Periodic checkpoint
        if new_cnt % SAVE_INTERVAL == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  >> Checkpoint saved ({len(results)}/{total})")

        time.sleep(REQUEST_DELAY)

    # ── Final save ─────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_ok  = sum(1 for e in results if e["type"] != "UNKNOWN")
    print(f"\n  [{dev}] Done.  Total={len(results)}  Identified={n_ok}  Saved → {out_path}")

print("\nAll device types processed.")
