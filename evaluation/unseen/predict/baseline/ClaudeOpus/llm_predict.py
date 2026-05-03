#!/usr/bin/env python3
"""
LLM-based device type and vendor identification for MEDIA_SERVER and VPN test sets.

Pipeline:
1. Filter test IPs from ipraw files using label files (known IPs only)
2. Build device fingerprint text for each IP
3. Call DeepSeek LLM to identify device type + vendor (Top-3)
4. Save:
   - type_result_method1_{dev}.csv   (ip, true_type, pred_type, true_vendor,
                                       vendor_top1, vendor_top2, vendor_top3)
   - llm_result_{dev}.json           (full LLM output per IP)
"""

import argparse
import os
import sys
import json
import time
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────
PREDICT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(PREDICT_DIR)))
sys.path.append(BASE_DIR)

from llm import LLM

LABEL_DIR      = os.path.join(BASE_DIR, 'platform_data', 'csv', 'label', 'known')
FULL_LABEL_DIR = os.path.join(BASE_DIR, 'platform_data', 'csv', 'label')
IPRAW_DIR      = os.path.join(BASE_DIR, 'platform_data', 'csv')

# DEVICE_TYPES = ['MEDIA_SERVER', 'VPN']

DEVICE_TYPES = ['VPN']

# ── CLI args ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='LLM device type/vendor identification')
parser.add_argument(
    '--type-only', action='store_true',
    help='Skip known-vendor (type+vendor) inference; only run type-only loop '
         'for Unknown-vendor IPs, then merge into all_result_type_{dev}.csv'
)
args = parser.parse_args()

# Known device type candidates (for LLM context)
ALL_DEVICE_TYPES = [
    'CAMERA', 'PRINTER', 'SCADA', 'NAS', 'ROUTER', 'NVR', "POWER_METER",
    'MEDIA_SERVER', 'VPN', 'BUILDING_AUTOMATION', 'MEDICAL', 'UNKNOWN'
]

# ── Helpers ───────────────────────────────────────────────────────────

def safe_str(val):
    if pd.isna(val) or val is None:
        return ''
    return str(val).strip()


def get_field(fp, field):
    return safe_str(fp.get(field, ''))


def build_fingerprint_text(fp):
    """Extract key fingerprint fields into concise text for LLM prompt."""
    fields = [
        ('Service Distribution',      'service-distribution'),
        ('Software Vendors',          'sw-vendors'),
        ('Software Products',         'sw-products'),
        ('Software Versions',         'sw-versions'),
        ('Software Info',             'sw-info'),
        ('Hardware Vendors',          'hw-vendors'),
        ('Hardware Products',         'hw-products'),
        ('Hardware Info',             'hw-info'),
        ('OS Vendor',                 'os-vendor'),
        ('OS Product',                'os-product'),
        ('OS Version',                'os-version'),
        ('TLS Certificate Subjects',  'cert-subjects'),
        ('TLS Certificate Issuers',   'cert-issuers'),
        ('TLS Certificate Info',      'cert-info'),
        ('TLS Versions',              'tls-versions'),
        ('HTTP Tags',                 'http-tags'),
        ('HTTP Favicon Hashes',       'http-favicon-hashes'),
        ('DNS Reverse',               'dns-reverse'),
        ('AS Name',                   'as-name'),
        ('WHOIS Organization',        'whois-organization-name'),
        ('Country',                   'loc-country'),
    ]
    lines = []
    for label, key in fields:
        val = get_field(fp, key)
        if val:
            if len(val) > 500:
                val = val[:500] + '...[truncated]'
            lines.append(f'- {label}: {val}')
    return '\n'.join(lines) if lines else '(No fingerprint data available)'


# ── Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an expert in IoT device fingerprinting, classification, and vendor identification.
Given a device fingerprint (network scan data), you must:
1. Identify the **device type** (exactly one label).
2. Identify the **vendor** (manufacturer/software brand), returning up to 3 high-confidence candidates.

**Device type candidates**: {', '.join(ALL_DEVICE_TYPES)}
- Use "UNKNOWN" for device type only if you cannot determine it with any confidence.

**Vendor identification rules**:
- Return a list of 1–3 vendors that you have high confidence in. Do NOT pad to 3 if only 1–2 are clearly supported by evidence.
- If you cannot identify any vendor with reasonable confidence, return ["unknown"].
- Each vendor entry should include a confidence score (0.0–1.0).

**Confidence thresholds**:
- High: ≥ 0.65 → include in vendor list
- Low:  < 0.65 → mark vendor as "unknown" OR omit that candidate

**Ambiguous keywords**: If the fingerprint contains technical terms, product names, or identifiers that you do not recognize or are uncertain about, list them in "ambiguous_keywords".

Respond ONLY in valid JSON with this exact structure:
{{
  "device_type": "<TYPE or UNKNOWN>",
  "type_confidence": <0.0–1.0>,
  "type_reason": "<concise explanation for device type assignment>",
  "vendor_top3": [
    {{"vendor": "<name>", "confidence": <0.0–1.0>}},
    ...
  ],
  "vendor_reason": "<concise explanation for vendor identification>",
  "ambiguous_keywords": ["<keyword1>", ...]
}}

If vendor confidence is below threshold for ALL candidates, set vendor_top3 to [{{"vendor": "unknown", "confidence": 0.0}}].
Write all text in English."""


SYSTEM_PROMPT_TYPE_ONLY = f"""You are an expert in IoT device fingerprinting and classification.
Given a device fingerprint (network scan data), identify the **device type** (exactly one label).

**Device type candidates**: {', '.join(ALL_DEVICE_TYPES)}
- Use "UNKNOWN" only if you cannot determine the type with any confidence.

**Ambiguous keywords**: If the fingerprint contains technical terms or identifiers you do not recognize, list them in "ambiguous_keywords".

Respond ONLY in valid JSON with this exact structure:
{{
  "device_type": "<TYPE or UNKNOWN>",
  "type_confidence": <0.0\u20131.0>,
  "type_reason": "<concise explanation>",
  "ambiguous_keywords": ["<keyword1>", ...]
}}
Write all text in English."""


def build_user_prompt(ip, fp_text):
    return f"""Analyze the following IoT device fingerprint and identify its type and vendor.

**Device IP**: {ip}

**Device Fingerprint**:
{fp_text}

Return your analysis as JSON."""


def build_user_prompt_type_only(ip, fp_text):
    return f"""Analyze the following IoT device fingerprint and identify its device type.

**Device IP**: {ip}

**Device Fingerprint**:
{fp_text}

Return your analysis as JSON."""


# ── Load data ─────────────────────────────────────────────────────────

print('Loading label and fingerprint data...')

label_dfs        = {}   # dev -> DataFrame(ip, vendor)
ipraw_dfs        = {}   # dev -> DataFrame(all columns, filtered to known-vendor IPs)
unknown_ipraw_dfs = {}  # dev -> DataFrame(all columns, filtered to Unknown-vendor IPs)

for dev in DEVICE_TYPES:
    label_path      = os.path.join(LABEL_DIR,      f'label_nonempty_{dev}.csv')
    full_label_path = os.path.join(FULL_LABEL_DIR, f'label_{dev}.csv')
    ipraw_path      = os.path.join(IPRAW_DIR,      f'ipraw_{dev}.csv')

    df_label = pd.read_csv(label_path)
    df_ipraw = pd.read_csv(ipraw_path, low_memory=False)

    known_ips = set(df_label['ip'].astype(str))
    df_test   = df_ipraw[df_ipraw['ip'].astype(str).isin(known_ips)].copy()

    df_full  = pd.read_csv(full_label_path)
    unk_ips  = set(df_full[df_full['vendor'].str.upper() == 'UNKNOWN']['ip'].astype(str))
    df_unk_test = df_ipraw[df_ipraw['ip'].astype(str).isin(unk_ips)].copy()

    label_dfs[dev]         = df_label
    ipraw_dfs[dev]         = df_test
    unknown_ipraw_dfs[dev] = df_unk_test

    print(f'  [{dev}] Known-vendor IPs: {len(known_ips)} | Matched in ipraw: {len(df_test)}')
    print(f'  [{dev}] Unknown-vendor IPs: {len(unk_ips)} | Matched in ipraw: {len(df_unk_test)}')


# ── Initialize LLM ────────────────────────────────────────────────────

print('\nInitializing DeepSeek LLM...')
llm_client = LLM()

# ── Main inference loop ───────────────────────────────────────────────

if args.type_only:
    print('\n[--type-only] Skipping known-vendor inference loop.')

for dev in DEVICE_TYPES if not args.type_only else []:
    print(f'\n{"=" * 60}')
    print(f'Processing [{dev}]')
    print('=' * 60)

    df_label = label_dfs[dev]
    df_test  = ipraw_dfs[dev]

    # Build ip→vendor ground truth map
    true_vendor_map = dict(zip(df_label['ip'].astype(str), df_label['vendor']))

    # Output paths
    csv_path  = os.path.join(PREDICT_DIR, f'all_result_method1_{dev}.csv')
    json_path = os.path.join(PREDICT_DIR, f'llm_result_{dev}.json')

    # Resume support: load existing JSON results
    existing = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            for entry in old_data:
                if 'device_type' in entry:
                    existing[entry['ip']] = entry
            print(f'  Loaded {len(existing)} cached results (resume mode)')
        except Exception:
            pass

    results_json = []   # full LLM output per IP
    results_csv  = []   # flattened for CSV

    total = len(df_test)
    for idx, (_, row) in enumerate(df_test.iterrows(), 1):
        ip       = str(row['ip'])
        true_v   = true_vendor_map.get(ip, 'unknown')
        true_t   = dev

        # Use cached result if available
        if ip in existing:
            entry = existing[ip]
            results_json.append(entry)
            v3 = entry.get('vendor_top3', [])
            results_csv.append({
                'ip':          ip,
                'true_type':   true_t,
                'pred_type':   entry.get('device_type', 'unknown'),
                'true_vendor': true_v,
                'vendor_top1': v3[0]['vendor'] if len(v3) > 0 else 'unknown',
                'vendor_top2': v3[1]['vendor'] if len(v3) > 1 else '',
                'vendor_top3': v3[2]['vendor'] if len(v3) > 2 else '',
                'type_confidence':  entry.get('type_confidence', 0.0),
                'vendor_top1_conf': v3[0].get('confidence', 0.0) if len(v3) > 0 else 0.0,
            })
            print(f'  [{idx}/{total}] Cached: {ip}')
            continue

        # Build fingerprint text
        fp_dict  = row.to_dict()

        fp_text  = build_fingerprint_text(fp_dict)
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': build_user_prompt(ip, fp_text)},
        ]

        # print(messages)

        # Call LLM
        try:
            response = llm_client.chat_with_llm('DEEPSEEK', messages, whether_json=True)
            print(response)

            # Normalise vendor_top3 (filter below-threshold)
            raw_v3 = response.get('vendor_top3', [])
            if isinstance(raw_v3, list):
                high_conf = [
                    v for v in raw_v3
                    if isinstance(v, dict) and float(v.get('confidence', 0)) >= 0.65
                ]
            else:
                high_conf = []
            if not high_conf:
                high_conf = [{'vendor': 'unknown', 'confidence': 0.0}]

            entry = {
                'ip':                ip,
                'true_type':         true_t,
                'true_vendor':       true_v,
                'device_type':       response.get('device_type', 'UNKNOWN'),
                'type_confidence':   response.get('type_confidence', 0.0),
                'type_reason':       response.get('type_reason', ''),
                'vendor_top3':       high_conf,
                'vendor_reason':     response.get('vendor_reason', ''),
                'ambiguous_keywords': response.get('ambiguous_keywords', []),
            }
            results_json.append(entry)
            results_csv.append({
                'ip':          ip,
                'true_type':   true_t,
                'pred_type':   entry['device_type'],
                'true_vendor': true_v,
                'vendor_top1': high_conf[0]['vendor'] if len(high_conf) > 0 else 'unknown',
                'vendor_top2': high_conf[1]['vendor'] if len(high_conf) > 1 else '',
                'vendor_top3': high_conf[2]['vendor'] if len(high_conf) > 2 else '',
                'type_confidence':  entry['type_confidence'],
                'vendor_top1_conf': high_conf[0].get('confidence', 0.0) if high_conf else 0.0,
            })
            print(f'  [{idx}/{total}] OK: {ip} | type={entry["device_type"]} '
                  f'| vendor={high_conf[0]["vendor"]}')

        except Exception as e:
            entry = {
                'ip':                ip,
                'true_type':         true_t,
                'true_vendor':       true_v,
                'device_type':       'UNKNOWN',
                'type_confidence':   0.0,
                'type_reason':       f'LLM error: {e}',
                'vendor_top3':       [{'vendor': 'unknown', 'confidence': 0.0}],
                'vendor_reason':     f'LLM error: {e}',
                'ambiguous_keywords': [],
            }
            results_json.append(entry)
            results_csv.append({
                'ip': ip, 'true_type': true_t, 'pred_type': 'UNKNOWN',
                'true_vendor': true_v, 'vendor_top1': 'unknown',
                'vendor_top2': '', 'vendor_top3': '',
                'type_confidence': 0.0, 'vendor_top1_conf': 0.0,
            })
            print(f'  [{idx}/{total}] ERROR: {ip} | {e}')

        # Periodic save every 20 entries
        if idx % 20 == 0:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            pd.DataFrame(results_csv).to_csv(csv_path, index=False)
            print(f'  >> Intermediate save ({idx}/{total})')

        time.sleep(0.3)

    # Final save
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results_csv).to_csv(csv_path, index=False)

    n_total       = len(results_csv)
    n_type_ok     = sum(1 for r in results_csv if r['pred_type'] == r['true_type'])
    n_vendor_unk  = sum(1 for r in results_csv if r['vendor_top1'] == 'unknown')
    n_vendor_ok   = sum(
        1 for r in results_csv
        if r['vendor_top1'] == r['true_vendor']
    )
    print(f'\n  [{dev}] Done. Total={n_total} | '
          f'Type Acc={n_type_ok/n_total:.2%} | '
          f'Vendor Top-1 Acc={n_vendor_ok/n_total:.2%} | '
          f'Vendor Unknown={n_vendor_unk}')
    print(f'  Saved CSV  → {csv_path}')
    print(f'  Saved JSON → {json_path}')

# ── Type-only inference for Unknown-vendor IPs ────────────────────────

for dev in DEVICE_TYPES:
    print(f'\n{"=" * 60}')
    print(f'Processing type-only [{dev}] (Unknown-vendor IPs)')
    print('=' * 60)

    df_unk_test = unknown_ipraw_dfs[dev]

    json_path_to = os.path.join(PREDICT_DIR, f'llm_result_typeonly_{dev}.json')
    csv_path_to  = os.path.join(PREDICT_DIR, f'typeonly_result_{dev}.csv')

    # Resume support
    existing_to = {}
    if os.path.exists(json_path_to):
        try:
            with open(json_path_to, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            for entry in old_data:
                if 'device_type' in entry:
                    existing_to[entry['ip']] = entry
            print(f'  Loaded {len(existing_to)} cached results (resume mode)')
        except Exception:
            pass

    results_json_to = []
    results_csv_to  = []

    total = len(df_unk_test)
    for idx, (_, row) in enumerate(df_unk_test.iterrows(), 1):
        ip     = str(row['ip'])
        true_t = dev

        if ip in existing_to:
            entry = existing_to[ip]
            results_json_to.append(entry)
            results_csv_to.append({
                'ip':               ip,
                'true_type':        true_t,
                'pred_type':        entry.get('device_type', 'UNKNOWN'),
                'true_vendor':      'Unknown',
                'vendor_top1':      '',
                'vendor_top2':      '',
                'vendor_top3':      '',
                'type_confidence':  entry.get('type_confidence', 0.0),
                'vendor_top1_conf': '',
            })
            print(f'  [{idx}/{total}] Cached: {ip}')
            continue

        fp_text  = build_fingerprint_text(row.to_dict())
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT_TYPE_ONLY},
            {'role': 'user',   'content': build_user_prompt_type_only(ip, fp_text)},
        ]

        try:
            response = llm_client.chat_with_llm('DEEPSEEK', messages, whether_json=True)
            print(response)

            entry = {
                'ip':                ip,
                'true_type':         true_t,
                'true_vendor':       'Unknown',
                'device_type':       response.get('device_type', 'UNKNOWN'),
                'type_confidence':   response.get('type_confidence', 0.0),
                'type_reason':       response.get('type_reason', ''),
                'ambiguous_keywords': response.get('ambiguous_keywords', []),
                'type_only':         True,
            }
            results_json_to.append(entry)
            results_csv_to.append({
                'ip':               ip,
                'true_type':        true_t,
                'pred_type':        entry['device_type'],
                'true_vendor':      'Unknown',
                'vendor_top1':      '',
                'vendor_top2':      '',
                'vendor_top3':      '',
                'type_confidence':  entry['type_confidence'],
                'vendor_top1_conf': '',
            })
            print(f'  [{idx}/{total}] OK: {ip} | type={entry["device_type"]}')

        except Exception as e:
            entry = {
                'ip':                ip,
                'true_type':         true_t,
                'true_vendor':       'Unknown',
                'device_type':       'UNKNOWN',
                'type_confidence':   0.0,
                'type_reason':       f'LLM error: {e}',
                'ambiguous_keywords': [],
                'type_only':         True,
            }
            results_json_to.append(entry)
            results_csv_to.append({
                'ip': ip, 'true_type': true_t, 'pred_type': 'UNKNOWN',
                'true_vendor': 'Unknown', 'vendor_top1': '',
                'vendor_top2': '', 'vendor_top3': '',
                'type_confidence': 0.0, 'vendor_top1_conf': '',
            })
            print(f'  [{idx}/{total}] ERROR: {ip} | {e}')

        if idx % 20 == 0:
            with open(json_path_to, 'w', encoding='utf-8') as f:
                json.dump(results_json_to, f, ensure_ascii=False, indent=2)
            pd.DataFrame(results_csv_to).to_csv(csv_path_to, index=False)
            print(f'  >> Intermediate save ({idx}/{total})')

        time.sleep(0.3)

    # Final save
    with open(json_path_to, 'w', encoding='utf-8') as f:
        json.dump(results_json_to, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results_csv_to).to_csv(csv_path_to, index=False)

    n_total = len(results_csv_to)
    if n_total > 0:
        n_type_ok = sum(1 for r in results_csv_to if r['pred_type'] == r['true_type'])
        print(f'\n  [{dev}] Type-only done. Total={n_total} | Type Acc={n_type_ok/n_total:.2%}')
    else:
        print(f'\n  [{dev}] No Unknown-vendor IPs to process.')
    print(f'  Saved JSON → {json_path_to}')
    print(f'  Saved CSV  → {csv_path_to}')

# ── Merge type results ─────────────────────────────────────────────────

print('\nMerging type+vendor and type-only results...')

for dev in DEVICE_TYPES:
    method1_path  = os.path.join(PREDICT_DIR, f'all_result_method1_{dev}.csv')
    typeonly_path = os.path.join(PREDICT_DIR, f'typeonly_result_{dev}.csv')
    merged_path   = os.path.join(PREDICT_DIR, f'all_result_type_{dev}.csv')

    df_method1  = pd.read_csv(method1_path)  if os.path.exists(method1_path)  else pd.DataFrame()
    df_typeonly = pd.read_csv(typeonly_path) if os.path.exists(typeonly_path) else pd.DataFrame()

    df_merged = pd.concat([df_method1, df_typeonly], ignore_index=True)
    df_merged.to_csv(merged_path, index=False)
    print(f'  [{dev}] Merged {len(df_method1)} known-vendor + {len(df_typeonly)} type-only = {len(df_merged)} total')
    print(f'  Saved → {merged_path}')

print('\nAll done!')
