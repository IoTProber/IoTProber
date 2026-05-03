#!/usr/bin/env python3
"""
Analyze vendor identification results from LLM prediction.

Metrics:
- Recognition rate (% non-unknown vendor_top1)
- Top-1 accuracy  (vendor_top1 == true_vendor)
- Top-3 accuracy  (true_vendor in {vendor_top1, vendor_top2, vendor_top3})
- Per-vendor: TP, TN, FP, FN, Precision, Recall, F1  (micro + macro)
- Macro Precision, Recall, F1

Outputs:
- Printed metrics table
- vendor_analysis.csv
- unseen_vendor_llm_top10.png / .pdf  — Top-10 vendor recognition chart
  (style matching unseen_vendor_comparison.pdf)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import re
from collections import defaultdict

matplotlib.rcParams.update({
    'font.size':        28,
    'axes.titlesize':   28,
    'axes.labelsize':   28,
    'xtick.labelsize':  28,
    'ytick.labelsize':  28,
    'legend.fontsize':  28,
    'font.family':      'DejaVu Sans',
})

PREDICT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(PREDICT_DIR)))
LABEL_DIR    = os.path.join(BASE_DIR, 'platform_data', 'csv', 'label', 'known')
DEVICE_TYPES = ['MEDIA_SERVER', 'VPN']

# ── Load results ──────────────────────────────────────────────────────

dfs = {}
for dev in DEVICE_TYPES:
    csv_path = os.path.join(PREDICT_DIR, f'all_result_method1_{dev}.csv')
    if not os.path.exists(csv_path):
        print(f'[WARN] Missing: {csv_path}')
        continue
    dfs[dev] = pd.read_csv(csv_path)

if not dfs:
    print('No result files found. Run llm_predict.py first.')
    sys.exit(1)

# ── Vendor name normalization & canonicalization ─────────────────────

def normalize_vendor(name):
    """Lowercase + strip all non-alphanumeric chars for fuzzy matching."""
    if pd.isna(name) or str(name).strip() == '':
        return 'unknown'
    s = re.sub(r'[^a-z0-9]', '', str(name).lower())
    return s if s else 'unknown'

for _dev in list(dfs.keys()):
    _d = dfs[_dev]
    _d['true_vendor_norm'] = _d['true_vendor'].apply(normalize_vendor)
    _d['pred_top1_norm']   = _d['vendor_top1'].apply(normalize_vendor)
    _d['pred_top2_norm']   = (_d['vendor_top2'].apply(normalize_vendor)
                              if 'vendor_top2' in _d.columns
                              else pd.Series(['unknown'] * len(_d), index=_d.index))
    _d['pred_top3_norm']   = (_d['vendor_top3'].apply(normalize_vendor)
                              if 'vendor_top3' in _d.columns
                              else pd.Series(['unknown'] * len(_d), index=_d.index))

df_all = pd.concat(dfs.values(), ignore_index=True)

# Canonical name: for each norm-group, pick the most frequent original true_vendor
canon_map = (
    df_all.groupby('true_vendor_norm')['true_vendor']
    .agg(lambda s: s.value_counts().index[0])
)
for _dev in list(dfs.keys()):
    dfs[_dev]['true_vendor_canonical'] = dfs[_dev]['true_vendor_norm'].map(canon_map)

# ── Per-device metrics ────────────────────────────────────────────────

SEP  = '─' * 90
SEP2 = '═' * 90

print(f'\n{SEP2}')
print('VENDOR IDENTIFICATION ANALYSIS')
print(SEP2)


def compute_vendor_metrics(df):
    """Return dict of per-device metrics (case-insensitive, fuzzy-matched vendor names)."""
    n = len(df)
    n_recog   = (df['pred_top1_norm'] != 'unknown').sum()

    # Top-1 / Top-3 match using normalized columns (vectorized)
    top1_mask = df['true_vendor_norm'] == df['pred_top1_norm']
    top3_mask = (
        top1_mask |
        (df['true_vendor_norm'] == df['pred_top2_norm']) |
        (df['true_vendor_norm'] == df['pred_top3_norm'])
    )
    n_top1_ok = int(top1_mask.sum())
    n_top3_ok = int(top3_mask.sum())

    recog_rate = n_recog / n if n > 0 else 0.0
    top1_acc   = n_top1_ok / n if n > 0 else 0.0
    top3_acc   = n_top3_ok / n if n > 0 else 0.0

    # Per-vendor metrics (Top-1 only, keyed by canonical name)
    vendors = sorted(df['true_vendor_canonical'].unique())
    per_v   = {}
    for v in vendors:
        v_norm     = normalize_vendor(v)
        mask_true  = df['true_vendor_norm'] == v_norm
        tp = int((mask_true  & (df['pred_top1_norm'] == v_norm)).sum())
        fn = int((mask_true  & (df['pred_top1_norm'] != v_norm)).sum())
        fp = int((~mask_true & (df['pred_top1_norm'] == v_norm)).sum())
        tn = int((~mask_true & (df['pred_top1_norm'] != v_norm)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_v[v] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}

    macro_p  = np.mean([m['precision'] for m in per_v.values()]) if per_v else 0.0
    macro_r  = np.mean([m['recall']    for m in per_v.values()]) if per_v else 0.0
    macro_f1 = np.mean([m['f1']        for m in per_v.values()]) if per_v else 0.0

    return {
        'n_total':      n,
        'n_recognized': int(n_recog),
        'recog_rate':   recog_rate,
        'n_top1_ok':    n_top1_ok,
        'top1_acc':     top1_acc,
        'n_top3_ok':    n_top3_ok,
        'top3_acc':     top3_acc,
        'macro_p':      round(macro_p, 4),
        'macro_r':      round(macro_r, 4),
        'macro_f1':     round(macro_f1, 4),
        'per_vendor':   per_v,
    }


all_metrics = {}
for dev in DEVICE_TYPES:
    if dev not in dfs:
        continue
    m = compute_vendor_metrics(dfs[dev])
    all_metrics[dev] = m

    print(f'\n[{dev}]')
    print(f'  Total IPs      : {m["n_total"]}')
    print(f'  Recognition    : {m["recog_rate"]:.4f}  ({m["n_recognized"]}/{m["n_total"]})')
    print(f'  Top-1 Accuracy : {m["top1_acc"]:.4f}  ({m["n_top1_ok"]}/{m["n_total"]})')
    print(f'  Top-3 Accuracy : {m["top3_acc"]:.4f}  ({m["n_top3_ok"]}/{m["n_total"]})')
    print(f'  Macro Precision: {m["macro_p"]:.4f}')
    print(f'  Macro Recall   : {m["macro_r"]:.4f}')
    print(f'  Macro F1       : {m["macro_f1"]:.4f}')
    print(f'\n  Per-Vendor Metrics (Top-1):')
    print(f'  {"Vendor":<22} {"Support":>8} {"TP":>6} {"TN":>6} {"FP":>6} {"FN":>6} '
          f'{"Prec":>8} {"Recall":>8} {"F1":>8}')
    print(f'  {SEP}')
    for v, vm in sorted(m['per_vendor'].items()):
        sup = (dfs[dev]['true_vendor_canonical'] == v).sum()
        print(f'  {v:<22} {sup:>8} {vm["TP"]:>6} {vm["TN"]:>6} {vm["FP"]:>6} {vm["FN"]:>6} '
              f'{vm["precision"]:>8.4f} {vm["recall"]:>8.4f} {vm["f1"]:>8.4f}')

# ── Overall summary ───────────────────────────────────────────

_n_tot   = sum(m['n_total']    for m in all_metrics.values())
_n_top1  = sum(m['n_top1_ok']  for m in all_metrics.values())
_n_top3  = sum(m['n_top3_ok']  for m in all_metrics.values())
_n_recog = sum(m['n_recognized'] for m in all_metrics.values())
_all_f1s = [vm['f1'] for m in all_metrics.values() for vm in m['per_vendor'].values()]
_all_p   = [vm['precision'] for m in all_metrics.values() for vm in m['per_vendor'].values()]
_all_r   = [vm['recall']    for m in all_metrics.values() for vm in m['per_vendor'].values()]

print(f'\n{"Per-Device-Type Summary":}')
print(f'  {SEP}')
print(f'  {"Device Type":<16} {"Support":>8} {"Top-1 Acc":>10} {"Top-3 Acc":>10} {"Macro Prec":>11} {"Macro Recall":>13} {"Macro F1":>10}')
print(f'  {SEP}')
for _dev, _m in all_metrics.items():
    print(f'  {_dev:<16} {_m["n_total"]:>8} {_m["top1_acc"]:>10.4f} {_m["top3_acc"]:>10.4f} '
          f'{_m["macro_p"]:>11.4f} {_m["macro_r"]:>13.4f} {_m["macro_f1"]:>10.4f}')
print(f'  {SEP}')

print(f'\n[OVERALL  (all device types combined)]')
print(f'  Total IPs           : {_n_tot}')
print(f'  Recognition Rate    : {_n_recog/_n_tot:.4f}  ({_n_recog}/{_n_tot})')
print(f'  Top-1 Accuracy      : {_n_top1/_n_tot:.4f}  ({_n_top1}/{_n_tot})')
print(f'  Top-3 Accuracy      : {_n_top3/_n_tot:.4f}  ({_n_top3}/{_n_tot})')
print(f'  Overall Macro Prec  : {np.mean(_all_p):.4f}')
print(f'  Overall Macro Recall: {np.mean(_all_r):.4f}')
print(f'  Overall Macro F1    : {np.mean(_all_f1s):.4f}')

print(f'\n{SEP2}')

# ── Save vendor_analysis.csv ──────────────────────────────────────────

rows = []
for dev in DEVICE_TYPES:
    if dev not in all_metrics:
        continue
    m  = all_metrics[dev]
    df = dfs[dev]
    for v, vm in m['per_vendor'].items():
        sup = (df['true_vendor_canonical'] == v).sum()
        rows.append({
            'device_type':  dev,
            'vendor':       v,
            'support':      sup,
            **vm,
        })

pd.DataFrame(rows).to_csv(
    os.path.join(PREDICT_DIR, 'vendor_analysis.csv'), index=False)
print(f'\nSaved → {os.path.join(PREDICT_DIR, "vendor_analysis.csv")}')

# ── Top-10 vendor chart (matching unseen_vendor_comparison.pdf style) ─

PALETTE = [
    '#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66',
    '#77BEDB', '#F7A935', '#A4D4AE', '#E8908D', '#9E9AC8', '#D9D9D9',
]
HATCHES     = ['///', '\\\\\\', '+++', 'xxx', '---',
               '...', '////', '\\\\\\\\', 'xxxx', '++++', '']
TOP_N       = 10
RANK_LABELS = [f'Top-{i+1}' for i in range(TOP_N)] + ['Others']
DEV_LABELS  = {'MEDIA_SERVER': 'MEDIA SERVER', 'VPN': 'VPN'}


def build_bar_data(dev, df, n_top=TOP_N):
    """
    Returns (ident_rate, top3_acc_pct, top_n_list, others_pct)
    top_n_list: [(vendor, pct_of_total), ...] for top-3-correctly-identified vendors
    """
    n          = len(df)
    ident      = (df['pred_top1_norm'] != 'unknown').sum()
    top3_mask  = (
        (df['true_vendor_norm'] == df['pred_top1_norm']) |
        (df['true_vendor_norm'] == df['pred_top2_norm']) |
        (df['true_vendor_norm'] == df['pred_top3_norm'])
    )
    correct = top3_mask.sum()

    ident_rate  = ident   / n * 100
    correct_pct = correct / n * 100

    # Among top-3-correctly identified — canonical vendor distribution
    correct_vendors = df.loc[top3_mask, 'true_vendor_canonical']
    vc         = correct_vendors.value_counts()
    top        = vc.head(n_top)
    top_list   = [(v, c / n * 100) for v, c in top.items()]
    others_pct = max(0.0, correct_pct - sum(p for _, p in top_list))

    return ident_rate, correct_pct, top_list, others_pct


bar_data = {}
for dev in DEVICE_TYPES:
    if dev not in dfs:
        continue
    bar_data[dev] = build_bar_data(dev, dfs[dev])

active_devs = [d for d in DEVICE_TYPES if d in bar_data]

fig, ax = plt.subplots(figsize=(24, 16))
n_bars = len(active_devs)
bar_w  = 0.55
x      = np.array([i * 1.8 for i in range(n_bars)])
bottoms = np.zeros(n_bars)
side_q  = []

for rank in range(TOP_N + 1):
    vals = []
    for bi, dev in enumerate(active_devs):
        ident_rate, correct_pct, top_list, others_pct = bar_data[dev]
        if rank < TOP_N:
            v = top_list[rank][1] if rank < len(top_list) else 0.0
        else:
            v = others_pct
        vals.append(v)
    vals = np.array(vals)

    ax.bar(x, vals, bar_w, bottom=bottoms,
           color=PALETTE[rank], hatch=HATCHES[rank],
           edgecolor='white', linewidth=0.5, label=RANK_LABELS[rank], zorder=3)
    ax.bar(x, vals, bar_w, bottom=bottoms,
           fill=False, edgecolor='#555', linewidth=0.65, zorder=4)

    bbox_in = dict(boxstyle='round,pad=0.10', facecolor='white',
                   edgecolor='none', alpha=0.75)
    for bi, (dev, val) in enumerate(zip(active_devs, vals)):
        if val < 0.3:
            continue
        _, _, top_list, _ = bar_data[dev]
        name = top_list[rank][0] if rank < len(top_list) else 'Others'
        if rank == TOP_N:
            name = 'Others'
        y_mid = bottoms[bi] + val / 2
        if val >= 12:
            ax.text(x[bi], y_mid, f'{name}\n{val:.2f}%',
                    ha='center', va='center', fontsize=28,
                    fontweight='bold', zorder=5, bbox=bbox_in)
        elif val >= 4:
            ax.text(x[bi], y_mid, f'{name}  {val:.2f}%',
                    ha='center', va='center', fontsize=28,
                    fontweight='bold', zorder=5, bbox=bbox_in)
        else:
            side_q.append((bi, rank, name, val, y_mid))

    bottoms += vals

# Side annotations for tiny segments (< 4%)
from collections import defaultdict as _ddict
bar_groups = _ddict(list)
for entry in side_q:
    bar_groups[entry[0]].append(entry)

MIN_GAP  = 6.0
SIDE_OFF = 0.20

for bi, entries in bar_groups.items():
    entries.sort(key=lambda e: e[4])
    x_tip    = x[bi] + bar_w / 2
    x_text   = x_tip + SIDE_OFF
    x_anchor = x_tip + 0.01

    prev_y  = -999
    label_ys = []
    for _, _, _, _, y_mid in entries:
        y = max(y_mid, prev_y + MIN_GAP)
        label_ys.append(y)
        prev_y = y

    y_max_frame = 101.0
    y_min_frame = 3.0
    if label_ys and label_ys[-1] > y_max_frame:
        shift    = label_ys[-1] - y_max_frame
        label_ys = [y - shift for y in label_ys]
    if label_ys and label_ys[0] < y_min_frame:
        shift    = y_min_frame - label_ys[0]
        label_ys = [y + shift for y in label_ys]

    bbox_s = dict(boxstyle='round,pad=0.10', facecolor='white',
                  edgecolor='#bbb', alpha=0.93, linewidth=0.5)
    for j, (_, rank, name, val, y_mid) in enumerate(entries):
        ax.annotate(
            f'{name}  {val:.2f}%',
            xy=(x_anchor, y_mid),
            xytext=(x_text, label_ys[j]),
            fontsize=28, fontweight='bold',
            ha='left', va='center',
            bbox=bbox_s,
            arrowprops=dict(arrowstyle='-', linestyle='dashed',
                            linewidth=1.2, color='#999',
                            shrinkA=0, shrinkB=2),
            zorder=7, clip_on=False,
        )

# Ident / Acc labels above bars
for bi, dev in enumerate(active_devs):
    ident_rate, correct_pct, _, _ = bar_data[dev]
    ax.text(x[bi], 102, f'Ident: {ident_rate:.2f}%',
            ha='center', va='bottom', fontsize=28,
            fontweight='bold', color='crimson', zorder=5)
    ax.text(x[bi], 107, f'Top3Acc: {correct_pct:.2f}%',
            ha='center', va='bottom', fontsize=28,
            fontweight='bold', color='navy', zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([DEV_LABELS.get(d, d) for d in active_devs], fontsize=28)
ax.set_ylabel('Vendor Proportion (% of total test IPs)')
ax.set_xlabel('Device Type (Baseline)')
ax.set_xlim(x[0] - 0.45, x[-1] + 1.55)
ax.set_ylim(0, 118)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v)}' if v <= 100 else ''))

handles = [
    mpatches.Patch(facecolor=PALETTE[i], hatch=HATCHES[i],
                   edgecolor='#555', label=RANK_LABELS[i])
    for i in range(TOP_N + 1)
]
ax.legend(handles=handles, loc='upper center', ncol=6,
          bbox_to_anchor=(0.50, -0.10),
          frameon=True, edgecolor='#999', fancybox=False, fontsize=28)

plt.tight_layout()

png_path = os.path.join(PREDICT_DIR, 'unseen_vendor_llm_top10.png')
pdf_path = os.path.join(PREDICT_DIR, 'unseen_vendor_llm_top10.pdf')
plt.savefig(png_path, dpi=200, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f'Saved chart → {png_path}')
print(f'Saved chart → {pdf_path}')
plt.show()
