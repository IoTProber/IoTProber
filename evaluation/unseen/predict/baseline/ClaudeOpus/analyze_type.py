#!/usr/bin/env python3
"""
Analyze device TYPE identification results from LLM prediction.

Metrics computed:
- Recognition rate (% non-UNKNOWN predictions)
- Per-class: TP, TN, FP, FN, Precision, Recall, F1
- Macro-average Precision, Recall, F1
- Misclassification counts (what wrong labels were assigned)

Output: printed table + type_analysis.csv saved next to this script.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

PREDICT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_TYPES = ['MEDIA_SERVER', 'VPN']

# ── Load results ──────────────────────────────────────────────────────

dfs = []
for dev in DEVICE_TYPES:
    csv_path = os.path.join(PREDICT_DIR, f'all_result_type_{dev}.csv')
    if not os.path.exists(csv_path):
        print(f'[WARN] Missing: {csv_path}')
        continue
    df = pd.read_csv(csv_path)
    dfs.append(df)

if not dfs:
    print('No result files found. Run llm_predict.py first.')
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
n_total = len(df_all)

print(f'Total samples: {n_total}')

# ── Recognition rate ──────────────────────────────────────────────────

df_all['pred_type_norm'] = df_all['pred_type'].str.upper().str.strip()
n_unknown_pred = (df_all['pred_type_norm'].isin(['UNKNOWN', ''])).sum()
n_recognized   = n_total - n_unknown_pred
recognition_rate = n_recognized / n_total if n_total > 0 else 0.0

print(f'\nRecognition Rate: {recognition_rate:.4f} ({n_recognized}/{n_total} non-UNKNOWN)')

# ── Per-class metrics ─────────────────────────────────────────────────

all_classes = sorted(df_all['true_type'].unique().tolist())
# Also include any classes predicted but not in true set
pred_classes = df_all['pred_type_norm'].unique().tolist()
all_known    = set(all_classes)

metrics = {}
for cls in all_classes:
    df_cls    = df_all[df_all['true_type'] == cls]
    true_pos  = (df_cls['pred_type_norm'] == cls).sum()
    false_neg = (df_cls['pred_type_norm'] != cls).sum()
    false_pos = 0   # not applicable: no cross-class negatives in per-type evaluation
    true_neg  = 0   # not applicable: same reason

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall    = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    n_cls          = (df_all['true_type'] == cls).sum()
    n_correct      = true_pos
    n_misclassified = false_neg

    accuracy = true_pos / n_cls if n_cls > 0 else 0.0

    metrics[cls] = {
        'support':   int(n_cls),
        'TP':        int(true_pos),
        'TN':        int(true_neg),
        'FP':        int(false_pos),
        'FN':        int(false_neg),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'accuracy':  round(accuracy, 4),
        'correct':   int(n_correct),
        'misclassified': int(n_misclassified),
    }

# Macro averages
macro_p   = np.mean([m['precision'] for m in metrics.values()])
macro_r   = np.mean([m['recall']    for m in metrics.values()])
macro_f1  = np.mean([m['f1']        for m in metrics.values()])
macro_acc = np.mean([m['accuracy']  for m in metrics.values()])

# Overall accuracy (micro)
n_correct_total  = sum(m['correct'] for m in metrics.values())
overall_accuracy = n_correct_total / n_total if n_total > 0 else 0.0

# ── Misclassification breakdown ───────────────────────────────────────

misclass = defaultdict(lambda: defaultdict(int))
for _, row in df_all.iterrows():
    true_t = row['true_type']
    pred_t = row['pred_type_norm']
    if pred_t != true_t:
        misclass[true_t][pred_t] += 1

# ── Print table ───────────────────────────────────────────────────────

SEP  = '─' * 100
SEP2 = '═' * 100

print(f'\n{SEP2}')
print('DEVICE TYPE IDENTIFICATION ANALYSIS')
print(SEP2)

print(f'\n{"Recognition Rate":30s}: {recognition_rate:.4f}  ({n_recognized}/{n_total} non-UNKNOWN predictions)')
print(f'{"Overall Accuracy":30s}: {overall_accuracy:.4f}  ({n_correct_total}/{n_total} correctly classified)')
print(f'{"Macro Accuracy":30s}: {macro_acc:.4f}')

print(f'\n{"Per-Class Metrics":}')
print(SEP)
header = (f"{'Class':<20} {'Support':>8} {'Correct':>8} {'Misclf':>8} "
          f"{'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
print(header)
print(SEP)
for cls in all_classes:
    m = metrics[cls]
    print(f"{cls:<20} {m['support']:>8} {m['correct']:>8} {m['misclassified']:>8} "
          f"{m['TP']:>6} {m['TN']:>6} {m['FP']:>6} {m['FN']:>6} "
          f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>10.4f}")
print(SEP)
print(f"{'Macro Average':<20} {n_total:>8} "
      f"{'':>8} {'':>8} {'':>6} {'':>6} {'':>6} {'':>6} "
      f"{macro_p:>10.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {macro_acc:>10.4f}")
print(f"{'Overall Accuracy':<20} {n_total:>8} {n_correct_total:>8} {'':>8} {'':>6} {'':>6} {'':>6} {'':>6} "
      f"{'':>10} {'':>8} {'':>8} {overall_accuracy:>10.4f}")
print(SEP)

print(f'\n{"Misclassification Breakdown":}')
print(SEP)
_hdr = 'True \\ Predicted'
print(f"{_hdr:<22}", end='')
# Build all predicted labels seen in errors
err_pred_labels = sorted({
    pred for true_d in misclass for pred in misclass[true_d]
})
for lbl in err_pred_labels:
    print(f"{lbl:>18}", end='')
print()
print(SEP)
for cls in all_classes:
    print(f"{cls:<22}", end='')
    for lbl in err_pred_labels:
        cnt = misclass[cls].get(lbl, 0)
        print(f"{cnt:>18}", end='')
    print()
print(SEP)

print(f'\n{SEP2}')

# ── Save CSV ──────────────────────────────────────────────────────────

rows = []
for cls in all_classes:
    m = metrics[cls]
    row = {'class': cls, **m}
    # misclass breakdown columns
    for lbl in err_pred_labels:
        row[f'misclf_as_{lbl}'] = misclass[cls].get(lbl, 0)
    rows.append(row)

# Summary row
summary_row = {
    'class':      'MACRO_AVG',
    'support':    n_total,
    'TP': '', 'TN': '', 'FP': '', 'FN': '',
    'correct':    sum(m['correct'] for m in metrics.values()),
    'misclassified': sum(m['misclassified'] for m in metrics.values()),
    'precision':  round(macro_p,   4),
    'recall':     round(macro_r,   4),
    'f1':         round(macro_f1,  4),
    'accuracy':   round(overall_accuracy, 4),
}
rows.append(summary_row)

df_out = pd.DataFrame(rows)
out_path = os.path.join(PREDICT_DIR, 'type_analysis.csv')
df_out.to_csv(out_path, index=False)
print(f'\nSaved → {out_path}')
