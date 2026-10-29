# SVM_validation — Drift Evaluation Ground Truth & Correctness Criteria

This directory holds the drift-detection evaluation referenced by the paper's
drift-correctness table. A reviewer noted that some IPs listed in
`IoTProber/incorrect_ips.csv` also appear in `Ground/drift_ips.csv`, which
looks contradictory. It is not — membership and correctness answer two
different questions at two different stages. This file documents the exact
semantics.

## Directory layout

| Path | What it is |
|---|---|
| `Ground/class_drift_scores.csv` | **Class-level ground truth.** One row per (device_type, vendor) class: `drift_score`, `n_old`, `n_new`, `drifted` (True/False). 182 classes total; 65 drifted=True, 117 drifted=False. |
| `Ground/drift_ips.csv` | **IP-level expansion of the drifted classes**: all 1,357 validation IPs whose (device_type, vendor) class has `drifted=True`. Verified: its 65 distinct classes are exactly the 65 `drifted=True` classes; no `drifted=False` class contributes an IP. |
| `IoTProber/detected_ips.csv` | The 814 IPs flagged as drifted by IoTProber's drift detector in this evaluation. |
| `IoTProber/correct_ips.csv` | 773 of the 814 flagged IPs — see correctness criterion below. |
| `IoTProber/incorrect_ips.csv` | The other 41 flagged IPs. |
| `performance/IoTProber/predict/*_val_pred.csv` | A *separate* classification benchmark population (7,239 IPs, `ip,vendor,label,label_pred,correct`). **Disjoint** from the 814 drift IPs — do not join against it. |

## The two-stage semantics

**Stage 1 — drift detection (membership).** The detection evaluation runs over
the population of IPs from `drifted=True` classes (`drift_ips.csv`). Therefore:

- Every flagged IP belongs to a ground-truth drifted class by construction:
  **detection precision is 100% by design**, not an empirical result.
- Detection **recall = 814 / 1,357 = 60.0%**; the 543 unflagged ground-truth
  IPs are the misses.

**Stage 2 — post-drift re-identification (correctness).** For each *flagged*
IP, the pipeline re-identifies the device; the IP is

- `correct` — the post-drift re-identification matches the ground-truth
  (type, vendor) label: **773 / 814 = 95.0%** (the paper's reported
  percentage);
- `incorrect` — the re-identification mismatches (41 IPs, concentrated in 7
  classes; PRINTER×25, BUILDING_AUTOMATION×8, CAMERA×3, NVR×3, MEDICAL×1,
  ROUTER×1).

Consequently, an IP can be in `drift_ips.csv` **and** in `incorrect_ips.csv`
simultaneously without contradiction: the first says "it truly drifted
(Stage-1 membership)", the second says "the detector flagged it correctly
but its post-drift re-identification was wrong (Stage-2)".

## Column semantics of correct/incorrect/detected CSVs

All four columns — `ip, vendor, device_type, class_drift_score` — are
**ground-truth attributes** of the listed IPs (joinable to
`Ground/drift_ips.csv`; values verified identical). They are *not* detector
outputs. The correct/incorrect split itself is the only judgment column
(implied by the file membership).

## Known artifact gap (honest note)

The per-IP *predicted* labels produced during Stage 2 were not retained in
this artifact, and the script that generated the correct/incorrect split was
not committed. The split reproduces the paper's reported percentage
(773/814), and its criterion is as documented above, but reviewers wishing to
audit Stage 2 per-IP need the predictions. Regenerating `correct_ips.csv` /
`incorrect_ips.csv` with an explicit `label_pred` column (and committing the
generator) is the planned follow-up.

## Quick reproduction

```python
import pandas as pd
det = pd.read_csv("IoTProber/detected_ips.csv")
cor = pd.read_csv("IoTProber/correct_ips.csv")
inc = pd.read_csv("IoTProber/incorrect_ips.csv")
gt  = pd.read_csv("Ground/drift_ips.csv")
assert len(cor) + len(inc) == len(det) == 814
assert set(det["ip"]) <= set(gt["ip"])          # membership: all flagged are truth
print("recall      =", len(det) / len(gt))       # 0.600
print("correctness =", len(cor) / len(det))      # 0.950
```
