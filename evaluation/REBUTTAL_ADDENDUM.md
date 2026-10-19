# Rebuttal Addendum — Artifact Corrections and Completions

Date: 2026-10-19 · Applies to the CCS '26 submission of *IoTProber*.

This addendum answers four reviewer comments about the released artifacts:
(1) invalidated full-pipeline records, missing graph-workflow scripts and
adapter/data paths; (2) the label-matching strategy behind Table 10
(15.91 vs 1.42 percentage points); (3) the missing Alarm/Controller main
predictions and Table 8 baseline outputs; (4) a small runnable package with
a permanent archival identifier.

Everything below is reproducible from the pinned revision listed in §5.

---

## 1. Runnable package, missing scripts, unified paths

### 1.1 What was missing and is now published

Commit `004c3f2` ("Remove evaluation directory") had deleted the tracked
evaluation tooling, and several scripts referenced by the documentation were
never committed. The following are now in the repository **and** the
HuggingFace dataset (prefix `evaluation/`):

| File | Role |
|---|---|
| `evaluation/drift/CADE/svm.py` | **original** SVM trainer/scorer for Table 10, restored verbatim from the initial commit `25f6a47` (905 lines) — the source of both matching rules in §2 |
| `evaluation/drift/score_table10.py` | canonical, versioned Table-10 scorer (v1.1.0) with every matching rule explicit |
| `evaluation/unseen/score_table8.py` | canonical, versioned Table-8 scorer (v1.1.0) with both vendor-matching conventions |
| `evaluation/unseen/split_and_evaluate.py` | the loader/metric functions the Table-8 scorer imports (single source of truth) |
| `evaluation/unseen/recompute_vendor_f1.py` | referenced by `VERIFICATION_v2.md` (also published) |
| `evaluation/run_bench_100.py`, `evaluation/bench_100/{run_bench_v2,score_v2,finalize_report,dump_evidence,prewarm_local}.py` | bench harness referenced by `bench_100/IMPROVEMENT_ANALYSIS.md` (also published) |
| `evaluation/bench_100/IMPROVEMENT_ANALYSIS.md`, `evaluation/drift/PACA_V2_REPORT.md`, `evaluation/unseen/VERIFICATION_v2.md` | the reports that reference the scripts above |
| `evaluation/predict/{TYPE}_{type,vendor}_prediction.json` (22 files, all 11 types) | the decision-agent main predictions, incl. **ALARM/CONTROLLER** |
| `evaluation/validation/type/predict/IoTProber/type_{ALARM,CONTROLLER}.csv`, `.../vendor/predict/IoTProber/vendor_{ALARM,CONTROLLER}.csv` | completes the 11-type prediction sets in the validation tree |
| `evaluation/drift/CADE/SVM_validation/performance/{IoTProber,CADE}/val_metrics.json`, `val_classification_report.txt`, `{IoTProber,CADE}/detection_reasons.json`, `Ground/drift_reasons.json` | provenance of every Table-10 number (restored from `25f6a47`) |

The graph workflow itself (`graph/cluster.py`, `construction.py`, `build.py`,
`build_neo4j.py`, `update_graph.py`, `vector.py`, `HGT.py`, `api.py`,
`regen_overall_summaries.py`, `repair_overall_community.py`) was already
tracked and unchanged; the gap was confined to `evaluation/`.

### 1.2 Full-pipeline (fullflow) records and validity gating

Only records with retrieval schema `2026-09-25-v3` are valid measurements:
earlier runs could retrieve the query IP itself, exposed the true-type
filename to the decision tool, read scores greater than one as cosine
similarities, and treated missing community reports as a synthetic 0.5
match. Validity is enforced **in code, not by convention** —
`demo/evaluate_fullflow.py` refuses any manifest or record row whose schema
version differs (`SCHEMA_VERSION`, lines 141/153), and
`demo/run_fullflow_demo.py` resumes only records stamped with the current
`PIPELINE_VERSION` (lines 42/84), so legacy rows can neither enter a metric
nor silently mix into a new run. Every record and manifest is stamped with
its `schema_version` at write time.

The **valid reference run** is published on the dataset mirror at
`evaluation/demo/fullflow/`: `manifest.json` (`schema_version
2026-09-25-v3`, CAMERA completed) and `evaluation_report.json` (completion
rate 1.0, zero self-match and score-range violations).

Regeneration (from `demo/fullflow/README.md`, verbatim):

```bash
EL=/path/to/python            # env with requirements.txt installed
$EL demo/select_demo_data.py --materialize-validation   # fixed-seed random cases
CUDA_VISIBLE_DEVICES=0 $EL demo/run_fullflow_demo.py     # all 11 types, resumable
CUDA_VISIBLE_DEVICES=0 $EL demo/run_fullflow_demo.py --showcase-only --types CAMERA
$EL demo/evaluate_fullflow.py                            # accuracy + integrity checks
```

The evaluator reports completion rate, accuracy over completed cases and
end-to-end accuracy over every selected case, so missing evidence can never
disappear from a denominator.

### 1.3 One canonical path layout

All path knobs with their overrides (see `path_config.py`, `agent/agent.py`):

| Resource | Default | Override |
|---|---|---|
| Fingerprint data root (`rag/`, `all/`, `label/`) | `platform_data/csv` | `export IOTPROBER_CSV_DATA_DIR=$PWD/platform_data/csv` |
| Local-KB alias the retriever expects | `csv/local/1` → `csv/rag` | `mkdir -p platform_data/csv/local && ln -sfn ../rag platform_data/csv/local/1` |
| Vector index | `platform_data/csv/rag/vectorDB` | `IOTPROBER_VECTOR_DB_DIR` |
| Unseen LoRA adapter | `evaluation/unseen/llama3/results_v2/final_model` (`agent/agent.py:60`) | `--unseen_adapter` / ctor arg |
| Drift artifacts | `drift_data/autoencoder_drift_v2` (`path_config.DRIFT_OUTPUT_DIR`) | `--drift_dir` / `IOTPROBER_DRIFT_MODEL_DIR` |
| Run outputs | repo dirs | `IOTPROBER_VALIDATION_DIR`, `IOTPROBER_QUERY_DB_DIR`, `IOTPROBER_PREDICTION_DIR` |
| LLM channels | `config/llm_config.local.json` (copy of template; no key is hard-coded) | per-channel sections |
| Base models (not redistributable) | `./Meta-Llama-3.1-8B-Instruct`, `./qwen3_embedding_06b` | download from the official cards (guide §1.2) |

Two portability traps and their fixes:

```bash
# HuggingFace stores the entity-graph export with a space; HGT.py expects entity_graph/
ln -sfn "entity graph" entity_graph
# csv/local is not redistributed; it is an alias of csv/rag (see table above)
```

---

## 2. Table 10 — the exact label-matching strategy (and a correction)

**Both rules are now public code.** `evaluation/drift/CADE/svm.py` (restored
from initial commit `25f6a47`) is the single script that produced the paper's
Stage-2 and Stage-3 rows:

- **Stage 2** was scored with **byte equality**
  (`svm.py::run_test`: `correct = (label == label_pred)`) →
  `0.4937 / 0.7403 / 0.6892` (macro-F1 / weighted-F1 / accuracy), reproduced
  exactly by `score_table10.py --match exact`.
- **Stage 3** was scored with `svm.py::normalize_label`: split the label on
  the **first** `-`, lowercase the device-type part as-is, strip every
  non-`[a-z0-9]` character from the **vendor part only**, then apply the
  vendor alias `{"electroind": "electroindustries"}`. This rule agrees with
  the shipped `correct` column of the Stage-3 CSVs on **7,239/7,239** rows
  for both methods and reproduces the paper's rows **bit-exactly**:
  IoTProber `6141/7239 = 0.8483`, `0.5509 / 0.8711`; CADE `6082/7239 =
  0.8402`, `0.5430 / 0.8649` (see the restored `val_metrics.json` files).

The reviewer's arithmetic is confirmed. Applying one consistent rule to the
same 7,239 records gives:

| Rule (both stages) | Stage 2 acc | Stage 3 IoTProber | Δ | Stage 3 CADE | Δ |
|---|---|---|---|---|---|
| exact (byte equality) | 0.6892 | 0.6709 | **−1.83 pp** | 0.6654 | −2.38 pp |
| original (`svm.py`) | 0.8341 | 0.8483 | **+1.42 pp** | 0.8402 | +0.61 pp |
| full normalization (case+punct) | 0.8335 | 0.8478 | +1.43 pp | 0.8396 | +0.61 pp |

The published **+15.91 pp** headline compared Stage-2-exact with
Stage-3-fuzzy; it is a rule-mixing artifact, not a retraining gain, and the
same applies to the +5.72 pp macro-F1 claim (consistent-rule macro-F1 delta:
0.5180 → 0.5509 = +3.29 pp). **Correction to be made in the revision:**
report both stages under the `original` rule — accuracy +1.42 pp (IoTProber)
/ +0.61 pp (CADE) — and keep the rule-independent drift-detection results,
which stand: among detected samples, drift-detection correctness
**94.96 % (773/814, IoTProber)** vs **68.97 % (767/1,112, CADE)**
(`detected_ips.csv` / `correct_ips.csv`, confirmed by row counts; all 814
IoTProber-detected IPs are ground-truth drift IPs with matching vendor and
device type).

**Scoring code and version** (this is the deliverable requested):

```bash
hf download IoTProber/raw_dataset --repo-type dataset \
   --include "evaluation/drift/*" --revision f4dbbd53ceb2f3ad44eba79e81744dc78b1866cc --local-dir .
python evaluation/drift/score_table10.py --json table10_record.json   # v1.1.0
python evaluation/drift/score_table10.py --match original             # the paper's Stage-3 rule
```

Why exact matching is misleading here: the SVM label vocabulary itself
contains four spellings of one vendor (`NVR-Merit Lilin / LILIN / LiLin /
Li-Lin`, 1,265 of 7,239 rows), which byte equality scores as errors in both
stages.

---

## 3. Alarm/Controller main predictions and Table 8 baseline outputs

### 3.1 What is now published

- `evaluation/predict/{ALARM,CONTROLLER}_{type,vendor}_prediction.json` and
  the four `type_/vendor_{ALARM,CONTROLLER}.csv` files (see §1.1) — the
  IoTProber decision predictions for the two 2026-07-21 types, with full
  provenance per IP (both decision branches, retrieval evidence, unseen/drift
  first-stage, `pipeline_version`).
- Table 8 baseline outputs were already on the dataset and are complete:
  `evaluation/unseen/predict/baseline/Mahmood/predict_{dev}.json` (B1),
  `.../ClaudeOpus/{all_result_type,llm_result,type_analysis,...}_{dev}.*`
  (B2), `.../llama3raw/*` — plus per-IP `type_predictions_{method}_{dev}.csv`
  and `vendor_predictions_{method}_{dev}.csv` for all four methods.

### 3.2 Coverage disclosure (what the published Alarm/Controller files are)

The paper's test sets are `evaluation/validation/45_features/test_ALARM_1.csv`
(177 IPs) and `test_CONTROLLER_1.csv` (117 IPs). The baselines cover them
fully and reproduce the paper's B1 numbers exactly
(`evaluation/validation/baseline/Mahmood/metrics.json`: Alarm type
61/177 = 34.46 %, vendor 130/177 = 73.45 %; Controller 93/117 = 79.49 %,
108/117 = 92.31 %). The published IoTProber predictions, however, are:

- **ALARM**: the fixed-seed 40-IP demo/validation pool (0-overlap with the
  177-IP test set — the pool is three-way-excluded from the KB by
  construction). Type accuracy 31/40 = 77.5 %, Bosch 75 %.
- **CONTROLLER**: a 2026-10-18 bench re-run covering 100 of the 117 test IPs
  (type 97/100 = 97.0 %, Schneider Electric 96 %, no abstentions), plus an
  older 111-IP validation run.

So the paper's exact Alarm (133/177 = 75.14 % recall; Bosch 83.0 %; id-rate
99.44 %) and Controller (117/117) cells are **not yet reproducible from the
published files**: 137 Alarm and 17 Controller test IPs have no committed
IoTProber prediction anywhere. Closing this requires a re-run, not a
re-upload; the exact commands (guide §3.3, undecided IPs scored as wrong):

```bash
export IOTPROBER_CSV_DATA_DIR=$PWD/platform_data/csv
export RUN=$PWD/runs/known IOTPROBER_VALIDATION_DIR=$RUN/validation \
       IOTPROBER_QUERY_DB_DIR=$RUN/query_db IOTPROBER_PREDICTION_DIR=$RUN/predict
mkdir -p $IOTPROBER_VALIDATION_DIR
cp evaluation/validation/46_features/test_{ALARM,CONTROLLER}_1.csv $IOTPROBER_VALIDATION_DIR/
CUDA_VISIBLE_DEVICES=0 python agent/agent.py --decompose --local --community --reasoning \
    --decision --device ALARM CONTROLLER --llm DEEPSEEK --top_k 5 --gpu 0 --quick_resume \
    --unseen_adapter evaluation/unseen/llama3/results_v2/final_model --unseen_load_in_4bit \
    --drift_dir drift_data/autoencoder_drift_v2
python evaluation/check_results.py --suite known --devices ALARM CONTROLLER
```

### 3.3 Table 8 scoring code, conventions and the B1 column

`split_and_evaluate.py` (published) is the canonical Table-8 scorer;
`score_table8.py` v1.1.0 wraps it and reports both vendor conventions:

```bash
python evaluation/unseen/score_table8.py --json table8_record.json
```

Verified per-cell provenance:

- IoTProber type cells (F1 `2·acc/(1+acc)`): reproduce exactly
  (VPN 0.9963/0.9926; Media Server 0.9980/0.9961).
- IoTProber vendor cells reproduce under **exact (case-sensitive) match**:
  VPN 3572/3933 = 0.9082, Media Server 8932/9525 = 0.9377; the fuzzy
  convention gives 0.9090/0.9378 (the difference is exactly three
  SonicWall/SonicWALL·TP-Link/TP-LINK IPs and one Asus/ASUS IP).
- B2 (Claude-Opus) cells reproduce under **fuzzy normalization**
  (0.6299/0.9654 and 0.8596/0.9186; its own released `type_analysis.csv`
  contains the type numbers verbatim).
  ⇒ Table 8 mixed two vendor-scoring passes; the revision will state one
  convention (we will report both, as `score_table8.py` does).
- **B1 (Mahmood) cells are not reproducible from the released artifacts.**
  The released `predict_{dev}.json` files yield type 0.7802/0.6397 (Media
  Server) and 0.7668/0.6217 (VPN), vendor 0.5630/0.9415 and 0.5645/0.9321,
  under every convention tested (fuzzy/exact × full/labeled/dedup subsets ×
  both label sources). The printed cells (0.8234/0.7512, 0.3421/0.2156,
  0.5123/0.8456, 0.6234/0.8912) admit no integer solution at any released
  denominator (7,804/3,933/12,050/9,525), i.e. they were computed from a B1
  prediction matrix that was not preserved. We will either correct the B1
  column to the recomputed values above or re-run B1; the raw baseline
  outputs and the recomputation script are published either way.

---

## 4. Small runnable package

The minimal reproduction package is the pinned dataset revision + the two
scorers (no GPU, no API keys, ~30 MB download):

```bash
hf download IoTProber/raw_dataset --repo-type dataset --revision f4dbbd53ceb2f3ad44eba79e81744dc78b1866cc \
   --include "evaluation/drift/*" "evaluation/unseen/*" "evaluation/predict/*" \
   "evaluation/validation/type/predict/*" "evaluation/validation/vendor/predict/*" \
   "evaluation/validation/baseline/*" "evaluation/REBUTTAL_ADDENDUM.md" --local-dir pkg && cd pkg
python evaluation/drift/score_table10.py    # Table 10, all rules + detection stage
python evaluation/unseen/score_table8.py    # Table 8, both vendor conventions
```

Both scorers emit a JSON record with scorer version, library versions and
SHA-256 of every input. Larger regeneration (full pipeline, graph, adapter)
uses the run books in `evaluation/README.md` (§2.4–2.8, §3.2–3.5) and
`demo/fullflow/README.md`.

### 4.1 Single-file unseen-detector example (ALARM)

`demo/unseen_detector_demo.py` is one directly runnable file that shows, for
one fixed-seed ALARM validation device: the recorded non-empty LOCAL /
COMMUNITY / REASONING evidence (schema `2026-09-25-v3`), a live 4-bit run of
`UnseenDeviceDetector` (Llama-3.1-8B + the released LoRA adapter, no API keys
needed), and a ranked top-3 vendor prediction — rank 1 from the detector's
`classified_vendor`, ranks 2–3 evidence-scored from the fingerprint and the
community cluster report against the adapter's `known_vendors.json` lexicon —
followed by an explicit top-3 check. Verified reference output (ALARM,
ip 107.89.86.78): evidence non-empty at all three levels, detector classifies
ALARM / Bosch (confidence 1.0, 1.8 s), top-3 = Bosch (detector), Cradlepoint,
Axis → top-3 HIT at rank 1. `--skip-model` runs the evidence + ranking part
on CPU only.

```bash
python demo/unseen_detector_demo.py              # ALARM showcase (1 GPU, ~2 min)
python demo/unseen_detector_demo.py --skip-model # evidence + ranking only, any CPU
```

---

## 5. Permanent identifiers

| Artifact | Identifier |
|---|---|
| Version-of-record archive (code + data) | Zenodo DOI **10.5281/zenodo.23440077** (a new version of this record is being deposited with this addendum; the concept DOI stays citable) |
| Dataset mirror, **pinned content revision** (every input file behind the numbers in §2–§3) | `IoTProber/raw_dataset @ f4dbbd53ceb2f3ad44eba79e81744dc78b1866cc` — https://huggingface.co/datasets/IoTProber/raw_dataset/tree/f4dbbd53ceb2f3ad44eba79e81744dc78b1866cc (this addendum and the manifest are published at the branch tip alongside it) |
| Code mirror | https://github.com/IoTProber/IoTProber (commit that adds this addendum) |
| File-level manifest | `evaluation/MANIFEST_f4dbbd53.txt` (next section) |

HuggingFace commit hashes are immutable: every number in §2–§3 remains
resolvable even as `main` moves ahead.

---

## 6. Open items that need a run, not a publication

1. IoTProber predictions for the 137 Alarm + 17 Controller uncovered test
   IPs (commands in §3.2) — required to make the paper's exact Alarm/
   Controller cells reproducible.
2. B1 (Mahmood) unseen-device predictions: locate the original run or re-run
   (§3.3); until then the B1 column must carry the recomputed values.
3. Zenodo version bump (upload of the pinned revision snapshot; authors'
   Zenodo account required).
