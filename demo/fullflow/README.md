# Full-Pipeline Evaluation

The fullflow evaluator runs the complete IoTProber pipeline:

```text
decomposition -> local retrieval -> community retrieval -> reasoning path
              -> unseen/drift checks -> Gemini + Claude decision
```

`run_fullflow_demo.py` uses each device type's fixed-seed random
`validation_cases.json` by default. The three post-selected UI examples in
`cases.json` are used only when `--showcase-only` is explicitly supplied.

Each `results.json` row retains successful and failed predictions together with
the local, community, and reasoning evidence. `showcase.json` surfaces failures
first instead of silently selecting only correct results. Writes are atomic and
long runs resume only records created by the current pipeline schema.

A case is `completed` only when it has both a decision and current-schema
records for all three retrieval levels. The evaluator reports completion rate,
accuracy among completed cases, and end-to-end accuracy over every selected
case; unavailable evidence therefore cannot disappear from a metric denominator.

## Invalidated legacy results

Results generated before retrieval schema `2026-09-25-v3` are not valid
evaluation measurements. Those runs could retrieve the query IP itself, exposed
the true-type filename to the decision tool, interpreted scores greater than one
as cosine similarity, and treated missing community reports as a synthetic 0.5
match. They must not be quoted as IoTProber accuracy.

## Run

```bash
EL=/root/anaconda3/envs/elastic_slm/bin/python

# Refresh the complete random validation material without changing curated UI cases.
$EL demo/select_demo_data.py --materialize-validation

# Evaluate all 11 types. Compatible completed IPs are resumed.
CUDA_VISIBLE_DEVICES=0 $EL demo/run_fullflow_demo.py

# Explicitly run only the curated UI examples.
CUDA_VISIBLE_DEVICES=0 $EL demo/run_fullflow_demo.py --showcase-only

# Discard generated retrieval/prediction caches for the requested types.
CUDA_VISIBLE_DEVICES=0 $EL demo/run_fullflow_demo.py --fresh --types CAMERA NAS

# Compute final accuracy, per-layer retrieval accuracy/availability, Recall@K,
# self-match/range integrity checks, high-confidence errors, and 10-bin ECE.
$EL demo/evaluate_fullflow.py
```

The full run requires the configured LLM endpoints, the unseen adapter, the
drift artifacts, and the local/community index data.
