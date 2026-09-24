# IoTProber Demo — Minimal Showcase for 11 Device Types

For each of the 11 RAG device types defined in `config/rag_devices.json`, 40 validation-set test cases are randomly selected. They are evaluated through the production inference pipeline with the v2 adapter
(`UnseenDeviceDetector._build_aligned_prompt → _generate_classification → _classification_novelty_result`).
The top three correctly classified cases with the highest type confidence are retained, with vendor diversity preferred within each device type.

## Three-Stage Candidate Pool Filtering

Before reviewing the rankings, it is important to understand how the demo candidate pool is filtered:

1. **Training/validation IP exclusion** (65,027 IPs) — otherwise the demo would measure memorization rather than recognition.
2. **Cross-type duplicate IP exclusion** (15,346 IPs) — the same fingerprint appears in multiple device-type files with conflicting labels.
3. **Generic cloud host exclusion** (approximately 8,900 rows) — AWS, GCP, Azure, and similar hosts are identified through `dns-reverse`, `as-name`, and `whois`; at one point, 85% of the POWER_METER corpus consisted of AWS EC2 hosts.

Candidate-pool accuracy after filtering (40 candidates per type unless otherwise noted):

| Device Type | Correct | | Device Type | Correct | | Device Type | Correct |
|---|---|---|---|---|---|---|---|
| ALARM | 40/40 | | CONTROLLER | 40/40 | | NVR | 40/40 |
| POWER_METER | 38/38 | | MEDICAL | 39/40 | | NAS | 39/40 |
| SCADA | 39/40 | | BUILDING_AUTOMATION | 38/40 | | PRINTER | 37/40 |
| CAMERA | 36/40 | | ROUTER | 34/40 | | | |

> In this demo, a result is considered "correct" when it matches the directory label in the ipraw corpus. The corpus labels themselves contain noise, including cross-type duplicates and mislabeled cloud hosts; filters 2 and 3 above are intended to remove as many of these samples as possible.
> CONTROLLER has fewer than 40 rich-fingerprint samples, so its candidates are drawn from the full dataset and marked as `rich_filter_bypassed` in `selection_summary.json`.

## Directory Structure

```
demo/
├── {TYPE}/cases.json            # Three cases: full fingerprint + classification + novelty + drift (+ CAMERA vector neighbors)
├── selection_summary.json       # Pool size, exclusions, correct count, maximum confidence, and filter fallback for each type
├── select_demo_data.py          # Selection script (--candidates / --rank R / --merge, sharded across eight GPUs)
├── demo_showcase.ipynb          # Executed minimal showcase notebook with charts
├── app.py + static/index.html   # Local visualization web interface
├── screenshot_ui.py             # Optional headless-browser screenshots for visual regression testing
└── README.md
```

## Environment Requirements

| Task | Environment | Reason |
|---|---|---|
| View the notebook or open the web interface | `iotprober` | Only pandas, matplotlib, and Flask are required. |
| **Online classification** (`/api/classify`) | **`elastic_slm`** | Uses the same PyTorch/Transformers numerical environment that generated `cases.json`, making the results comparable. |
| **Run selection again** (`select_demo_data.py`) | **`elastic_slm`** | The script imports the training code and depends on `datasets`, which is not installed in the `iotprober` environment. |

```bash
EL=/root/anaconda3/envs/elastic_slm/bin/python     # Inference and selection
IO=/root/anaconda3/envs/iotprober/bin/python       # Web interface and notebook only
```

## Usage

```bash
# 1) Notebook (static showcase; no GPU required)
cd demo && $IO -m jupyter nbconvert --to notebook --execute --inplace demo_showcase.ipynb
#    Alternatively, open the already-executed demo_showcase.ipynb directly.

# 2) Web interface → http://localhost:5001
cd demo && $EL app.py
#    "Online Classification" loads the v2 adapter on demand. In testing, it used
#    approximately 8.7 GB of VRAM on one GPU and took about one minute on first load.
#    The classification result matches the stored result, while confidence may differ
#    by approximately 1e-5 because of floating-point nondeterminism in 4-bit inference.
#    GPU 0 is used by default. With device_map="auto" in unseen.py, the model would
#    otherwise occupy approximately 27 GB across all eight GPUs. To select another GPU:
#    DEMO_GPU=3 $EL app.py

# 3) Run selection again (only needed when the corpus changes)
#    Prerequisites that are easy to overlook:
#      · /dev/shm/ipraw/ipraw_{TYPE}.csv  ← Raw fingerprints for 11 device types,
#        approximately 8.4 GB in tmpfs and lost after a reboot.
#        Rebuild by downloading with:
#        hf_hub_download('IoTProber/raw_dataset', 'platform_data/rag/ipraw_files.tar.gz',
#                        repo_type='dataset')
#        and then extracting the archive to /dev/shm/ipraw.
#      · /dev/shm/demo_select             ← Intermediate candidates.json files,
#        also stored in tmpfs.
#    Run the following three commands in order during the same boot session:
cd demo
$EL select_demo_data.py --candidates                   # Build the candidate pool with all three filters (~5 minutes)
for i in 0 1 2 3 4 5 6 7; do                            # Eight-GPU sharded inference (~2 minutes)
  CUDA_VISIBLE_DEVICES=$i $EL select_demo_data.py --rank $i --world_size 8 &
done
wait
$EL select_demo_data.py --merge                        # Selection + drift + CAMERA neighbors (~2 minutes)
```

## Showcase Contents

Each case contains the complete 46-column fingerprint, adapter classification (device type, vendor, two confidence scores, and two novelty probabilities), a PACA drift score (τ=8.7755, loaded from artifacts rather than hard-coded), and vector neighbors for CAMERA cases.

```bash
python - <<'PY'
import json
c = json.load(open('demo/CAMERA/cases.json'))[0]
print(c['ip'], c['result']['classified_type'], c['result']['classified_vendor'], c['result']['type_confidence'])
PY
```
