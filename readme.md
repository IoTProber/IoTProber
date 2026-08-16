# IoTProber

**IoTProber** is a graph-based, multi-perspective IoT device identification system that combines active network fingerprinting, heterogeneous graph construction, multi-level RAG retrieval, and LLM reasoning to classify and identify IoT devices at scale.
![Framework Diagram](./arch.png)

---

## Table of Contents

- [Overview](#overview)
- [Dependencies & Environment](#dependencies--environment)
- [External Services](#external-services)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Memory & Speed Optimizations](#memory--speed-optimizations)
- [System Architecture](#system-architecture)
- [Decision Graph Workflow](#decision-graph-workflow)
- [Directory Structure](#directory-structure)
- [Module Descriptions](#module-descriptions)
- [Query Latency](#query-latency)

---

## Overview

IoTProber collects multi-perspective network fingerprints for IoT devices (cameras, routers, NAS, SCADA systems, etc.) via the [Censys](https://censys.io) platform, builds a hierarchical community graph, and identifies unknown devices through a multi-stage pipeline:

1. **Data Acquisition** — Pull per-device IP fingerprints from Censys (25 features across 11 perspectives).
2. **Graph & Embedding Construction** — Cluster fingerprints per perspective, summarize clusters with LLMs, and learn device-level embeddings via a Heterogeneous Graph Transformer (HGT).
3. **Multi-Level Retrieval** — For a query device, perform local vector similarity search, community-level retrieval, and reasoning-path retrieval.
4. **First-Stage Detection & Decision** — A LangGraph-orchestrated decision workflow (`agent/agent.py::IoTDecisionGraph`) first runs **unseen-device detection** (fine-tuned LLaMA-3.1-8B, with a Tavily web-search ReAct loop when evidence is insufficient; emits an independent *new-type* probability and *new-vendor* probability, each with a label or `"none"`); only when **both** probabilities are below 0.5 does it run **in-class concept-drift detection** (Perspective-Aware Contrastive Autoencoder, PACA). Two parallel LLM ReAct branches (Gemini + Claude) then run against the retrieval tools and are combined by confidence-weighted joint voting to produce the final device type and vendor.

---

## Dependencies & Environment

All Python dependencies are listed in **`requirements.txt`**.

### Hardware & Memory Requirements

| Component | Minimum Configuration | Optimal Configuration |
|-----------|----------------------|----------------------|
| **GPU** | NVIDIA V100 / RTX 4070 (16 GB VRAM) | NVIDIA H800 / H100 (80 GB VRAM) |
| **System RAM** | 16 GB | 32+ GB |
| **Storage** | 500 GB SSD | 2T+ GB NVMe SSD |
| **OS** | Ubuntu 20.04+ / macOS | Ubuntu 22.04 LTS |
| **Primary use case** | Embedding + LLaMA inference, vector retrieval | Full pipeline incl. QLoRA LLaMA fine-tuning |

> **Note:** Milvus Lite loads all IVF_FLAT index vectors into memory — 16 GB RAM is the hard minimum. Milvus does not support Windows; Ubuntu or macOS is required.

### Create environment

Either conda **or** a lightweight `venv`/`uv` environment works. The environment is
named `iotprober` to keep it separate from `base`.

```bash
# Option A — conda
conda create -n iotprober python=3.10
conda activate iotprober
pip install -r requirements.txt

# Option B — venv / uv (isolated, no conda needed)
uv venv --python 3.10 --seed .venv-iotprober
uv pip install -r requirements.txt   # or: .venv-iotprober/bin/pip install -r requirements.txt
```

> **LangChain is pinned to the 1.x line** (`requirements.txt`: `langchain==1.1.0`, `langgraph==1.0.4`). The control-plane decision workflow (`agent/agent.py::IoTDecisionGraph`) is built on `langgraph.graph.StateGraph`; the legacy `AgentExecutor` / `create_openai_tools_agent` API still used by `agent/decision.py::DecisionAgent` is restored via the `langchain-classic==1.0.0` compatibility package, which must be installed alongside `langchain` 1.x.

> **GPU-only extras** (`bitsandbytes`, `torch-geometric`, RAPIDS `cuml`) are listed
> but commented out in `requirements.txt` because they cannot be pip-installed on
> macOS/arm64. Enable them on a CUDA Linux host. Without `bitsandbytes`, run the
> unseen LLaMA detector in full/half precision (`load_in_4bit=False`).

### Local model

**Download LLaMA 3.1-8B Instruct from the Hugging Face**

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Put under the folder **./Meta-Llama-3.1-8B-Instruct**

**Download Qwen-3 Embedding Model from the Hugging Face**

https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

Put under the folder **./qwen3_embedding_06b**

The `Qwen3-Embedding-0.6B` model is required for generating 1024-dim fingerprint embeddings.

## External Services

| Service | Purpose | Config |
|---------|---------|--------|
| **Censys** (Search or Platform API) | IP fingerprint data collection | `acquire_data.py` constructor args |
| **Shodan** | Alternative/supplementary fingerprint data collection | `acquire_data.py::ShodanData`, API key in `search_config.json` |
| **Neo4j** | Hierarchical device knowledge graph | `bolt://localhost:7687`, default user `neo4j`, password `<Neo4j password>` (see `graph/build.py`) |
| **Milvus Lite** | Local ANN vector search | Auto-created at `platform_data/csv/local/1/vectorDB/milvus.db` |
| **Claude** (Anthropic) | LLM reasoning & decision | `llm_config.json` → `CLAUDE` |
| **DeepSeek** | LLM reasoning & decomposition | `llm_config.json` → `DEEPSEEK` |
| **Gemini** (Google) | LLM reasoning & clustering summaries | `llm_config.json` → `GEMINI` |
| **OpenAI** | Alternative LLM backend | `llm_config.json` → `OPENAI` |
| **Tavily** | Web search for the `unseen` ReAct loop | `agent/tools/tavily_search.py` |

---

## Configuration

### `llm_config.json`

Stores API credentials and model identifiers for all LLM backends:

```json
{
    "DEEPSEEK": { "API_KEY": "...", "BASE_URL": "https://api.deepseek.com", "MODEL": "deepseek-chat" },
    "CLAUDE":   { "API_KEY": "...", "BASE_URL": "...", "MODEL": "claude-sonnet-4-6" },
    "GEMINI":   { "API_KEY": "...", "BASE_URL": "...", "MODEL": "gemini-3.1-pro-preview" },
    "OPENAI":   { "API_KEY": "...", "BASE_URL": "...", "MODEL": "gpt-4o" }
}
```

### `perspective_info.json`

Defines the 11 fingerprint perspectives used for multi-perspective analysis, their feature columns, and retrieval weights:

| Perspective | Features | Retrieval Weight |
|-------------|---------|-----------------|
| `as` | AS number, name, BGP prefix, country | 0.03 |
| `whois` | Network/organization handle & name | 0.03 |
| `dns` | DNS reverse lookup | 0.04 |
| `body` | HTTP response body | 0.08 |
| `htags` | HTTP HTML tags | 0.08 |
| `sd` | Service port distribution | 0.09 |
| `os` | OS vendor, product, version | 0.10 |
| `sw` | Software vendors, products, versions | 0.11 |
| `hw` | Hardware vendors, products, versions | 0.13 |
| `hfavicons` | HTTP favicon URLs | 0.15 |
| `certificate` | TLS cert subjects, issuers, versions | 0.16 |

### `rag_devices.json`

Lists the 11 known IoT device types that the system is trained to identify:
`NAS`, `NVR`, `POWER_METER`, `BUILDING_AUTOMATION`, `MEDICAL`, `ROUTER`, `PRINTER`, `SCADA`, `CAMERA`, `ALARM`, `CONTROLLER`

---

### `new_devices.json`

Lists the 2 new IoT device types that the system aims to identify:
`MEDIA_SERVER`, `VPN`

---

## Quick Start

```bash
# 1. Activate the environment
conda activate iotprober          # or: source .venv-iotprober/bin/activate

# 2. Configure LLM API keys
vi llm_config.json

# 3. Acquire fingerprint data from Censys (requires Censys credentials)
python acquire_data.py --collect -collect_new --filter_new --filter_old --convert --org_id <Your Org ID> --token <Your Token>
python acquire_data.py --drift
```

**Or You Can Download Dataset from Our Huggging Face Repository including many large dataset that put under the folder **evaluation, platform_data, drift_data**.**

https://huggingface.co/datasets/IoTProber

```bash
# 4. Build the RAG hierarchical graph + vector store (Data Plane).
#    A single command runs every step in the correct dependency order:
#    cluster → build(Layer1 + entity_graph export) → HGT(+comprehensive clustering)
#            → build(Layer2/3) → vector(Milvus)
python graph/construction.py --all --gpu 0

# 5. Run the identification pipeline (Control Plane):
#    retrieval (local + community + reasoning) → LangGraph decision
#    (unseen ⇄ Tavily → gate → drift → prepare → gemini/claude ReAct → finish)
python agent/agent.py --decompose --local --community --reasoning --decision
```

## Memory & Speed Optimizations

To enhance system usability and adaptability across different environments, IoTProber adopts several mechanisms to improve retrieval speed and reduce memory consumption.

### Local Entity Retrieval

- **Memory-mapped, independently-normalised per-perspective storage**: Each device embedding is a `1024 × 11`-dimensional vector; loading all embeddings at once would cause memory overflow. Instead of pre-multiplying perspective weights into the stored vector, each database device `d` stores `d_stored = [d₁/‖d₁‖₂, d₂/‖d₂‖₂, …, d₁₁/‖d₁₁‖₂]` (per-perspective L2 normalisation only) via memory-mapped `.npy` files.
- **Query-time dynamic re-weighting**: For a query with availability mask `mᵢ`, IoTProber computes `W_A = Σᵢ wᵢmᵢ` and the relative weight `wᵢʳᵉˡ = wᵢmᵢ / W_A`, then builds `q_dyn = [w₁ʳᵉˡq₁/‖q₁‖₂, …, w₁₁ʳᵉˡq₁₁/‖q₁₁‖₂]`. The inner product `⟨d_stored, q_dyn⟩` equals the weighted sum of perspective-level cosine similarities, and the score is further scaled by the confidence-decay term `(W_A)^α` (`α = 0.5`) to discount queries with missing perspectives — this avoids unintentionally squaring the perspective weights and supports dynamic renormalisation when fingerprint perspectives are missing.
- **Min-heap Top-k + `argpartition` pre-filtering**: A min-heap-based Top-k selection combined with `argpartition`-based pre-filtering minimises overhead in the inner search loop.
- **Perspective weights** (table above, `perspective_info.json`): certificate and HTTP favicons carry the highest weights since they stay stable even when a device's network environment changes significantly (e.g., a certificate's issuer/subject is tightly bound to the physical device, and firmware versions often ship unique default favicons); hardware, software, and OS follow as strong vendor/product/version signals; DNS, WHOIS, and AS get the lowest weights since the same device type can be deployed across many regions worldwide, making these features highly variable.

### Community-Level Clustering Retrieval (Milvus)

- **IVF_FLAT index with `nlist = 128`**: Perspective clustering vectors are stored in a local Milvus Lite database, one collection per perspective. The IVF_FLAT algorithm partitions the high-dimensional space into 128 clusters; queries first identify the nearest clusters, then perform exact brute-force search only within those clusters, achieving a **30–40% improvement in retrieval speed**.

### Entity Graph Storage & HGT Training (Neo4j)

- Device embeddings are **not** redundantly stored as Neo4j node attributes, reducing graph storage overhead.
- HGT mini-batches are built with **HGSampling**, sampling neighbors with a degree-based probability to limit the impact of "super feature nodes".
- Edge-type attention/message weight matrices use **basis decomposition** `W_φ(e) = Σᵢ αᵢᵠ·Bᵢ` (linear combinations of a small basis set) to avoid a parameter explosion across edge types.
- During local entity retrieval, feature-node degree is capped at **50** so a "super feature node" cannot connect to excessive device nodes and inflate computation.

## System Architecture

```
Censys API
    │
    ▼
acquire_data.py         ← Data collection (25 features / 11 perspectives)
    │
    ▼
platform_data/csv/      ← Raw per-device IP fingerprint CSVs
    │
    ▼  graph/construction.py --all   (single command, runs the steps below in order)
    │
    ├─1─► graph/cluster.py      ← Per-perspective Qwen3 embedding + HDBSCAN + LLM summaries
    │         ▼                    → embedding_{persp}/, community/single/, embedding_local/
    │
    ├─2─► graph/build.py --layer1 --export
    │         ▼                    → Neo4j Layer-1 entity graph + entity_graph/{node,relation}.csv
    │
    ├─3─► graph/HGT.py          ← Heterogeneous Graph Transformer
    │         ▼                    (entity_graph + embedding_local → hgt_embeddings/, 1024-dim)
    │     graph/cluster.py --hgt   ← Comprehensive-view clustering on HGT embeddings
    │         ▼                    → community/embedding_overall/
    │
    ├─4─► graph/build.py --layer23 ← Neo4j Layer-2 / Layer-3 community graphs
    │         ▼
    │     Neo4j (bolt://localhost:7687)
    │
    └─5─► graph/vector.py       ← Store embeddings in Milvus Lite vector DB + local .npz files
              ▼
         platform_data/csv/local/1/vectorDB/milvus.db

graph/update_graph.py   ← Incremental update: insert new devices, predict cluster via the
                            saved HDBSCAN model, partial re-clustering every 50 insertions

Query Device Fingerprint
    │
    ▼
agent/agent.py  (IdentificationAgent)
    │
    ├── agent/decomposition.py  ← DecompositionAgent: parse query intent → problem types
    │
    ├── agent/retrieval.py      ← MultiLevelRetrieval
    │       ├── Local retrieval      (weighted cosine similarity over Milvus vectors)
    │       ├── Community retrieval  (cluster-level LLM reasoning)
    │       └── Reasoning-path retrieval (perspective-chain LLM reasoning)
    │
    └── agent/agent.py::IoTDecisionGraph   ← LangGraph decision workflow (StateGraph)
            ├── unseen ⇄ unseen_tools   agent/unseen.py (LLaMA-3.1-8B+LoRA) ⇄ Tavily ReAct
            ├── gate → drift            agent/drift.py (PACA), only if both unseen probs < 0.5
            └── prepare → gemini_agent/claude_agent ⇄ retrieval tools → finish
                                         confidence-weighted joint voting (agent/decision.py)
                                         → final device_type + vendor

agent/app.py    ← Flask REST API server (optional web interface)
```

---

## Decision Graph Workflow

`agent/agent.py::IoTDecisionGraph` implements the control-plane decision logic as a LangGraph `StateGraph`, replacing the earlier single `create_openai_tools_agent` ReAct implementation. The diagram below summarizes how `agent.py` (the LangGraph orchestrator), `unseen.py` (unseen-device model), and `decision.py` (joint-voting / retrieval tools) collaborate:

```
agent.py (LangGraph orchestrator)

  START → [unseen] ⇄ [unseen_tools] → [gate] → [drift] → [prepare] → [gemini_agent / claude_agent] → [finish] → END
              │            ▲
              │  Tavily    │
              └───────────►┘
          calls detect_unseen()  ⇐  unseen.py (unseen-device detection model)

  gate → drift only when both new_type_probability and new_vendor_probability < 0.5
       → otherwise skips straight to prepare

  prepare fans out to two parallel ReAct branches:
      gemini_agent ⇄ gemini_tools (multi-level retrieval)
      claude_agent ⇄ claude_tools (multi-level retrieval)
  both converge on finish, which applies decision.py's confidence-weighted joint-voting policy.
```

- **agent.py**: the graph orchestrator. Its `_unseen_node` calls `unseen.py`'s `detect_unseen()`; its `unseen_tools` node executes Tavily web search when evidence is insufficient (ReAct loop back to `unseen`).
- **unseen.py**: invoked by `agent.py`'s "model + judgment" module. Produces `needs_web_search` / `search_queries` / `predicted_type` / `predicted_vendor` / …, and its results are spliced into the prompt via the `adapter` path together with web-search results.
- **decision.py**: supplies the downstream decision-stage building blocks used after `gate` — retrieval tools (`local_retrieval` / `community_retrieval` / `reasoning_path_retrieval`) and the `DecisionAgent` joint-voting logic; it has no direct involvement in the `unseen` web-search ReAct loop, which is a closed interaction between `agent.py` and `unseen.py`.

---

## Directory Structure

```

├── acquire_data.py             # Censys / Shodan data acquisition (CensysData, ShodanData)
├── llm.py                      # LLM abstraction layer (Claude / DeepSeek / Gemini / OpenAI)
├── util.py                     # Shared utilities (feature loading, text processing, etc.)
├── llm_config.json             # LLM API keys, base URLs, and model names
├── perspective_info.json       # 11-perspective definitions with feature columns and weights
├── perspective_name.json       # Perspective cluster label names
├── local_used_features.txt     # 25 features used for local retrieval vectors
├── rag_devices.json            # Known IoT device type labels
├── new_devices.json            # Candidate new device type labels
├── requirements.txt            # Pinned Python dependencies
│
├── graph/                      # Graph construction and embedding
│   ├── construction.py         # GraphConstruction: one-shot pipeline driver (cluster→build→HGT→vector)
│   ├── cluster.py              # Multi-perspective clustering (HDBSCAN + KMeans + LLM)
│   ├── cluster_cuml.py         # GPU-accelerated clustering (NVIDIA RAPIDS cuML)
│   ├── build.py                # Neo4j hierarchical graph builder (HierarchicalGraph)
│   ├── build_neo4j.py          # Batch UNWIND Cypher loader (fast bulk Neo4j import)
│   ├── api.py                  # Neo4j Cypher query wrapper (ProtocolGraph)
│   ├── HGT.py                  # Heterogeneous Graph Transformer for device embeddings
│   ├── update_graph.py         # Incremental graph update (new devices, partial re-clustering)
│   └── vector.py               # Milvus Lite vector DB storage and indexing
│
├── agent/                      # Core identification and reasoning pipeline
│   ├── agent.py                # IdentificationAgent + IoTDecisionGraph (LangGraph orchestrator, main entry point)
│   ├── retrieval.py            # MultiLevelRetrieval: local / community / reasoning
│   ├── decomposition.py        # DecompositionAgent: query intent decomposition
│   ├── decision.py             # DecisionAgent: retrieval tools + confidence-weighted joint voting
│   ├── drift.py                # PACA: concept drift detection autoencoder
│   ├── unseen.py               # UnseenDeviceDetector: LLaMA-3.1-8B (+LoRA) unseen detection
│   ├── app.py                  # Flask REST API server (web interface backend)
│   └── tools/
│       └── tavily_search.py    # Tavily web-search tool used by the unseen ReAct loop
│
├── evaluation/                 # Test data and evaluation results (see Hugging Face dataset)
│   ├── split_data.py           # Train/test split utility
│   ├── validation/             # Per-device test CSVs (test_{DEV}_1.csv)
│   ├── unseen/                 # Unseen device evaluation data
│   └── drift/                  # Concept drift evaluation data
│
├── platform_data/              # Raw fingerprint data fetched from Censys/Shodan
│   └── csv/local/1/            # Processed CSV and embedding files
│       ├── ipraw_{DEV}.csv
│       ├── embedding_{persp}/
│       ├── community/
│       ├── hgt_embeddings/
│       └── vectorDB/milvus.db
│
├── rag_data/                   # Raw Censys/Shodan fingerprint CSVs (CensysData.save_path)
├── drift_data/                 # Concept drift detection outputs
├── qwen3_embedding_06b/        # Local Qwen3-Embedding-0.6B model weights
├── Meta-Llama-3.1-8B-Instruct/ # Local LLaMA model weights and LoRA adapters
└── agent/query_db/             # Cached retrieval results per device per IP
    ├── local/{DEV}_local.json
    ├── community/{DEV}_community.json
    └── reasoning/{DEV}_reasoning.json
```

---

## Module Descriptions

### `acquire_data.py` — Data Acquisition
- **Class**: `CensysData`
- Connects to Censys via either the **Search API** (`uid`/`secret`) or the **Platform API** (`org_id`/`personal_access_token`).
- Collects 25 multi-perspective network features per IP: AS info, WHOIS, OS, software stack, hardware, service distribution, HTTP body/tags/favicons, TLS certificates, and DNS reverse records.
- Saves raw IP fingerprints as CSVs under `platform_data/`.

### `graph/cluster.py` — Multi-Perspective Clustering
- **Class**: `GraphClustering`
- Encodes per-perspective feature strings with the local `Qwen3-Embedding-0.6B` model (1024-dim vectors).
- Applies **HDBSCAN** (or **KMeans** fallback) to cluster devices per perspective independently.
- Sends cluster content to an LLM (Claude / DeepSeek / Gemini) to generate human-readable community summaries.
- Outputs: `embedding_{persp}/` (raw vectors), `community/single/{persp}/` (cluster summaries), `community/embedding_overall/` (concatenated 11-perspective embeddings).

### `graph/HGT.py` — Heterogeneous Graph Transformer
- Builds a Device–Feature bipartite graph from `entity_graph/node.csv` and `entity_graph/relation.csv`.
- Device nodes are initialized with the mean of 11 per-perspective embeddings (1024-dim each).
- Feature nodes are initialized with `Qwen3` embeddings of `"feature_name: value"` strings; high-degree feature nodes are penalized by `1/log(degree)`.
- Trains an **HGT** model (PyTorch Geometric) to learn final 1024-dim device representations.
- Outputs saved to `platform_data/csv/local/1/hgt_embeddings/`.

### `graph/vector.py` — Vector Database Storage
- Reads per-perspective embedding CSVs and stores them into a **Milvus Lite** local database (`vectorDB/milvus.db`).
- Creates one Milvus collection per `(perspective, device_type)` pair for efficient ANN search.
- Supports drop/resume modes for incremental updates.

### `graph/build.py` & `graph/api.py` — Neo4j Graph
- `ProtocolGraph` (`api.py`) wraps a Neo4j connection for safe Cypher queries.
- `HierarchicalGraph` (`build.py`) constructs a property graph where device nodes are connected to feature value nodes via typed relationships (e.g., `Has_as_asn`, `Has_cert_subjects`).

### `graph/build_neo4j.py` — Batch Neo4j Loader
- Optimized bulk loader using batch `UNWIND` Cypher queries (5000 rows/batch) to build the Layer-1 hierarchical graph much faster than the row-by-row inserts in `build.py`.

### `graph/update_graph.py` — Incremental Graph Update
- Inserts newly discovered devices into the Layer-1 entity graph without a full rebuild.
- For each perspective (and the comprehensive view), predicts the device's cluster with the saved HDBSCAN model (`approximate_predict`); low-confidence/noise points get a new cluster ID, near-boundary points trigger an LLM cluster re-summary.
- Tracks an update counter and triggers **partial re-clustering** (HDBSCAN re-run on overlapping, updated clusters) every 50 incremental insertions.

### `graph/construction.py` — Pipeline Orchestrator
- **Class**: `GraphConstruction`
- Runs the full graph-construction pipeline (`--cluster → --build → --hgt → --vector`, or `--all`) as isolated subprocesses per step to avoid GPU memory leaks and module conflicts.
- Auto-reorders step flags into the correct data-dependency order (cluster/build must precede HGT; HGT precedes Layer-2/3 community build).

—————————————————————————————————————————

### `agent/agent.py` — Identification Agent & Decision Orchestrator (Main Entry Point)
- **Class**: `IdentificationAgent`
  - Orchestrates the full identification pipeline for batch evaluation: optional query decomposition via `DecompositionAgent`, loads test fingerprints from `evaluation/validation/test_{DEV}_1.csv`, calls `MultiLevelRetrieval` per IP, and saves results to `agent/query_db/{local,community,reasoning}/`.
  - Supports `--quick_resume` mode for fault-tolerant batch execution.
  - `run_decision()` builds and drives an `IoTDecisionGraph` per device, then persists results to `evaluation/predict/`, `evaluation/validation/type/predict/IoTProber/`, and `evaluation/validation/vendor/predict/IoTProber/`.
- **Class**: `IoTDecisionGraph` — a LangGraph `StateGraph` that replaces the legacy `create_openai_tools_agent` ReAct implementation for the control-plane decision stage (see [Decision Graph Workflow](#decision-graph-workflow)).
  - Graph branches: `START → unseen ⇄ unseen_tools (Tavily) → gate → drift (conditional) → prepare → {gemini_agent ⇄ gemini_tools, claude_agent ⇄ claude_tools} → finish → END`.
  - `unseen` calls `unseen.py`'s trained LLaMA-3.1-8B `detect_unseen()`; when evidence is insufficient it emits Tavily web-search tool calls and loops back (bounded by `_MAX_WEB_ITERS`).
  - `gate` routes to `drift` only when both `new_type_probability` and `new_vendor_probability` are below 0.5; otherwise it proceeds directly to `prepare`.
  - `prepare` fans out to two independent, parallel LLM ReAct branches (Gemini + Claude), each bound to the multi-level retrieval tool (bounded by `_MAX_TOOL_CALLS`).
  - `finish` applies `decision.py`'s confidence-weighted joint-voting policy to select the final `device_type` and `vendor`.

### `agent/retrieval.py` — Multi-Level Retrieval
- **Class**: `MultiLevelRetrieval`
- **Local retrieval**: Computes a weighted cosine similarity between the query fingerprint vector and all stored device vectors across 11 perspectives. Returns the top-k most similar devices.
- **Community retrieval**: Feeds local retrieval candidates and the query fingerprint to an LLM for cluster-level reasoning, leveraging community summaries.
- **Reasoning-path retrieval**: Constructs a multi-hop reasoning chain across perspectives, prompting the LLM to progressively narrow down the device identity.
- Manages the Milvus vector store, embedding model (`HuggingFaceEmbeddings`), and LLM calls.

### `agent/decomposition.py` — Query Decomposition
- **Class**: `DecompositionAgent`
- Parses a free-text user query into one or more structured problem types (`DEVICE_TYPE`, `DEVICE_VENDOR`, `DEVICE_LOCATION`, `DEVICE_MODEL`, `SIMILAR_DEVICES`).
- Uses LangChain with LLM backends (Gemini / DeepSeek / OpenAI).
- Results guide which retrieval sub-tasks to activate in `IdentificationAgent`.

### `agent/decision.py` — Decision Agent & Retrieval Tools
- Retrieval tools: `local_retrieval`, `community_retrieval`, `reasoning_path_retrieval` — `@tool`-decorated functions exposed both to `IoTDecisionGraph`'s ReAct branches and to the legacy `DecisionAgent` executor.
- **Class**: `DecisionAgent` (LangChain `create_openai_tools_agent`) — deploys two independent classification agents (Gemini + Claude), each equipped with the three RAG retrieval tools and a 9-step chain-of-thought reasoning prompt.
- Joint-voting policy (`_joint_vote`): combines the two LLM predictions by confidence score into the final `device_type` and `vendor`; when both LLMs agree on `device_type`, confidence is boosted slightly. This same policy is reused by the `finish` node of `IoTDecisionGraph` in `agent.py`.
- Has no direct involvement in the `unseen` web-search ReAct loop — that loop is a closed interaction between `agent.py` and `unseen.py`.

### `agent/drift.py` — Concept Drift Detection
- **Class**: `PerspectiveAwareCAE` (PACA) + `DriftDetector` (single-device inference)
- Detects **in-class concept drift**: same label but shifted feature space (e.g., a camera vendor silently updates firmware).
- Uses perspective-weighted reconstruction loss + contrastive loss:
  - Non-critical perspectives (`whois`, `as`, `dns`, `sd`, `htags`, `body`) → high weight → their reconstruction error signals drift.
  - Critical perspectives (`sw`, `hw`, `os`, `certificate`) → low weight → their change more likely indicates a new device/vendor.
- Detection rule (paper): score `S(x)=Σ_p α_p·mean((x−x̂)²)`; robust threshold `τ = median(S_ref) + γ·MAD(S_ref)` (`γ=3.5`); drift iff `S(x) > τ`; per-perspective z-scores give interpretable attribution.
- `run_drift_detection()` trains PACA and persists `paca_model.pt` + `paca_artifacts.pkl` (vectorizers/scaler/threshold); `DriftDetector.detect_query_device(fingerprint)` scores a single queried device for the first-stage gate. Outputs to `drift_data/autoencoder_drift/`.

### `agent/unseen.py` — Unseen Device Detection
- **Class**: `UnseenDeviceDetector`
- Two-stage detection:
  1. **Statistical pre-screening**: extracts numerical signals from retrieval output (path match scores, similarity gaps, perspective coverage rate).
  2. **LLM reasoning**: constructs a structured prompt with key/non-key perspective data and feeds it to **LLaMA-3.1-8B-Instruct** (local, optionally 4-bit quantized) for chain-of-thought unseen detection.
- Supports **LoRA SFT fine-tuning** to adapt the base model (`evaluation/unseen/llama3/`).
- Returns **two independent probabilities** — `new_type_probability` and `new_vendor_probability` — each gated at 0.5: a probability `> 0.5` emits the concrete new type/vendor label, otherwise that field is `"none"` (insufficient evidence keeps the probability below 0.5). Also returns `is_unseen`, `confidence`, and reasoning.
- Called directly by the `unseen` node in `agent.py`'s `IoTDecisionGraph`; when `needs_web_search` is set, `agent.py` drives a Tavily-backed ReAct loop and feeds the search results back into `detect_unseen()` on the next iteration.

### `agent/app.py` — Flask REST API Server
- Serves a lightweight web interface (`agent/web/`) and JSON API for interactive querying.
- Endpoints: `/api/decompose` (query decomposition only), `/api/retrieve` (decomposition + multi-level retrieval), `/api/history` (GET/DELETE), `/api/known-problems`, `/api/config` (GET) and `/api/config/llm` (POST, switch active LLM), `/api/health`.

—————————————————————————————————————————

### `llm.py` — LLM Abstraction Layer
- **Class**: `LLM`
- Unified interface for calling Claude (Anthropic), DeepSeek, Gemini, and OpenAI.
- Loads API keys and model names from `llm_config.json`.
- Supports single-turn `chat_with_llm()`, batch `batch_chat_with_llm()`, and JSON-mode responses.

### `util.py` — Shared Utilities
- Device label loading (`load_all_dev_labels`, `load_new_dev_labels`)
- Perspective config loading (`load_perspective_info`, `load_perspective_cluster_info`)
- Feature list loading (`load_local_used_features`)
- Vector preprocessing with weighted L2 normalization (`preprocess_vector`)
- LLM JSON output parsing (`convert_json_from_str`)
- Text chunking for embedding fallback (`chunk_text`)

---

## Query Latency

End-to-end latency was measured on the full production dataset: a one-time graph-construction pass over all collected devices, followed by a query-time evaluation pass over the held-out test set processed through the complete retrieval + decision pipeline.

| Stage | Devices | Total Time | Avg. Time per Query |
|---|---|---|---|
| Graph construction (one-time) | 421,759 | 148.25 h | — |
| End-to-end query (retrieval + decision) | 76,319 | 58.02 h | 2.73 s |

Graph construction is a one-time, offline cost amortized across all future queries, while the per-query latency reflects the online cost of identifying a single device at inference time. The graph-construction time covers the full offline pipeline: (1) per-perspective embedding computation over all fingerprints (`graph/cluster.py`, local Qwen3-Embedding-0.6B), (2) embedding clustering — HDBSCAN/KMeans per perspective plus the comprehensive-view clustering over HGT embeddings, (3) vector-database storage and `.npz`/`.npy` embedding-file generation (`graph/vector.py` Milvus Lite ingestion and the memory-mapped local embedding files consumed by retrieval), and (4) Neo4j hierarchical graph database construction — building Layer-1 entity nodes/relationships and Layer-2/Layer-3 community nodes/relationships (`graph/build.py` / `graph/build_neo4j.py`).

Query latency is primarily influenced by three factors: (1) the throughput of the local Qwen3 embedding computation used for local/community retrieval, (2) the response latency of the external LLM APIs (Gemini, Claude, DeepSeek) invoked during multi-level retrieval and the decision graph's ReAct loops, and (3) the CPU performance available for memory-mapped vector search and feature preprocessing. Variance across queries is largely attributable to these three factors rather than to the retrieval algorithm itself.
