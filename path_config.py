"""Central filesystem paths used by the graph and agent pipelines.

Only stable directories and fixed resource files belong here.  Filenames that
depend on a device, perspective, or run stay close to the code that creates
them.
"""

import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(ROOT_DIR, "graph")
AGENT_DIR = os.path.join(ROOT_DIR, "agent")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# Data roots
PLATFORM_DATA_DIR = os.path.join(ROOT_DIR, "platform_data")
CSV_DATA_DIR = os.path.abspath(
    os.environ.get(
        "IOTPROBER_CSV_DATA_DIR",
        "/mnt/zwj_ckpt/iotprober/platform_data/csv",
    )
)
LOCAL_DATA_DIR = os.path.join(CSV_DATA_DIR, "local", "1")
RAG_DATA_DIR = os.path.join(CSV_DATA_DIR, "rag")
LABEL_DATA_DIR = os.path.join(CSV_DATA_DIR, "label")
ALL_DATA_DIR = os.path.join(CSV_DATA_DIR, "all")

# Community and vector data
LOCAL_SINGLE_COMMUNITY_DIR = os.path.join(LOCAL_DATA_DIR, "community", "single")
LOCAL_OVERALL_COMMUNITY_DIR = os.path.join(
    LOCAL_DATA_DIR, "community", "embedding_overall"
)
RAG_SINGLE_COMMUNITY_DIR = os.path.join(RAG_DATA_DIR, "community", "single")
RAG_OVERALL_COMMUNITY_DIR = os.path.join(
    RAG_DATA_DIR, "community", "embedding_overall"
)
RAG_HGT_COMMUNITY_DIR = os.path.join(RAG_DATA_DIR, "community", "embedding_HGT")
RAG_MINOR_REVISION_DIR = os.path.join(RAG_DATA_DIR, "minor_revision")
LOCAL_VECTOR_DB_DIR = os.path.join(LOCAL_DATA_DIR, "vectorDB")
RAG_VECTOR_DB_DIR = os.path.join(RAG_DATA_DIR, "vectorDB")

# Models and generated artifacts
EMBEDDING_MODEL_DIR = os.path.join(ROOT_DIR, "qwen3_embedding_06b")
DEFAULT_LLM_MODEL_DIR = os.path.join(ROOT_DIR, "Meta-Llama-3.1-8B-Instruct")
ENTITY_GRAPH_DIR = os.path.join(ROOT_DIR, "entity_graph")
GRAPH_MODEL_DIR = os.path.join(GRAPH_DIR, "model")
GRAPH_UPDATE_DIR = os.path.join(GRAPH_DIR, "update")
DRIFT_OUTPUT_DIR = os.path.join(ROOT_DIR, "drift_data", "autoencoder_drift")

# The full Hugging Face snapshot already contains these embeddings.
HGT_INPUT_EMBEDDING_DIR = os.path.join(RAG_DATA_DIR, "embedding_local")

# Agent and evaluation output
QUERY_DB_DIR = os.path.abspath(os.environ.get(
    "IOTPROBER_QUERY_DB_DIR", os.path.join(AGENT_DIR, "query_db")
))
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
VALIDATION_DIR = os.path.abspath(os.environ.get(
    "IOTPROBER_VALIDATION_DIR", os.path.join(EVALUATION_DIR, "validation")
))
PREDICTION_DIR = os.path.abspath(os.environ.get(
    "IOTPROBER_PREDICTION_DIR", os.path.join(EVALUATION_DIR, "predict")
))
PREDICTION_RESULT_DIR = os.path.join(PREDICTION_DIR, "result")
TYPE_PREDICTION_DIR = os.path.join(
    VALIDATION_DIR, "type", "predict", "IoTProber"
)
VENDOR_PREDICTION_DIR = os.path.join(
    VALIDATION_DIR, "vendor", "predict", "IoTProber"
)
UNSEEN_DATASET_DIR = os.path.join(EVALUATION_DIR, "unseen", "llama3", "dataset")
UNSEEN_SUMMARY_CACHE_DIR = os.path.join(
    EVALUATION_DIR, "unseen", "llama3", "data_summary_cache"
)

# Fixed configuration and metadata files
LLM_CONFIG_TEMPLATE_FILE = os.path.join(CONFIG_DIR, "llm_config.json")
LOCAL_LLM_CONFIG_FILE = os.path.join(CONFIG_DIR, "llm_config.local.json")
LLM_CONFIG_FILE = (
    LOCAL_LLM_CONFIG_FILE
    if os.path.isfile(LOCAL_LLM_CONFIG_FILE)
    else LLM_CONFIG_TEMPLATE_FILE
)
# All API credentials (LLM + Tavily) live in the single config/llm_config.json —
# there is no second, flat-key config file to keep in sync.
RAG_DEVICES_FILE = os.path.join(CONFIG_DIR, "rag_devices.json")
ALL_DEVICES_FILE = os.path.join(CONFIG_DIR, "all_IoT_devices.json")
NEW_DEVICES_FILE = os.path.join(CONFIG_DIR, "new_devices.json")
RAG_DOMAIN_FILE = os.path.join(CONFIG_DIR, "rag_domain.json")
PERSPECTIVE_INFO_FILE = os.path.join(CONFIG_DIR, "perspective_info.json")
PERSPECTIVE_NAME_FILE = os.path.join(CONFIG_DIR, "perspective_name.json")
LOCAL_USED_FEATURES_FILE = os.path.join(CONFIG_DIR, "local_used_features.txt")
CLASSIFICATION_METADATA_FILE = os.path.join(UNSEEN_DATASET_DIR, "known_vendors.json")
UNSEEN_SUMMARY_CACHE_FILE = os.path.join(
    UNSEEN_SUMMARY_CACHE_DIR, "summary_cache.jsonl"
)

# Logs with a stable location (independent of the process working directory)
AGENT_LOG_FILE = os.path.join(ROOT_DIR, "agent.log")
RETRIEVAL_LOG_FILE = os.path.join(ROOT_DIR, "retrieval.log")
VECTOR_LOG_FILE = os.path.join(ROOT_DIR, "store_vector.log")
DECISION_LOG_FILE = os.path.join(AGENT_DIR, "decision.log")
UNSEEN_LOG_FILE = os.path.join(AGENT_DIR, "unseen.log")
HGT_LOG_FILE = os.path.join(ROOT_DIR, "HGT.log")
CONSTRUCTION_LOG_FILE = os.path.join(GRAPH_DIR, "construction.log")
