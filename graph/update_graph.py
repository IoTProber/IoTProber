"""
Graph incremental update algorithm.

When new IoT devices are discovered, this module updates the hierarchical graph
in-place without rebuilding from scratch:

1.  Insert each new device node into the Layer-1 entity graph (Neo4j
    Device–Feature bipartite graph).

2.  For every single-perspective and the comprehensive-perspective, use the
    saved HDBSCAN clusterer (joblib) to predict the cluster of the new device
    via ``approximate_predict``.  Depending on confidence / distance to the
    cluster centre:
      * low confidence or cluster == -1  →  assign a brand-new cluster ID.
      * distance < min_10_distance       →  trigger an LLM re-summary of that
        cluster (logged with timestamp, token usage, elapsed time) and update
        min_10_distance.
      * distance > max_distance          →  update max_distance.
    Neo4j Cluster nodes that received new members are marked ``updated=False``.

3.  A global update counter tracks how many devices have been incrementally
    inserted.  When it exceeds 50, a *partial re-clustering* is triggered:
      * Collect all device nodes belonging to clusters with ``updated=False``.
      * Find overlapping clusters whose ``max_distance`` circles intersect
        (centre distance ≤ max_distance₁ + max_distance₂).
      * Re-run HDBSCAN on the union of those device embeddings.
      * Assign new cluster IDs starting from 0, offset so they do not collide
        with any non-participating cluster ID.
    The counter is reset after each partial re-clustering.
"""

import os
import sys
import json
import time
import logging
import datetime
import hashlib
import numpy as np
import pandas as pd
import hdbscan
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import load_perspective_info, load_perspective_cluster_info, load_local_used_features
from llm import LLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINOR_REVISION = os.path.join(BASE, "platform_data", "csv", "rag", "minor_revision")
SINGLE_DIR = os.path.join(MINOR_REVISION, "single")
OVERALL_DIR = os.path.join(MINOR_REVISION, "embedding_overall")

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "avs01046"
NEO4J_UPDATED_URL = "neo4j://localhost:7688"

PCA_COLS = [f"pca{i}" for i in range(1, 257)]
PERSPECTIVES = [
    "as", "body", "certificate", "dns", "hfavicons",
    "hpart", "htags", "http", "hw", "os", "sd", "sw", "whois",
]
MIN_CLUSTER_SIZE = 20
RECLUSTER_THRESHOLD = 50          # trigger partial re-clustering after this many updates
CONFIDENCE_THRESHOLD = 0.3        # approximate_predict membership strength below this → new cluster
SUMMARY_LLM = "CLAUDE"            # LLM config key used for re-summarisation

# All update-related output files are stored in this directory
UPDATE_DIR = os.path.join(BASE, "graph", "update")
os.makedirs(UPDATE_DIR, exist_ok=True)

SUMMARY_LOG = os.path.join(UPDATE_DIR, "summary_log.jsonl")
COUNTER_FILE = os.path.join(UPDATE_DIR, "update_counter.json")
RECLUSTER_COUNT_FILE = os.path.join(UPDATE_DIR, "recluster_count.json")
NEXT_ID_FILE = os.path.join(UPDATE_DIR, "next_cluster_ids.json")
REPORT_FILE = os.path.join(UPDATE_DIR, "test_update_report.json")

# LLM pricing (USD per 1M tokens)
LLM_PRICING = {
    "CLAUDE": {"input": 3.0, "output": 15.0},
    "DEEPSEEK": {"input": 0.27, "output": 1.10},
    "GEMINI": {"input": 0.15, "output": 0.60},
}

# Device types loaded from rag_devices.json
def _load_all_test_devices():
    path = os.path.join(BASE, "rag_devices.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["IoT"]

ALL_TEST_DEVICES = _load_all_test_devices()
DEVICES_WITH_CLUSTERERS = ["ALARM", "CONTROLLER"]


# ─── Helpers ─────────────────────────────────────────────────────────────

def _value_hash(val: str) -> str:
    """Short MD5 hash for feature value (matches build_neo4j.py)."""
    return hashlib.md5(val.encode("utf-8")).hexdigest()[:16]


def _rel_type(feat: str) -> str:
    return "Has_" + feat.replace("-", "_").replace(".", "_")


def _load_cluster_info(perspective: str) -> dict:
    """Load cluster_info.json for a single-perspective or 'comprehensive'."""
    if perspective == "comprehensive":
        path = os.path.join(OVERALL_DIR, "cluster_info.json")
    else:
        path = os.path.join(SINGLE_DIR, f"embedding_{perspective}", "cluster_info.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("[cluster_info] Corrupted JSON for %s, returning empty dict", perspective)
        return {}


def _save_cluster_info(perspective: str, info: dict):
    if perspective == "comprehensive":
        path = os.path.join(OVERALL_DIR, "cluster_info.json")
    else:
        path = os.path.join(SINGLE_DIR, f"embedding_{perspective}", "cluster_info.json")
    # Atomic write: write to temp file then rename
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=float)
    os.replace(tmp_path, path)


def _load_clusterer(perspective: str, dev: str):
    """Load a saved HDBSCAN clusterer from joblib."""
    if perspective == "comprehensive":
        path = os.path.join(OVERALL_DIR, f"clusterer_{dev}.joblib")
    else:
        path = os.path.join(SINGLE_DIR, f"embedding_{perspective}", f"clusterer_{dev}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _compute_distance(point: np.ndarray, center: np.ndarray) -> float:
    return float(np.linalg.norm(point - center))


def _recompute_cluster_stats(points: np.ndarray) -> dict:
    """Recompute center, max_distance, min_10_distance for a set of points."""
    if len(points) == 0:
        return {}
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    max_dist = float(dists.max())
    if len(dists) >= 10:
        min_10 = float(np.sort(dists)[9])
    else:
        min_10 = -1
    return {
        "center": center.tolist(),
        "max_distance": max_dist,
        "min_10_distance": min_10,
        "node_count": len(points),
    }


# ─── Test-update helpers ──────────────────────────────────────────────────

TEST_DIR = os.path.join(BASE, "evaluation", "validation", "46_features")
LABEL_DIR = os.path.join(BASE, "platform_data", "csv", "label")
ALL_DIR = os.path.join(BASE, "platform_data", "csv", "all")
RAG_DIR = os.path.join(BASE, "platform_data", "csv", "rag")
LOCAL_FEATURES = os.path.join(BASE, "local_used_feature.txt")


def _load_features_list():
    with open(LOCAL_FEATURES) as f:
        return [l.strip() for l in f if l.strip()]


def _load_vendor_map(dev):
    path = os.path.join(LABEL_DIR, f"label_{dev}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["ip", "vendor"])
    df["ip"] = df["ip"].astype(str).str.strip()
    df["vendor"] = df["vendor"].fillna("Unknown").astype(str).str.strip()
    return dict(zip(df["ip"], df["vendor"]))


def _load_rag_pca_embeddings(dev):
    """Load per-perspective and comprehensive PCA embeddings for RAG devices."""
    embeddings = {}
    for persp in PERSPECTIVES:
        path = os.path.join(SINGLE_DIR, f"embedding_{persp}", f"ipraw_{dev}_embedding_{persp}_pca.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            pca_cols = [c for c in df.columns if c.startswith("pca")]
            embeddings[persp] = df[["ip"] + pca_cols].copy()
    path = os.path.join(OVERALL_DIR, f"ipraw_{dev}_embedding_overall_pca.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        emb_cols = [c for c in df.columns if c not in ("ip", "cluster")]
        embeddings["comprehensive"] = df[["ip"] + emb_cols].copy()
    return embeddings


def _load_rag_features(dev):
    """Load raw features for RAG devices to compute similarity."""
    path = os.path.join(RAG_DIR, f"ipraw_{dev}.csv")
    if not os.path.exists(path):
        path = os.path.join(ALL_DIR, f"ipraw_{dev}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _prepare_test_devices(dev, features_list):
    """Load test devices for a device type, with features, vendor, and embeddings."""
    test_path = os.path.join(TEST_DIR, f"test_{dev}_1.csv")
    if not os.path.exists(test_path):
        log.warning(f"Test file not found: {test_path}")
        return []

    df = pd.read_csv(test_path)
    df["ip"] = df["ip"].astype(str).str.strip()

    vendor_map = _load_vendor_map(dev)
    available_features = [f for f in features_list if f in df.columns]

    devices = []
    for _, row in df.iterrows():
        ip = row["ip"]
        features = {}
        for feat in available_features:
            val = row.get(feat, "")
            if pd.notna(val):
                features[feat] = str(val).strip()
            else:
                features[feat] = ""

        devices.append({
            "ip": ip,
            "device_type": dev,
            "features": features,
            "vendor": vendor_map.get(ip, "Unknown"),
            "embeddings": {},
        })

    return devices


def _add_embeddings_nearest_neighbor(devices, dev, features_list):
    """Add PCA embeddings to test devices using nearest-neighbor from RAG data."""
    rag_embeddings = _load_rag_pca_embeddings(dev)
    if not rag_embeddings:
        log.warning(f"No RAG PCA embeddings for {dev}, skipping embedding assignment")
        return

    rag_features = _load_rag_features(dev)
    if rag_features is None:
        log.warning(f"No RAG features for {dev}, using random RAG device for embeddings")
        for persp, emb_df in rag_embeddings.items():
            if len(emb_df) > 0:
                first_row = emb_df.iloc[0]
                emb_cols = [c for c in emb_df.columns if c != "ip"]
                emb = first_row[emb_cols].values.astype(float).tolist()
                for d in devices:
                    d["embeddings"][persp] = emb
        return

    rag_features["ip"] = rag_features["ip"].astype(str).str.strip()
    rag_feature_dict = {}
    for _, row in rag_features.iterrows():
        rag_feature_dict[row["ip"]] = {col: str(row.get(col, "")).strip() for col in features_list if col in row.index}

    persp_emb_maps = {}
    for persp, emb_df in rag_embeddings.items():
        emb_cols = [c for c in emb_df.columns if c != "ip"]
        ip_to_emb = {}
        for _, row in emb_df.iterrows():
            ip_to_emb[row["ip"]] = row[emb_cols].values.astype(float).tolist()
        persp_emb_maps[persp] = ip_to_emb

    for d in devices:
        test_feats = d["features"]
        best_ip = None
        best_score = -1
        for rag_ip, rag_feats in rag_feature_dict.items():
            score = sum(1 for f in features_list if f in test_feats and f in rag_feats and test_feats[f] and rag_feats[f] and test_feats[f] == rag_feats[f])
            if score > best_score:
                best_score = score
                best_ip = rag_ip

        if best_ip is None:
            for persp, ip_to_emb in persp_emb_maps.items():
                if ip_to_emb:
                    d["embeddings"][persp] = ip_to_emb[list(ip_to_emb.keys())[0]]
        else:
            for persp, ip_to_emb in persp_emb_maps.items():
                if best_ip in ip_to_emb:
                    d["embeddings"][persp] = ip_to_emb[best_ip]
                elif ip_to_emb:
                    d["embeddings"][persp] = ip_to_emb[list(ip_to_emb.keys())[0]]

    log.info(f"[Embedding] Assigned nearest-neighbor embeddings to {len(devices)} {dev} test devices")


def _get_db_stats(g):
    """Get Neo4j node/relationship counts by label."""
    stats = {}
    for r in g.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC").data():
        stats[r["l"]] = r["c"]
    stats["_relationships"] = g.run("MATCH ()-[r]->() RETURN count(r) AS c").data()[0]["c"]
    return stats


# ─── Main class ──────────────────────────────────────────────────────────

class GraphUpdater:
    """
    Incremental graph update engine.

    Usage::

        updater = GraphUpdater()
        updater.update_devices([
            {"ip": "1.2.3.4", "device_type": "ALARM",
             "features": {"as-asn": "123", ...},
             "embeddings": {"as": [0.1, ...], "comprehensive": [0.2, ...]}},
            ...
        ])
    """

    def __init__(self, neo4j_graph=None, llm=None):
        self.base_path = BASE
        self.perspective_info = load_perspective_info()
        self.perspective_cluster_info = load_perspective_cluster_info()
        self.perspective_names = list(self.perspective_cluster_info.keys())
        self.features = load_local_used_features()

        # Neo4j connection (py2neo Graph or compatible)
        if neo4j_graph is not None:
            self.db = neo4j_graph
        else:
            try:
                from py2neo import Graph
                self.db = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
            except Exception as e:
                log.warning("Neo4j connection failed: %s – operating in no-DB mode", e)
                self.db = None

        # LLM for cluster re-summarisation
        self.llm = llm if llm is not None else LLM()

        # Global update counter (persisted across runs)
        self.counter_file = COUNTER_FILE
        self.update_counter = self._load_counter()

        # Re-cluster event counter (persisted across runs)
        self.recluster_count_file = RECLUSTER_COUNT_FILE
        self.recluster_count = self._load_recluster_count()

        # Per-perspective next-new-cluster-ID tracker (persisted)
        self.next_id_file = NEXT_ID_FILE
        self.next_cluster_ids = self._load_next_ids()

    # ── persistence helpers ──────────────────────────────────────────

    def _load_counter(self) -> int:
        if os.path.exists(self.counter_file):
            with open(self.counter_file, "r") as f:
                return json.load(f).get("counter", 0)
        return 0

    def _save_counter(self):
        with open(self.counter_file, "w") as f:
            json.dump({"counter": self.update_counter}, f)

    def _load_recluster_count(self) -> int:
        if os.path.exists(self.recluster_count_file):
            with open(self.recluster_count_file, "r") as f:
                return json.load(f).get("recluster_count", 0)
        return 0

    def _save_recluster_count(self):
        with open(self.recluster_count_file, "w") as f:
            json.dump({"recluster_count": self.recluster_count}, f)

    def _load_next_ids(self) -> dict:
        if os.path.exists(self.next_id_file):
            with open(self.next_id_file, "r") as f:
                return json.load(f)
        return {}

    def _save_next_ids(self):
        with open(self.next_id_file, "w") as f:
            json.dump(self.next_cluster_ids, f)

    def _get_next_cluster_id(self, perspective: str, dev: str) -> int:
        """Return a new unique cluster ID for (perspective, dev) and increment.

        Ensures the new ID does not collide with existing cluster_info keys
        (HDBSCAN labels) NOR with Neo4j cluster_ids (global counter).
        """
        key = f"{perspective}_{dev}"
        cid = self.next_cluster_ids.get(key, 0)
        # Ensure it doesn't collide with existing IDs in cluster_info
        info = _load_cluster_info(perspective)
        existing = set()
        if dev in info:
            existing = set(int(k) for k in info[dev].keys())
        # Also check Neo4j cluster_ids to avoid collisions
        neo4j_ids = self._get_all_cluster_ids(perspective)
        existing = existing | set(int(x) for x in neo4j_ids)
        while cid in existing:
            cid += 1
        self.next_cluster_ids[key] = cid + 1
        self._save_next_ids()
        return int(cid)

    # ── Step 1: Layer-1 entity graph insertion ───────────────────────

    def _insert_device_node(self, ip: str, device_type: str, features: dict, vendor: str = "Unknown"):
        """
        Insert a single Device node + its Feature nodes into Neo4j Layer-1
        entity graph, mirroring build_neo4j.build_layer1 logic.
        """
        if self.db is None:
            log.warning("[L1] No Neo4j connection, skipping node insertion for %s", ip)
            return

        # MERGE Device node with vendor
        self.db.run(
            "MERGE (d:Device {ip: $ip, device_type: $dev}) SET d.vendor = $vendor",
            ip=ip, dev=device_type, vendor=vendor,
        )

        for feat_name, feat_value in features.items():
            val = str(feat_value).strip()
            if not val:
                continue
            vh = _value_hash(val)
            rt = _rel_type(feat_name)
            self.db.run(
                "MERGE (d:Device {ip: $ip, device_type: $dev}) "
                "MERGE (f:Feature {feature_name: $feat, value_hash: $vh}) "
                "SET f.value = $val "
                "MERGE (d)-[:%s]-(f)" % rt,
                ip=ip, dev=device_type, feat=feat_name, vh=vh, val=val,
            )

        log.info("[L1] Inserted device %s (%s) with %d features", ip, device_type, len(features))

    # ── Step 2: Per-perspective cluster prediction & update ──────────

    def _predict_cluster(self, clusterer, embedding: np.ndarray):
        """
        Use approximate_predict to get (label, probability).
        Returns (label, prob).
        """
        if clusterer is None:
            return -1, 0.0
        point = embedding.reshape(1, -1)
        labels, probs = hdbscan.approximate_predict(clusterer, point)
        return int(labels[0]), float(probs[0])

    def _trigger_llm_summary(self, perspective: str, dev: str, cluster_id: int,
                             device_ips: list, cluster_info_entry: dict) -> dict:
        """
        Trigger an LLM re-summary for a cluster.  Logs timestamp, token usage,
        and elapsed time to SUMMARY_LOG (JSONL).

        Returns the summary log entry dict.
        """
        t_start = time.time()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Build prompt from perspective info and device IPs
        persp_desc = self.perspective_info.get(perspective, {}).get("prompt_info", perspective)

        prompt = (
            f"You are an expert in IoT device fingerprinting analysis.\n"
            f"Perspective: {perspective} ({persp_desc})\n"
            f"Device type: {dev}\n"
            f"Cluster ID: {cluster_id}\n"
            f"Number of devices in cluster: {len(device_ips)}\n"
            f"Device IPs: {', '.join(device_ips[:50])}\n\n"
            f"Please provide a concise summary of the common patterns and "
            f"distribution characteristics of these devices in JSON format."
        )

        messages = [{"role": "user", "content": prompt}]

        token_usage = {}
        summary_text = ""
        try:
            summary_text, token_usage = self.llm.chat_with_llm(
                SUMMARY_LLM, messages, whether_json=True, return_usage=True
            )
        except Exception as e:
            log.error("[Summary] LLM call failed for %s/%s/cluster %d: %s",
                      perspective, dev, cluster_id, e)
            summary_text = {"error": str(e)}
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        elapsed = time.time() - t_start

        log_entry = {
            "timestamp": timestamp,
            "perspective": perspective,
            "device_type": dev,
            "cluster_id": cluster_id,
            "device_count": len(device_ips),
            "elapsed_seconds": round(elapsed, 3),
            "token_usage": token_usage,
            "summary": summary_text if isinstance(summary_text, dict) else str(summary_text),
        }

        # Append to JSONL log
        with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        log.info("[Summary] %s/%s/cluster %d re-summarised in %.2fs, tokens=%s",
                 perspective, dev, cluster_id, elapsed, token_usage)

        return log_entry

    def _update_neo4j_cluster_link(self, ip: str, perspective: str, dev: str,
                                    cluster_id: int, is_new_cluster: bool):
        """
        Update Neo4j: create/link Cluster node to the new Device node.
        Set ``updated=False`` on the Cluster node.
        """
        if self.db is None:
            return

        cluster_id = int(cluster_id)  # ensure Python int for Neo4j

        if is_new_cluster:
            # Create a new Cluster node
            self.db.run(
                "MERGE (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                "SET c.hdbscan_label = -1, c.is_outlier = false, c.updated = false",
                cid=cluster_id, persp=perspective, dev=dev,
            )
        else:
            # Mark existing cluster as updated=false
            self.db.run(
                "MATCH (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                "SET c.updated = false",
                cid=cluster_id, persp=perspective, dev=dev,
            )

        # Link device → cluster (match by ip AND device_type to avoid cross-type linking)
        self.db.run(
            "MATCH (d:Device {ip: $ip, device_type: $dev}) "
            "MATCH (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
            "MERGE (d)-[:IN_COMMUNITY {perspective: $persp}]->(c)",
            ip=ip, cid=cluster_id, persp=perspective, dev=dev,
        )

    def _update_single_perspective(self, dev: str, ip: str, perspective: str,
                                    embedding: np.ndarray):
        """
        Process one (device, perspective) pair:
        predict cluster, handle new cluster / distance updates / LLM summary.
        """
        clusterer = _load_clusterer(perspective, dev)
        if clusterer is None:
            log.warning("[Update] No clusterer for %s/%s, skipping", perspective, dev)
            return

        label, prob = self._predict_cluster(clusterer, embedding)
        label = int(label)  # convert numpy.int64 → Python int for Neo4j compatibility
        cluster_info = _load_cluster_info(perspective)
        dev_info = cluster_info.get(dev, {})

        is_new_cluster = False
        assigned_cluster_id = label

        # Case 1: low confidence or noise → new cluster
        if label == -1 or prob < CONFIDENCE_THRESHOLD:
            assigned_cluster_id = self._get_next_cluster_id(perspective, dev)
            is_new_cluster = True

            # Record new cluster in cluster_info
            new_entry = {
                "center": embedding.tolist(),
                "max_distance": 0.0,
                "min_10_distance": -1,
                "node_count": 1,
            }
            dev_info[str(assigned_cluster_id)] = new_entry
            cluster_info[dev] = dev_info
            _save_cluster_info(perspective, cluster_info)

            log.info("[Update] %s/%s/%s: new cluster %d (prob=%.3f, label=%d)",
                     perspective, dev, ip, assigned_cluster_id, prob, label)

            self._update_neo4j_cluster_link(ip, perspective, dev, assigned_cluster_id, True)
            return

        # Case 2: assigned to existing cluster
        cluster_key = str(label)
        if cluster_key not in dev_info:
            log.warning("[Update] %s/%s: predicted label %d not in cluster_info, treating as new",
                        perspective, dev, label)
            assigned_cluster_id = self._get_next_cluster_id(perspective, dev)
            is_new_cluster = True
            dev_info[str(assigned_cluster_id)] = {
                "center": embedding.tolist(),
                "max_distance": 0.0,
                "min_10_distance": -1,
                "node_count": 1,
            }
            cluster_info[dev] = dev_info
            _save_cluster_info(perspective, cluster_info)
            self._update_neo4j_cluster_link(ip, perspective, dev, assigned_cluster_id, True)
            return

        entry = dev_info[cluster_key]
        center = np.array(entry["center"])
        dist = _compute_distance(embedding, center)

        # Update node count
        entry["node_count"] = entry.get("node_count", 0) + 1

        # Track incrementally added device for _get_cluster_device_ips / _get_cluster_points
        entry.setdefault("added_ips", []).append(ip)
        entry.setdefault("added_embeddings", []).append(embedding.tolist())

        # Check min_10_distance → trigger LLM re-summary
        center_recomputed = False
        if entry["min_10_distance"] != -1 and dist < entry["min_10_distance"]:
            log.info("[Update] %s/%s/%s: dist=%.4f < min_10=%.4f, triggering re-summary",
                     perspective, dev, ip, dist, entry["min_10_distance"])

            # Get all device IPs in this cluster from Neo4j (or from PCA CSV)
            device_ips = self._get_cluster_device_ips(perspective, dev, label)

            self._trigger_llm_summary(perspective, dev, label, device_ips, entry)

            # Update min_10_distance: recompute from all points including the new one
            all_points = self._get_cluster_points(perspective, dev, label)
            if all_points is not None and len(all_points) > 0:
                all_points = np.vstack([all_points, embedding.reshape(1, -1)])
                stats = _recompute_cluster_stats(all_points)
                entry["min_10_distance"] = stats["min_10_distance"]
                entry["center"] = stats["center"]
                entry["max_distance"] = stats["max_distance"]
                center_recomputed = True
            else:
                # Fallback: just update min_10 to the new distance
                entry["min_10_distance"] = dist

        # Check max_distance → update
        elif dist > entry["max_distance"]:
            entry["max_distance"] = dist
            log.info("[Update] %s/%s/%s: dist=%.4f > max=%.4f, updated max_distance",
                     perspective, dev, ip, dist, entry["max_distance"])

        # Update center incrementally (skip if already recomputed above)
        if not center_recomputed:
            old_count = entry["node_count"] - 1
            if old_count > 0:
                new_center = (np.array(entry["center"]) * old_count + embedding) / entry["node_count"]
                entry["center"] = new_center.tolist()

        dev_info[cluster_key] = entry
        cluster_info[dev] = dev_info
        _save_cluster_info(perspective, cluster_info)

        self._update_neo4j_cluster_link(ip, perspective, dev, assigned_cluster_id, False)

        log.info("[Update] %s/%s/%s → cluster %d (prob=%.3f, dist=%.4f)",
                 perspective, dev, ip, assigned_cluster_id, prob, dist)

    def _get_cluster_device_ips(self, perspective: str, dev: str, cluster_label: int) -> list:
        """Retrieve all IPs belonging to a cluster from the PCA CSV file
        and the incremental device log."""
        ips = []
        if perspective == "comprehensive":
            csv_path = os.path.join(OVERALL_DIR, f"ipraw_{dev}_embedding_overall_pca.csv")
        else:
            csv_path = os.path.join(SINGLE_DIR, f"embedding_{perspective}",
                                    f"ipraw_{dev}_embedding_{perspective}_pca.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["ip", "cluster"])
            ips = df[df["cluster"] == cluster_label]["ip"].astype(str).tolist()

        # Also include incrementally added devices tracked in cluster_info
        info = _load_cluster_info(perspective)
        dev_info = info.get(dev, {})
        entry = dev_info.get(str(cluster_label), {})
        if entry and "added_ips" in entry:
            ips.extend(entry["added_ips"])
        return ips

    def _get_cluster_points(self, perspective: str, dev: str, cluster_label: int):
        """Retrieve all PCA embedding points for a cluster from the PCA CSV
        and the incremental device log."""
        if perspective == "comprehensive":
            csv_path = os.path.join(OVERALL_DIR, f"ipraw_{dev}_embedding_overall_pca.csv")
        else:
            csv_path = os.path.join(SINGLE_DIR, f"embedding_{perspective}",
                                    f"ipraw_{dev}_embedding_{perspective}_pca.csv")

        points = None
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if perspective == "comprehensive":
                feature_cols = [c for c in df.columns if c not in ("ip", "cluster")]
            else:
                feature_cols = [c for c in df.columns if c.startswith("pca")]

            sub = df[df["cluster"] == cluster_label]
            if not sub.empty:
                points = sub[feature_cols].values

        # Also include incrementally added device embeddings from cluster_info
        info = _load_cluster_info(perspective)
        dev_info = info.get(dev, {})
        entry = dev_info.get(str(cluster_label), {})
        if entry and "added_embeddings" in entry:
            added = np.array(entry["added_embeddings"])
            if points is not None:
                points = np.vstack([points, added])
            else:
                points = added
        return points

    def _update_comprehensive(self, dev: str, ip: str, overall_embedding: np.ndarray):
        """Process the comprehensive-perspective cluster update."""
        self._update_single_perspective(dev, ip, "comprehensive", overall_embedding)

    # ── Step 3: Partial re-clustering ────────────────────────────────

    def _get_stale_clusters(self, perspective: str) -> list:
        """
        Query Neo4j for all Cluster nodes with ``updated=False`` for the given
        perspective.  Returns list of dicts: {cluster_id, device_type, hdbscan_label}.
        """
        if self.db is None:
            return []

        results = self.db.run(
            "MATCH (c:Cluster {perspective: $persp, updated: false}) "
            "RETURN c.cluster_id AS cid, c.device_type AS dev, c.hdbscan_label AS hlabel",
            persp=perspective,
        ).data()
        return results

    def _get_all_cluster_ids(self, perspective: str) -> set:
        """Get all cluster IDs currently in use for a perspective (from Neo4j)."""
        if self.db is None:
            return set()
        results = self.db.run(
            "MATCH (c:Cluster {perspective: $persp}) RETURN DISTINCT c.cluster_id AS cid",
            persp=perspective,
        ).data()
        return set(r["cid"] for r in results)

    def _find_overlapping_clusters(self, perspective: str, stale_clusters: list,
                                    cluster_info: dict) -> list:
        """
        Given stale clusters, find other clusters whose max_distance circles
        overlap (centre_distance ≤ max_distance₁ + max_distance₂).
        Returns the full set of cluster IDs to include in re-clustering.
        """
        stale_ids = set()
        stale_data = []  # (dev, cluster_id, center, max_distance)

        for sc in stale_clusters:
            dev = sc["dev"]
            cid = sc["cid"]
            info = cluster_info.get(dev, {})
            entry = info.get(str(sc["hlabel"]), None)
            if entry is None:
                # Try by cluster_id directly
                for k, v in info.items():
                    if int(k) == cid:
                        entry = v
                        break
            if entry is None:
                continue
            stale_ids.add((dev, cid))
            stale_data.append((dev, cid, np.array(entry["center"]), entry["max_distance"]))

        # Find overlapping clusters
        all_include = set(stale_ids)
        for dev in cluster_info:
            dev_info = cluster_info[dev]
            for ckey, centry in dev_info.items():
                cid = int(ckey)
                if (dev, cid) in stale_ids:
                    continue
                center2 = np.array(centry["center"])
                max2 = centry["max_distance"]

                for sdev, scid, center1, max1 in stale_data:
                    if sdev != dev:
                        continue
                    d = _compute_distance(center1, center2)
                    if d <= max1 + max2:
                        all_include.add((dev, cid))
                        break

        return all_include

    def _partial_recluster(self, perspective: str):
        """
        Perform partial re-clustering for a single perspective:
        1. Find all stale (updated=False) clusters.
        2. Find overlapping clusters.
        3. Collect all device nodes in the subset.
        4. Re-run HDBSCAN.
        5. Assign new cluster IDs that don't collide with non-participating IDs.
        6. Update Neo4j and cluster_info.

        Key mapping: Neo4j Cluster nodes use a global ``cluster_id`` counter
        (assigned in build_neo4j.py), while ``cluster_info.json`` keys and the
        PCA CSV ``cluster`` column use per-device HDBSCAN labels (0-based).
        The ``stale_clusters`` from Neo4j carry both ``cid`` (Neo4j cluster_id)
        and ``hlabel`` (HDBSCAN label).  We use ``hlabel`` to filter the CSV
        and ``cid`` to delete old Neo4j nodes.
        """
        log.info("[Recluster] Starting partial re-clustering for perspective '%s'", perspective)

        stale_clusters = self._get_stale_clusters(perspective)
        if not stale_clusters:
            log.info("[Recluster] No stale clusters for '%s', skipping", perspective)
            return

        cluster_info = _load_cluster_info(perspective)
        subset = self._find_overlapping_clusters(perspective, stale_clusters, cluster_info)

        if not subset:
            log.info("[Recluster] Empty subset for '%s', skipping", perspective)
            return

        log.info("[Recluster] %s: %d stale clusters, %d total clusters in subset",
                 perspective, len(stale_clusters), len(subset))

        # Build a lookup from stale_clusters: (dev, neo4j_cid) → hdbscan_label
        stale_lookup = {}
        for sc in stale_clusters:
            stale_lookup[(sc["dev"], sc["cid"])] = sc["hlabel"]

        # Group subset by device type: dev → set of Neo4j cluster_ids
        dev_clusters = {}
        for dev, cid in subset:
            dev_clusters.setdefault(dev, set()).add(cid)

        # Determine the starting new cluster_id (avoid collision with all existing Neo4j IDs)
        all_used_ids = self._get_all_cluster_ids(perspective)
        max_existing_id = max(all_used_ids) if all_used_ids else -1
        next_id = max_existing_id + 1

        for dev, cid_set in dev_clusters.items():
            # Load PCA CSV for this device
            if perspective == "comprehensive":
                csv_path = os.path.join(OVERALL_DIR, f"ipraw_{dev}_embedding_overall_pca.csv")
            else:
                csv_path = os.path.join(SINGLE_DIR, f"embedding_{perspective}",
                                        f"ipraw_{dev}_embedding_{perspective}_pca.csv")

            if not os.path.exists(csv_path):
                log.warning("[Recluster] CSV not found: %s", csv_path)
                continue

            df = pd.read_csv(csv_path)
            if perspective == "comprehensive":
                feat_cols = [c for c in df.columns if c not in ("ip", "cluster")]
            else:
                feat_cols = [c for c in df.columns if c.startswith("pca")]

            # Map Neo4j cluster_ids in cid_set → HDBSCAN labels for CSV filtering
            dev_info = cluster_info.get(dev, {})
            hlabels_to_include = set()
            for cid in cid_set:
                # From stale_clusters lookup
                hlabel = stale_lookup.get((dev, cid))
                if hlabel is not None:
                    hlabels_to_include.add(hlabel)
                # Also check cluster_info: if a key matches the cid numerically,
                # it may be a newly assigned cluster_id (from incremental update)
                # that was used directly as both Neo4j cluster_id and CSV label
                if str(cid) in dev_info:
                    hlabels_to_include.add(cid)

            sub_df = df[df["cluster"].isin(hlabels_to_include)]
            if sub_df.empty:
                log.warning("[Recluster] No devices found for %s/%s subset", perspective, dev)
                continue

            X = sub_df[feat_cols].values
            log.info("[Recluster] %s/%s: re-clustering %d devices",
                     perspective, dev, len(X))

            # Re-run HDBSCAN
            new_clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                             prediction_data=True)
            new_labels = new_clusterer.fit_predict(X)

            if (new_labels == -1).all():
                new_labels[:] = 0

            # Assign new cluster IDs starting from next_id, avoiding collisions
            new_id_map = {}  # new hdbscan label → new cluster_id
            for hl in sorted(set(new_labels)):
                hl = int(hl)
                if hl == -1:
                    continue
                new_id_map[hl] = int(next_id)
                next_id += 1

            # Update the CSV with new cluster labels
            sub_df = sub_df.copy()
            sub_df["cluster"] = [new_id_map.get(l, -1) for l in new_labels]

            # Update the original DataFrame
            df.loc[sub_df.index, "cluster"] = sub_df["cluster"]
            df.to_csv(csv_path, index=False)

            # Save new clusterer
            if perspective == "comprehensive":
                clusterer_path = os.path.join(OVERALL_DIR, f"clusterer_{dev}.joblib")
            else:
                clusterer_path = os.path.join(SINGLE_DIR, f"embedding_{perspective}",
                                               f"clusterer_{dev}.joblib")
            joblib.dump(new_clusterer, clusterer_path)

            # Update cluster_info for this device
            new_dev_info = {}
            for hl, cid in new_id_map.items():
                mask = new_labels == hl
                points = X[mask]
                stats = _recompute_cluster_stats(points)
                new_dev_info[str(cid)] = stats

            # Handle noise points (label == -1) as singletons
            noise_mask = new_labels == -1
            noise_ip_list = []
            if noise_mask.any():
                noise_indices = np.where(noise_mask)[0]
                for idx_in_sub in noise_indices:
                    noise_ip = str(sub_df.iloc[idx_in_sub]["ip"])
                    cid = next_id
                    next_id += 1
                    point = X[idx_in_sub]
                    new_dev_info[str(cid)] = {
                        "center": point.tolist(),
                        "max_distance": 0.0,
                        "min_10_distance": -1,
                        "node_count": 1,
                    }
                    noise_ip_list.append((noise_ip, cid))

            # Merge: remove old entries for participating clusters, add new ones
            old_keys_to_remove = set()
            for ckey in dev_info:
                ckey_int = int(ckey)
                # Remove if this HDBSCAN label is in the subset
                if ckey_int in hlabels_to_include:
                    old_keys_to_remove.add(ckey)
            for k in old_keys_to_remove:
                dev_info.pop(k, None)
            dev_info.update(new_dev_info)
            cluster_info[dev] = dev_info

            # Update Neo4j: remove old cluster nodes for subset, create new ones
            if self.db is not None:
                # Delete old IN_COMMUNITY edges and Cluster nodes for the subset
                for cid in cid_set:
                    cid = int(cid)
                    self.db.run(
                        "MATCH (d:Device)-[r:IN_COMMUNITY]->(c:Cluster "
                        "{cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                        "DELETE r",
                        cid=cid, persp=perspective, dev=dev,
                    )
                    self.db.run(
                        "MATCH (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                        "DETACH DELETE c",
                        cid=cid, persp=perspective, dev=dev,
                    )

                # Create new Cluster nodes and edges for valid clusters
                for hl, cid in new_id_map.items():
                    self.db.run(
                        "MERGE (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                        "SET c.hdbscan_label = $hl, c.is_outlier = false, c.updated = true",
                        cid=int(cid), persp=perspective, dev=dev, hl=int(hl),
                    )
                    member_ips = sub_df[new_labels == hl]["ip"].astype(str).tolist()
                    for mip in member_ips:
                        self.db.run(
                            "MATCH (d:Device {ip: $ip}) "
                            "MATCH (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                            "MERGE (d)-[:IN_COMMUNITY {perspective: $persp}]->(c)",
                            ip=mip, cid=cid, persp=perspective, dev=dev,
                        )

                # Create noise singleton Cluster nodes and edges in Neo4j
                for noise_ip, cid in noise_ip_list:
                    self.db.run(
                        "MERGE (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                        "SET c.hdbscan_label = -1, c.is_outlier = true, c.updated = true",
                        cid=int(cid), persp=perspective, dev=dev,
                    )
                    self.db.run(
                        "MATCH (d:Device {ip: $ip}) "
                        "MATCH (c:Cluster {cluster_id: $cid, perspective: $persp, device_type: $dev}) "
                        "MERGE (d)-[:IN_COMMUNITY {perspective: $persp}]->(c)",
                        ip=noise_ip, cid=int(cid), persp=perspective, dev=dev,
                    )

            log.info("[Recluster] %s/%s: %d new clusters + %d noise singletons, IDs %s",
                     perspective, dev, len(new_id_map), len(noise_ip_list),
                     list(new_id_map.values()))

        _save_cluster_info(perspective, cluster_info)
        log.info("[Recluster] Perspective '%s' re-clustering complete", perspective)

    def _check_and_recluster(self):
        """Check if the global counter exceeds threshold and trigger re-clustering."""
        if self.update_counter < RECLUSTER_THRESHOLD:
            return

        log.info("[Recluster] Update counter %d ≥ threshold %d, triggering partial re-clustering",
                 self.update_counter, RECLUSTER_THRESHOLD)

        # Re-cluster all perspectives including comprehensive
        all_perspectives = list(self.perspective_names) + ["comprehensive"]
        for perspective in all_perspectives:
            self._partial_recluster(perspective)

        # Reset counter
        self.update_counter = 0
        self._save_counter()

        # Increment re-cluster event counter
        self.recluster_count += 1
        self._save_recluster_count()
        log.info("[Recluster] Counter reset to 0, total re-cluster events: %d", self.recluster_count)

    # ── Public API ───────────────────────────────────────────────────

    def update_devices(self, devices: list):
        """
        Incrementally update the graph with a batch of new devices.

        Each device dict must contain:
            - ip: str
            - device_type: str (e.g. "ALARM", "CONTROLLER")
            - features: dict of {feature_name: value} for Layer-1
            - embeddings: dict of {perspective_name: [float, ...]} PCA embeddings
              (must include "comprehensive" for the overall perspective)
        """
        for dev_info in devices:
            ip = dev_info["ip"]
            dev_type = dev_info["device_type"]
            features = dev_info.get("features", {})
            embeddings = dev_info.get("embeddings", {})

            log.info("=== Updating device %s (%s) ===", ip, dev_type)

            # Step 1: Insert into Layer-1 entity graph
            vendor = dev_info.get("vendor", "Unknown")
            self._insert_device_node(ip, dev_type, features, vendor)

            # Step 2: Update each single-perspective cluster
            for perspective in self.perspective_names:
                emb = embeddings.get(perspective)
                if emb is None:
                    log.warning("[Update] No embedding for %s/%s, skipping", perspective, ip)
                    continue
                emb_arr = np.array(emb, dtype=np.float64)
                self._update_single_perspective(dev_type, ip, perspective, emb_arr)

            # Step 2b: Update comprehensive-perspective cluster
            overall_emb = embeddings.get("comprehensive")
            if overall_emb is not None:
                self._update_comprehensive(dev_type, ip, np.array(overall_emb, dtype=np.float64))
            else:
                log.warning("[Update] No comprehensive embedding for %s, skipping", ip)

            # Increment global counter
            self.update_counter += 1
            self._save_counter()

            # Step 3: Check if partial re-clustering is needed
            self._check_and_recluster()

        log.info("Batch update complete. Counter: %d/%d", self.update_counter, RECLUSTER_THRESHOLD)

    # ── Test-update entry point ─────────────────────────────────────

    def run_test_update(self, devices=None, neo4j_url=NEO4J_UPDATED_URL,
                        neo4j_user="neo4j", neo4j_pass="avs01046"):
        """
        Run graph update for test set devices on neo4j_updated (port 7688).
        Collects metrics: update frequency, Layer-1 time, cluster summary time + LLM cost,
        re-clustering events, total update time.

        For ALARM/CONTROLLER (which have saved HDBSCAN clusterers),
          use nearest-neighbor PCA embeddings from RAG data.
        For other device types, no clusterers exist so only Layer-1 is updated.

        All results are saved to graph/update/.

        :param devices: list of device types to run (default: all 11)
        :param neo4j_url: Neo4j connection URL
        :param neo4j_user: Neo4j username
        :param neo4j_pass: Neo4j password
        """
        from py2neo import Graph as Py2neoGraph

        if devices is None:
            devices = ALL_TEST_DEVICES

        log.info("=" * 70)
        log.info("Graph Update for Test Set Devices on neo4j_updated")
        log.info("=" * 70)

        features_list = _load_features_list()
        log.info(f"Features: {len(features_list)}")

        # Connect to neo4j_updated
        g_updated = Py2neoGraph(neo4j_url, auth=(neo4j_user, neo4j_pass))
        before_stats = _get_db_stats(g_updated)
        log.info(f"BEFORE: {before_stats}")

        # Reuse the existing db connection if it points to the same URL,
        # otherwise create a new updater with the target DB
        self.db = g_updated

        # Clear all state files
        for f in [COUNTER_FILE, NEXT_ID_FILE, RECLUSTER_COUNT_FILE, SUMMARY_LOG]:
            if os.path.exists(f):
                os.remove(f)
        self.update_counter = 0
        self.recluster_count = 0
        self.next_cluster_ids = {}

        # Metrics collection
        metrics = {}
        total_t0 = time.time()

        for dev in devices:
            log.info(f"\n{'='*60}")
            log.info(f"Processing {dev}")
            log.info(f"{'='*60}")

            t_dev_start = time.time()
            recluster_before = self.recluster_count

            # Prepare test devices
            test_devices = _prepare_test_devices(dev, features_list)
            if not test_devices:
                log.warning(f"No test devices for {dev}")
                continue

            n_devices = len(test_devices)
            log.info(f"[{dev}] {n_devices} test devices to update")

            # Add embeddings for devices with clusterers
            if dev in DEVICES_WITH_CLUSTERERS:
                t_emb_start = time.time()
                _add_embeddings_nearest_neighbor(test_devices, dev, features_list)
                t_emb = time.time() - t_emb_start
                log.info(f"[{dev}] Embedding preparation: {t_emb:.1f}s")
            else:
                t_emb = 0
                log.info(f"[{dev}] No clusterers available, only Layer-1 update")

            # Run update
            t_update_start = time.time()

            BATCH_SIZE = 500
            n_summaries_before = 0
            if os.path.exists(SUMMARY_LOG):
                with open(SUMMARY_LOG) as f:
                    n_summaries_before = sum(1 for _ in f)

            for i in range(0, n_devices, BATCH_SIZE):
                batch = test_devices[i:i + BATCH_SIZE]
                self.update_devices(batch)
                log.info(f"[{dev}] Progress: {min(i + BATCH_SIZE, n_devices)}/{n_devices}")

            t_update = time.time() - t_update_start
            t_dev_total = time.time() - t_dev_start

            # Count summaries for this device
            n_summaries_after = 0
            if os.path.exists(SUMMARY_LOG):
                with open(SUMMARY_LOG) as f:
                    n_summaries_after = sum(1 for _ in f)
            n_summaries = n_summaries_after - n_summaries_before

            # Count re-clustering events triggered during this device's update
            n_reclusters = self.recluster_count - recluster_before

            metrics[dev] = {
                "n_devices": n_devices,
                "n_updates": n_devices,
                "n_summaries": n_summaries,
                "n_reclusters": n_reclusters,
                "embedding_time_s": t_emb,
                "update_time_s": t_update,
                "total_time_s": t_dev_total,
            }

            log.info(f"[{dev}] Done: {n_devices} devices, {n_summaries} summaries, {n_reclusters} reclusters, {t_dev_total:.1f}s")

        total_time = time.time() - total_t0

        # Collect LLM token usage from summary log
        llm_stats = {}
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0

        if os.path.exists(SUMMARY_LOG):
            with open(SUMMARY_LOG) as f:
                for line in f:
                    entry = json.loads(line)
                    dev = entry.get("device_type", "unknown")
                    usage = entry.get("token_usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)

                    pricing = LLM_PRICING.get(SUMMARY_LLM, {"input": 0, "output": 0})
                    cost = (tokens_in / 1_000_000 * pricing["input"]) + (tokens_out / 1_000_000 * pricing["output"])

                    if dev not in llm_stats:
                        llm_stats[dev] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
                    llm_stats[dev]["calls"] += 1
                    llm_stats[dev]["tokens_in"] += tokens_in
                    llm_stats[dev]["tokens_out"] += tokens_out
                    llm_stats[dev]["cost_usd"] += cost

                    total_tokens_in += tokens_in
                    total_tokens_out += tokens_out
                    total_cost += cost

        # Total re-clustering events across all devices
        recluster_count = self.recluster_count

        # Get AFTER stats
        after_stats = _get_db_stats(g_updated)

        # ─── Print Results ─────────────────────────────────────────────────
        print(f"\n{'='*80}")
        print("Graph Update Results — Test Set on neo4j_updated")
        print(f"{'='*80}\n")

        # Table 1: Timing
        print("─" * 90)
        print("Table 1: Timing Statistics (seconds)")
        print("─" * 90)
        print(f"{'Device':<25} {'Devices':>8} {'Updates':>8} {'Summaries':>10} {'Reclusters':>11} {'Emb(s)':>8} {'Update(s)':>10} {'Total(s)':>10}")
        for dev in devices:
            m = metrics.get(dev, {})
            print(f"{dev:<25} {m.get('n_devices', 0):>8} {m.get('n_updates', 0):>8} "
                  f"{m.get('n_summaries', 0):>10} {m.get('n_reclusters', 0):>11} "
                  f"{m.get('embedding_time_s', 0):>8.1f} {m.get('update_time_s', 0):>10.1f} "
                  f"{m.get('total_time_s', 0):>10.1f}")
        print(f"{'─'*90}")
        total_devices = sum(m.get("n_devices", 0) for m in metrics.values())
        total_updates = sum(m.get("n_updates", 0) for m in metrics.values())
        total_summaries = sum(m.get("n_summaries", 0) for m in metrics.values())
        total_emb = sum(m.get("embedding_time_s", 0) for m in metrics.values())
        total_update = sum(m.get("update_time_s", 0) for m in metrics.values())
        print(f"{'TOTAL':<25} {total_devices:>8} {total_updates:>8} {total_summaries:>10} {recluster_count:>11} "
              f"{total_emb:>8.1f} {total_update:>10.1f} {total_time:>10.1f}")

        # Table 2: LLM Cost
        print(f"\n{'─'*80}")
        print(f"Table 2: LLM API Token & Cost (LLM={SUMMARY_LLM})")
        print("─" * 80)
        print(f"{'Device':<25} {'Calls':>8} {'Tokens In':>12} {'Tokens Out':>12} {'Total Tok':>12} {'Cost(USD)':>12}")
        for dev in devices:
            s = llm_stats.get(dev, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
            total_tok = s["tokens_in"] + s["tokens_out"]
            print(f"{dev:<25} {s['calls']:>8} {s['tokens_in']:>12} {s['tokens_out']:>12} {total_tok:>12} {s['cost_usd']:>12.4f}")
        print(f"{'─'*80}")
        total_calls = sum(s["calls"] for s in llm_stats.values())
        total_tok_all = total_tokens_in + total_tokens_out
        print(f"{'TOTAL':<25} {total_calls:>8} {total_tokens_in:>12} {total_tokens_out:>12} {total_tok_all:>12} {total_cost:>12.4f}")

        # Table 3: Neo4j Database Changes
        print(f"\n{'─'*80}")
        print("Table 3: Neo4j Database Changes (neo4j_updated)")
        print("─" * 80)
        print(f"{'Label':<16} {'Before':>14} {'After':>14} {'Delta':>14}")
        all_labels = sorted(set(list(before_stats.keys()) + list(after_stats.keys())))
        for label in all_labels:
            if label.startswith("_"):
                continue
            b = before_stats.get(label, 0)
            a = after_stats.get(label, 0)
            d = a - b
            print(f"{label:<16} {b:>14,} {a:>14,} {d:>+14,}")
        b_rel = before_stats.get("_relationships", 0)
        a_rel = after_stats.get("_relationships", 0)
        print(f"{'Relationships':<16} {b_rel:>14,} {a_rel:>14,} {a_rel - b_rel:>+14,}")

        # Table 4: Layer-1 vs Cluster Update breakdown (only for devices with clusterers)
        clusterer_devs = [d for d in devices if d in DEVICES_WITH_CLUSTERERS]
        if clusterer_devs:
            print(f"\n{'─'*80}")
            print("Table 4: Layer-1 Insertion vs Cluster Update Time (clusterer devices only)")
            print("─" * 80)
            print(f"{'Device':<25} {'L1+Cluster(s)':>14} {'Summary(s)':>12} {'Emb Prep(s)':>12} {'Total(s)':>10}")
            for dev in clusterer_devs:
                m = metrics.get(dev, {})
                summary_time = 0.0
                if os.path.exists(SUMMARY_LOG):
                    with open(SUMMARY_LOG) as f:
                        for line in f:
                            entry = json.loads(line)
                            if entry.get("device_type") == dev:
                                summary_time += entry.get("elapsed_seconds", 0)
                print(f"{dev:<25} {m.get('update_time_s', 0):>14.1f} {summary_time:>12.1f} "
                      f"{m.get('embedding_time_s', 0):>12.1f} {m.get('total_time_s', 0):>10.1f}")

        # Save report
        report = {
            "timing": {dev: metrics.get(dev, {}) for dev in devices},
            "llm_cost": {dev: llm_stats.get(dev, {}) for dev in devices},
            "llm_total": {"calls": total_calls, "tokens_in": total_tokens_in, "tokens_out": total_tokens_out, "cost_usd": total_cost},
            "neo4j_before": before_stats,
            "neo4j_after": after_stats,
            "total_time_s": total_time,
            "recluster_count": recluster_count,
        }
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental graph update")
    parser.add_argument("--devices", type=str, default="all",
                        help="Comma-separated list of device types to update, or 'all' (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without modifying Neo4j")
    args = parser.parse_args()

    if args.devices.strip().lower() == "all":
        dev_list = None
    else:
        dev_list = [d.strip() for d in args.devices.split(",") if d.strip()]

    updater = GraphUpdater()

    if args.dry_run:
        log.info("Dry run mode – no Neo4j modifications")
        updater.db = None

    updater.run_test_update(devices=dev_list)
