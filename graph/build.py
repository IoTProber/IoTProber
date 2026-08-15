import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
from api import ProtocolGraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


def feature_to_rel_type(feature_name: str) -> str:
    """Convert a feature name like 'as-asn' to a valid Cypher relationship type 'Has_as_asn'."""
    return "Has_" + feature_name.replace("-", "_").replace(".", "_")


class HierarchicalGraph:
    def __init__(self, devices=None):
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.data_path = os.path.join(self.base_path, "platform_data", "csv", "local", "1")
        self.community_path = os.path.join(self.data_path, "community", "single")
        self.overall_path = os.path.join(self.data_path, "community", "embedding_overall")
        self.graph = ProtocolGraph("neo4j://localhost:7687", "neo4j", "avs01046")

        self.fingerprint_features = self._load_fingerprint_features()
        self.device_labels = self._discover_device_labels(devices=devices)
        self.perspectives = self._discover_perspectives()

        # Global, monotonically-increasing cluster-ID counters. Each single
        # perspective keeps its own counter and the comprehensive view keeps a
        # separate one (paper §4.4: "不同perspective和comprehensive分别维护计数器").
        # Both valid clusters and per-device outlier (cluster == -1) singletons
        # draw their unique IDs from the same per-key counter.
        self._cluster_counters = {}

    def _next_cluster_id(self, key: str) -> int:
        """Return the next globally-unique cluster id for the given counter key
        (a single-perspective name, or 'comprehensive'), incrementing it."""
        cid = self._cluster_counters.get(key, 0)
        self._cluster_counters[key] = cid + 1
        return cid

    def _load_fingerprint_features(self):
        features_path = os.path.join(self.base_path, "local_used_feature.txt")
        with open(features_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        log.info(f"Loaded {len(features)} fingerprint features: {features}")
        return features

    def _discover_device_labels(self, devices=None):
        device_labels = []
        for fname in sorted(os.listdir(self.data_path)):
            if fname.startswith("ipraw_") and fname.endswith(".csv"):
                dev = fname[len("ipraw_"):-len(".csv")]
                device_labels.append(dev)
        if devices:
            device_set = set(devices)
            device_labels = [d for d in device_labels if d in device_set]
            log.info(f"Device filter applied: {device_labels}")
        log.info(f"Discovered {len(device_labels)} device types: {device_labels}")
        return device_labels

    def _discover_perspectives(self):
        perspectives = []
        if os.path.isdir(self.community_path):
            for pname in sorted(os.listdir(self.community_path)):
                pdir = os.path.join(self.community_path, pname)
                if os.path.isdir(pdir) and pname.startswith("embedding_"):
                    perspectives.append(pname)
        log.info(f"Discovered {len(perspectives)} perspectives: {perspectives}")
        return perspectives

    # -------------------------------------------------------------------------
    # Layer 1: Device Entity Graph  (O(N * k) construction)
    # -------------------------------------------------------------------------

    def build_layer1_device(self, dev: str):
        """
        Reads ipraw_{dev}.csv and constructs the first-layer entity graph:
          - Device node: {ip, device_type}
          - Feature node per unique (feature_name, value) pair: {feature_name, value}
          - Has_<feature> undirected edge between each Device and its Feature nodes

        Two devices sharing the same feature value will share the same Feature node,
        creating the path (A:Device)-[:Has_f]-(f:Feature)-[:Has_f]-(B:Device).
        Total complexity: O(N * k), where k = 25 fingerprint features.
        """
        csv_path = os.path.join(self.data_path, f"ipraw_{dev}.csv")
        if not os.path.exists(csv_path):
            log.warning(f"[Layer 1] CSV not found: {csv_path}, skipping.")
            return

        header_df = pd.read_csv(csv_path, nrows=0)
        available_features = [f for f in self.fingerprint_features if f in header_df.columns]
        missing = set(self.fingerprint_features) - set(available_features)
        if missing:
            log.warning(f"[Layer 1] {dev}: features not found in CSV columns: {missing}")

        use_cols = ["ip"] + available_features
        df = pd.read_csv(csv_path, usecols=use_cols)
        df[available_features] = df[available_features].fillna("").astype(str)
        df["ip"] = df["ip"].astype(str)

        total = len(df)
        log.info(f"[Layer 1] {dev}: {total} devices, {len(available_features)} features")

        for i, row in df.iterrows():
            ip = row["ip"].strip()

            device_node, _ = self.graph.CreateNode(["Device"], {
                "ip": ip,
                "device_type": dev
            })

            if device_node is None:
                log.warning(f"[Layer 1] Failed to create Device node for IP {ip}, skipping.")
                continue

            for feat in available_features:
                value = row[feat].strip()

                feature_node, _ = self.graph.CreateNode(["Feature"], {
                    "feature_name": feat,
                    "value": value
                })

                if feature_node is None:
                    log.warning(f"[Layer 1] Failed to create Feature node {feat}={value!r}, skipping.")
                    continue

                rel_type = feature_to_rel_type(feat)
                self.graph.CreateRelationship(device_node, feature_node, rel_type, {})

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                log.info(f"[Layer 1] {dev}: {i + 1}/{total} devices processed")

        log.info(f"[Layer 1] {dev} done!")

    # -------------------------------------------------------------------------
    # Shared summary-cluster materialization (paper §4.4)
    # -------------------------------------------------------------------------

    def _materialize_clusters(self, df, perspective_attr: str, counter_key: str,
                              device_type: str, layer_tag: str):
        """
        Turn one device-type's HDBSCAN result into "Cluster" summary nodes and
        Device→Cluster IN_COMMUNITY edges.

        df                : DataFrame with columns ['ip', 'cluster'] (HDBSCAN label).
        perspective_attr  : value stored on the Cluster.perspective attribute,
                            i.e. a single-perspective name or 'comprehensive'.
        counter_key       : key used to draw globally-unique cluster IDs.
        device_type       : device type of this batch (Cluster.device_type).
        layer_tag         : log prefix, e.g. 'Layer 2' / 'Layer 3'.

        Valid clusters (label >= 0) become one Cluster node each; every outlier
        (label == -1) device becomes its own singleton Cluster node. All cluster
        IDs come from the same per-key global counter.
        """
        df_valid = df[df["cluster"] != -1]
        df_outlier = df[df["cluster"] == -1]

        n_valid = df_valid["cluster"].nunique()
        n_outlier = len(df_outlier)
        log.info(f"[{layer_tag}] {perspective_attr}/{device_type}: "
                 f"{len(df_valid)} devices in {n_valid} clusters, "
                 f"{n_outlier} outliers → singleton clusters")

        edges = 0

        # --- Valid clusters: one Cluster node per HDBSCAN label -------------
        for hdbscan_label in sorted(df_valid["cluster"].unique()):
            hlabel = int(hdbscan_label)
            cid = self._next_cluster_id(counter_key)
            cluster_node, _ = self.graph.CreateNode(["Cluster"], {
                "cluster_id": cid,
                "perspective": perspective_attr,
                "device_type": device_type,
                "hdbscan_label": hlabel,
                "is_outlier": False,
            })

            members = df_valid[df_valid["cluster"] == hlabel]["ip"]
            for ip in members:
                if self._link_device_to_cluster(ip.strip(), cluster_node, perspective_attr):
                    edges += 1

        # --- Outliers: one singleton Cluster node per device ---------------
        for ip in df_outlier["ip"]:
            cid = self._next_cluster_id(counter_key)
            cluster_node, _ = self.graph.CreateNode(["Cluster"], {
                "cluster_id": cid,
                "perspective": perspective_attr,
                "device_type": device_type,
                "hdbscan_label": -1,
                "is_outlier": True,
            })
            if self._link_device_to_cluster(ip.strip(), cluster_node, perspective_attr):
                edges += 1

        log.info(f"[{layer_tag}] {perspective_attr}/{device_type} done! ({edges} edges created)")

    def _link_device_to_cluster(self, ip: str, cluster_node, perspective_attr: str) -> bool:
        """Create a Device→Cluster IN_COMMUNITY edge. Returns True on success."""
        if cluster_node is None:
            return False
        device_node = self.graph.MatchSingleNode(["Device"], {"ip": ip})
        if device_node is None:
            log.warning(f"[Cluster] Device node not found for IP {ip}, skipping edge.")
            return False
        self.graph.CreateRelationship(
            device_node, cluster_node, "IN_COMMUNITY",
            {"perspective": perspective_attr}
        )
        return True

    # -------------------------------------------------------------------------
    # Layer 2: Community Graph
    # -------------------------------------------------------------------------

    def build_layer2_community(self, perspective: str):
        """
        Reads the HDBSCAN clustering results (PCA CSV) for each device type under
        the given perspective directory and constructs the second-layer summary
        clusters (paper §4.4 "Summary Clusters Generation"):

          - Summary entity node of type "Cluster" per cluster, carrying
            {cluster_id, perspective, device_type, hdbscan_label, is_outlier}.
            The `perspective` attribute is the single-perspective name (e.g. 'as'),
            which is how single-perspective clusters are distinguished from the
            comprehensive-view clusters (perspective == 'comprehensive').
          - IN_COMMUNITY edge from every first-layer Device node to the Cluster
            node(s) it belongs to. A device is assigned to multiple
            single-perspective clusters (one per perspective).

        Cluster IDs are globally unique and monotonically increasing within this
        perspective (shared across device types via self._next_cluster_id).

        Outliers (HDBSCAN label == -1) are NOT discarded: each outlier device is
        turned into its own singleton "Cluster" node with a unique cluster_id, so
        rare-vendor devices are preserved for downstream LLM analysis.

        Clusters are split by device type: because clustering runs per device
        type, every (device_type, hdbscan_label) pair is already a distinct
        Cluster node.

        perspective: subdirectory name under community/single, e.g. 'embedding_as'
        """
        perspective_name = perspective[len("embedding_"):]
        perspective_dir = os.path.join(self.community_path, perspective)

        log.info(f"[Layer 2] Building summary Cluster nodes for perspective: {perspective_name}")

        for dev in self.device_labels:
            pca_csv = os.path.join(
                perspective_dir,
                f"ipraw_{dev}_embedding_{perspective_name}_pca.csv"
            )
            if not os.path.exists(pca_csv):
                log.warning(f"[Layer 2] PCA CSV not found: {pca_csv}, skipping.")
                continue

            df = pd.read_csv(pca_csv, usecols=["ip", "cluster"])
            df["ip"] = df["ip"].astype(str)
            df["cluster"] = df["cluster"].astype(int)

            self._materialize_clusters(
                df=df,
                perspective_attr=perspective_name,
                counter_key=perspective_name,
                device_type=dev,
                layer_tag="Layer 2",
            )

        log.info(f"[Layer 2] Perspective '{perspective_name}' complete.")

    # -------------------------------------------------------------------------
    # Layer 3: Comprehensive-View Community Graph
    # -------------------------------------------------------------------------

    def build_layer3_overall(self):
        """
        Reads the comprehensive-view HDBSCAN clustering results from
        community/embedding_overall/ipraw_{dev}_embedding_overall_pca.csv and
        constructs the comprehensive-view summary clusters (paper §4.4):

          - Summary entity node of type "Cluster" with perspective ==
            'comprehensive', carrying {cluster_id, perspective, device_type,
            hdbscan_label, is_outlier}. Each device IP belongs to only one
            comprehensive-view cluster.
          - IN_COMMUNITY edge from each Device node to its Cluster node.

        Cluster IDs use a dedicated global 'comprehensive' counter, kept separate
        from the per-single-perspective counters. Outliers (label == -1) each
        become their own singleton Cluster node (rare-vendor preservation).
        """
        log.info("[Layer 3] Building comprehensive-view summary Cluster nodes")

        for dev in self.device_labels:
            pca_csv = os.path.join(self.overall_path, f"ipraw_{dev}_embedding_overall_pca.csv")
            if not os.path.exists(pca_csv):
                log.warning(f"[Layer 3] PCA CSV not found: {pca_csv}, skipping.")
                continue

            df = pd.read_csv(pca_csv, usecols=["ip", "cluster"])
            df["ip"] = df["ip"].astype(str)
            df["cluster"] = df["cluster"].astype(int)

            self._materialize_clusters(
                df=df,
                perspective_attr="comprehensive",
                counter_key="comprehensive",
                device_type=dev,
                layer_tag="Layer 3",
            )

        log.info("[Layer 3] Comprehensive-view summary clusters complete.")

    # -------------------------------------------------------------------------
    # Entity Graph Export  (produces HGT inputs: entity_graph/{node,relation}.csv)
    # -------------------------------------------------------------------------

    def export_entity_graph(self):
        """
        Export the Layer-1 Device–Feature entity graph to
        entity_graph/node.csv and entity_graph/relation.csv, matching the schema
        that HGT.py expects:

          node.csv:     _id, _labels (':Device' or ':Feature'), ip, device_type,
                        feature_name, value
          relation.csv: _start, _end, _type (Has_<feature>)

        These CSVs are the inputs to HGT.py (Operating Mode Representation). They
        were previously assumed to come from a Neo4j `apoc.export.csv` dump, but no
        code produced them. This method regenerates them directly from the raw
        ipraw_{dev}.csv fingerprints so the pipeline is self-contained (no Neo4j /
        APOC dependency), while reusing the exact same Device/Feature/Has_* schema
        as build_layer1_device().
        """
        entity_graph_dir = os.path.join(self.base_path, "entity_graph")
        os.makedirs(entity_graph_dir, exist_ok=True)

        node_rows = []          # {_id, _labels, ip, device_type, feature_name, value}
        relation_rows = []      # {_start, _end, _type}
        feature_id_cache = {}   # (feature_name, value) -> node _id
        next_id = 0

        for dev in self.device_labels:
            csv_path = os.path.join(self.data_path, f"ipraw_{dev}.csv")
            if not os.path.exists(csv_path):
                log.warning(f"[Export] CSV not found: {csv_path}, skipping.")
                continue

            header_df = pd.read_csv(csv_path, nrows=0)
            available_features = [f for f in self.fingerprint_features if f in header_df.columns]
            use_cols = ["ip"] + available_features
            df = pd.read_csv(csv_path, usecols=use_cols)
            df[available_features] = df[available_features].fillna("").astype(str)
            df["ip"] = df["ip"].astype(str)

            log.info(f"[Export] {dev}: {len(df)} devices, {len(available_features)} features")

            for _, row in df.iterrows():
                ip = row["ip"].strip()
                device_id = next_id
                next_id += 1
                node_rows.append({
                    "_id": device_id, "_labels": ":Device",
                    "ip": ip, "device_type": dev,
                    "feature_name": "", "value": "",
                })

                for feat in available_features:
                    value = row[feat].strip()
                    key = (feat, value)
                    feat_id = feature_id_cache.get(key)
                    if feat_id is None:
                        # Shared Feature node across devices (same as MERGE in Neo4j)
                        feat_id = next_id
                        next_id += 1
                        feature_id_cache[key] = feat_id
                        node_rows.append({
                            "_id": feat_id, "_labels": ":Feature",
                            "ip": "", "device_type": "",
                            "feature_name": feat, "value": value,
                        })
                    relation_rows.append({
                        "_start": device_id, "_end": feat_id,
                        "_type": feature_to_rel_type(feat),
                    })

        node_path = os.path.join(entity_graph_dir, "node.csv")
        relation_path = os.path.join(entity_graph_dir, "relation.csv")
        pd.DataFrame(node_rows, columns=["_id", "_labels", "ip", "device_type", "feature_name", "value"]).to_csv(node_path, index=False)
        pd.DataFrame(relation_rows, columns=["_start", "_end", "_type"]).to_csv(relation_path, index=False)

        log.info(f"[Export] {len(node_rows)} nodes → {node_path}")
        log.info(f"[Export] {len(relation_rows)} relations → {relation_path}")

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def run_layer1(self):
        log.info("=== Building Layer 1: Device Entity Graph ===")
        for dev in self.device_labels:
            self.build_layer1_device(dev)

    def run_layer23(self):
        log.info("=== Building Layer 2: Community Graph ===")
        for perspective in self.perspectives:
            self.build_layer2_community(perspective)

        log.info("=== Building Layer 3: Comprehensive-View Community Graph ===")
        self.build_layer3_overall()

    def run(self):
        self.run_layer1()
        self.run_layer23()
        log.info("=== Hierarchical Graph Construction Complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neo4j hierarchical graph builder + entity-graph exporter"
    )
    parser.add_argument("--layer1", action="store_true",
                        help="Build Layer 1 (Device–Feature entity graph) only")
    parser.add_argument("--export", action="store_true",
                        help="Export entity_graph/{node,relation}.csv (HGT inputs)")
    parser.add_argument("--layer23", action="store_true",
                        help="Build Layer 2 + Layer 3 (community graphs) only")
    parser.add_argument("--all", action="store_true",
                        help="Build all layers (Layer1 + Layer2 + Layer3), same as legacy run()")
    parser.add_argument("--devices", nargs="*", default=None,
                        help="Device types to process (default: all discovered)")
    args = parser.parse_args()

    graph = HierarchicalGraph(devices=args.devices)

    # Default (no flag) = legacy full build, for backward compatibility.
    if not (args.layer1 or args.export or args.layer23 or args.all):
        graph.run()
    else:
        if args.layer1:
            graph.run_layer1()
        if args.export:
            graph.export_entity_graph()
        if args.layer23:
            graph.run_layer23()
        if args.all:
            graph.run()