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
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.data_path = os.path.join(self.base_path, "platform_data", "csv", "local", "1")
        self.community_path = os.path.join(self.data_path, "community", "single")
        self.overall_path = os.path.join(self.data_path, "community", "embedding_overall")
        self.graph = ProtocolGraph("neo4j://localhost:7687", "neo4j", "12345678")

        self.fingerprint_features = self._load_fingerprint_features()
        self.device_labels = self._discover_device_labels()
        self.perspectives = self._discover_perspectives()

    def _load_fingerprint_features(self):
        features_path = os.path.join(self.base_path, "local_used_features.txt")
        with open(features_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        log.info(f"Loaded {len(features)} fingerprint features: {features}")
        return features

    def _discover_device_labels(self):
        device_labels = []
        for fname in sorted(os.listdir(self.data_path)):
            if fname.startswith("ipraw_") and fname.endswith(".csv"):
                dev = fname[len("ipraw_"):-len(".csv")]
                device_labels.append(dev)
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
    # Layer 2: Community Graph
    # -------------------------------------------------------------------------

    def build_layer2_community(self, perspective: str):
        """
        Reads the HDBSCAN clustering results (PCA CSV) for each device type under
        the given perspective directory, and constructs the second-layer community graph:
          - SingleCommunity node per (perspective, device_type, cluster_id): {perspective, device_type, cluster_id}
          - IN_COMMUNITY edge from each Device node to its SingleCommunity node

        Noise points (cluster == -1) are excluded.
        perspective: subdirectory name under community/single, e.g. 'embedding_as'
        """
        perspective_name = perspective[len("embedding_"):]
        perspective_dir = os.path.join(self.community_path, perspective)

        log.info(f"[Layer 2] Building community graph for perspective: {perspective_name}")

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
            df_valid = df[df["cluster"] != -1].copy()

            n_clusters = df_valid["cluster"].nunique()
            log.info(f"[Layer 2] {perspective_name}/{dev}: {len(df_valid)} devices in {n_clusters} clusters")

            cluster_node_cache = {}
            for cluster_id in df_valid["cluster"].unique():
                cid = int(cluster_id)
                community_node, _ = self.graph.CreateNode(["SingleCommunity"], {
                    "perspective": perspective_name,
                    "device_type": dev,
                    "cluster_id": cid
                })
                cluster_node_cache[cid] = community_node

            processed = 0
            for _, row in df_valid.iterrows():
                ip = row["ip"].strip()
                cluster_id = int(row["cluster"])

                device_node = self.graph.MatchSingleNode(["Device"], {"ip": ip})
                if device_node is None:
                    log.warning(f"[Layer 2] Device node not found for IP {ip}, skipping community edge.")
                    continue

                community_node = cluster_node_cache.get(cluster_id)
                if community_node is None:
                    continue

                self.graph.CreateRelationship(
                    device_node, community_node, "IN_COMMUNITY",
                    {"perspective": perspective_name}
                )
                processed += 1

            log.info(f"[Layer 2] {perspective_name}/{dev} done! ({processed} edges created)")

        log.info(f"[Layer 2] Perspective '{perspective_name}' complete.")

    # -------------------------------------------------------------------------
    # Layer 3: Comprehensive-View Community Graph
    # -------------------------------------------------------------------------

    def build_layer3_overall(self):
        """
        Reads the comprehensive-view HDBSCAN clustering results from
        community/embedding_overall/ipraw_{dev}_embedding_overall_pca.csv and constructs
        the third-layer community graph:
          - ComCluster node per (device_type, cluster_id): {device_type, cluster_id}
          - IN_COMMUNITY edge from each Device node to its ComCluster node

        Noise points (cluster == -1) are excluded.
        """
        log.info("[Layer 3] Building comprehensive-view community graph")

        for dev in self.device_labels:
            pca_csv = os.path.join(self.overall_path, f"ipraw_{dev}_embedding_overall_pca.csv")
            if not os.path.exists(pca_csv):
                log.warning(f"[Layer 3] PCA CSV not found: {pca_csv}, skipping.")
                continue

            df = pd.read_csv(pca_csv, usecols=["ip", "cluster"])
            df["ip"] = df["ip"].astype(str)
            df["cluster"] = df["cluster"].astype(int)
            df_valid = df[df["cluster"] != -1].copy()

            n_clusters = df_valid["cluster"].nunique()
            log.info(f"[Layer 3] overall/{dev}: {len(df_valid)} devices in {n_clusters} clusters")

            cluster_node_cache = {}
            for cluster_id in df_valid["cluster"].unique():
                cid = int(cluster_id)
                community_node, _ = self.graph.CreateNode(["ComCluster"], {
                    "device_type": dev,
                    "cluster_id": cid
                })
                cluster_node_cache[cid] = community_node

            processed = 0
            for _, row in df_valid.iterrows():
                ip = row["ip"].strip()
                cluster_id = int(row["cluster"])

                device_node = self.graph.MatchSingleNode(["Device"], {"ip": ip})
                if device_node is None:
                    log.warning(f"[Layer 3] Device node not found for IP {ip}, skipping community edge.")
                    continue

                community_node = cluster_node_cache.get(cluster_id)
                if community_node is None:
                    continue

                self.graph.CreateRelationship(device_node, community_node, "IN_COMMUNITY", {})
                processed += 1

            log.info(f"[Layer 3] overall/{dev} done! ({processed} edges created)")

        log.info("[Layer 3] Comprehensive-view community graph complete.")

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def run(self):
        log.info("=== Building Layer 1: Device Entity Graph ===")
        for dev in self.device_labels:
            self.build_layer1_device(dev)

        log.info("=== Building Layer 2: Community Graph ===")
        for perspective in self.perspectives:
            self.build_layer2_community(perspective)

        log.info("=== Building Layer 3: Comprehensive-View Community Graph ===")
        self.build_layer3_overall()

        log.info("=== Hierarchical Graph Construction Complete ===")


if __name__ == "__main__":
    graph = HierarchicalGraph()
    graph.run()