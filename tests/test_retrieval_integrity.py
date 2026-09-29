import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(ROOT, "agent")
for path in (ROOT, AGENT):
    if path not in sys.path:
        sys.path.insert(0, path)

import retrieval  # noqa: E402
import decision  # noqa: E402
import agent as agent_module  # noqa: E402


def bare_retriever(directory, perspectives=("p1", "p2"), embedding_dim=2):
    obj = retrieval.MultiLevelRetrieval.__new__(retrieval.MultiLevelRetrieval)
    obj.local_npz_dir = directory
    obj.retrieval_perspective_names = list(perspectives)
    obj.embedding_dim = embedding_dim
    obj.embedding_overall_dim = len(perspectives) * embedding_dim
    obj._validated_vector_files = set()
    obj._cluster_assignment_cache = {}
    obj._cluster_assignment_errors = {}
    obj._cluster_report_cache = {}
    obj._statistical_report_cache = {}
    obj._single_assignment_cache = {}
    obj._single_summary_cache = {}
    obj.retrieval_history = []
    return obj


class RetrievalIntegrityTests(unittest.TestCase):
    def test_empty_local_index_preserves_search_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            rows, compared, device_best = obj._search_local_vectors(
                np.zeros(4, dtype=np.float32), top_k=5
            )

            self.assertEqual(rows, [])
            self.assertEqual(compared, 0)
            self.assertEqual(device_best, [])

    def test_simple_similarity_does_not_count_ip_as_evidence(self):
        obj = bare_retriever("unused")

        score = obj._simple_similarity(
            {"ip": "query", "os": "Linux"},
            {"ip": "other", "os": "Linux"},
        )

        self.assertEqual(score, 1.0)

    def test_corrupt_cluster_assignments_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            obj.com_view_path = directory
            with open(
                os.path.join(directory, "ipraw_POWER_METER_embedding_overall_pca.csv"),
                "wb",
            ) as handle:
                handle.write(b"\x00\x00not-a-csv")

            result = obj.community_retrieval(
                {"ip": "query"},
                [{"ip": "candidate", "device_type": "POWER_METER"}],
                cache_metadata={"schema_version": "test"},
            )

            self.assertEqual(result["matched_clusters"], [])
            self.assertEqual(
                result["unavailable_clusters"][0]["reason"],
                "cluster_assignments_invalid",
            )

    def test_local_search_is_weighted_cosine_and_excludes_query_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            embeddings = np.array([
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ], dtype=np.float32)
            np.save(os.path.join(directory, "CAMERA_embeddings.npy"), embeddings)
            np.save(os.path.join(directory, "CAMERA_ips.npy"), np.array(["query", "other"]))

            query = np.array([0.75, 0.0, 0.25, 0.0], dtype=np.float32)
            rows, compared, device_best = obj._search_local_vectors(
                query, top_k=2, exclude_ips={"query"}
            )

            self.assertEqual(compared, 2)
            self.assertEqual([row["ip"] for row in rows], ["other"])
            self.assertEqual(rows[0]["similarity_score"], 0.75)
            self.assertGreaterEqual(rows[0]["similarity_score"], -1.0)
            self.assertLessEqual(rows[0]["similarity_score"], 1.0)
            self.assertEqual([row["ip"] for row in device_best], ["other"])

    def test_cache_is_bound_to_full_fingerprint_and_top_k(self):
        fingerprint = {"ip": "1.2.3.4", "os-vendor": "vendor-a"}
        record = {"cache_metadata": retrieval.build_cache_metadata(fingerprint, 5)}

        self.assertTrue(retrieval.cache_record_matches(record, fingerprint, 5))
        self.assertFalse(retrieval.cache_record_matches(
            record, {"ip": "1.2.3.4", "os-vendor": "vendor-b"}, 5
        ))
        self.assertFalse(retrieval.cache_record_matches(record, fingerprint, 10))
        self.assertFalse(retrieval.cache_record_matches({}, fingerprint, 5))

    def test_community_marks_noise_and_missing_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            obj.com_view_path = os.path.join(directory, "overall")
            os.makedirs(obj.com_view_path)
            pd.DataFrame([
                {"ip": "noise", "cluster": -1},
                {"ip": "member", "cluster": 0},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )

            result = obj.community_retrieval(
                {"ip": "query"},
                [
                    {"ip": "noise", "device_type": "CAMERA"},
                    {"ip": "member", "device_type": "CAMERA"},
                ],
                cache_metadata={"schema_version": "test"},
            )

            self.assertEqual(result["matched_clusters"], [])
            self.assertEqual(
                {row["reason"] for row in result["unavailable_clusters"]},
                {"noise_cluster", "cluster_report_missing"},
            )

    def test_importance_uses_complete_cluster_not_local_hit_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory, perspectives=("p",), embedding_dim=2)
            obj.com_view_path = os.path.join(directory, "overall")
            obj.single_view_path = os.path.join(directory, "single")
            os.makedirs(obj.com_view_path)
            single_dir = os.path.join(obj.single_view_path, "embedding_p")
            os.makedirs(single_dir)

            pd.DataFrame([
                {"ip": "a", "cluster": 7},
                {"ip": "b", "cluster": 7},
                {"ip": "c", "cluster": 7},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )
            pd.DataFrame([
                {"ip": "a", "cluster": 2},
                {"ip": "b", "cluster": 2},
                {"ip": "c", "cluster": 2},
                {"ip": "outside", "cluster": -1},
            ]).to_csv(
                os.path.join(single_dir, "ipraw_CAMERA_embedding_p_pca.csv"),
                index=False,
            )
            with open(os.path.join(single_dir, "CAMERA_cluster_summaries.json"), "w") as handle:
                json.dump([{
                    "cluster_id": 2,
                    "analysis": {"common_patterns": {"field": "value"}},
                }], handle)

            result = obj.calculate_importance([{
                "device_type": "CAMERA",
                "cluster_id": 7,
                "related_ips": ["a"],
            }])

            cluster = result["CAMERA_7"]
            self.assertEqual(cluster["cluster_size"], 3)
            self.assertEqual(cluster["important_features"], ["p"])
            self.assertEqual(cluster["feature_importance"]["p"]["support"], 3)

    def test_missing_summary_uses_statistical_report_when_raw_members_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory, perspectives=("p",), embedding_dim=2)
            obj.com_view_path = os.path.join(directory, "overall")
            obj.perspective_info_config = {"p": {"cols": ["field"]}}
            os.makedirs(obj.com_view_path)
            pd.DataFrame([
                {"ip": "a", "cluster": 3},
                {"ip": "b", "cluster": 3},
                {"ip": "c", "cluster": 3},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )
            all_dir = os.path.join(directory, "all")
            os.makedirs(all_dir)
            pd.DataFrame([
                {"ip": "a", "field": "same"},
                {"ip": "b", "field": "same"},
                {"ip": "c", "field": "other"},
            ]).to_csv(os.path.join(all_dir, "ipraw_CAMERA.csv"), index=False)
            obj._match_fingerprint_with_cluster = lambda *args, **kwargs: {
                "available": True,
                "similarity_score": 0.8,
                "matched_features": ["field"],
                "unmatched_features": [],
            }

            with mock.patch.object(retrieval, "CSV_DATA_DIR", directory):
                result = obj.community_retrieval(
                    {"ip": "query", "field": "same"},
                    [{"ip": "a", "device_type": "CAMERA"}],
                    cache_metadata={"schema_version": "test"},
                )

            self.assertEqual(len(result["matched_clusters"]), 1)
            match = result["matched_clusters"][0]
            self.assertEqual(match["report_source"], "deterministic_frequency_summary")
            self.assertEqual(match["cluster_size"], 3)
            self.assertEqual(match["report"]["sample_size"], 3)

    def test_decision_tool_does_not_expose_filename_label(self):
        with tempfile.TemporaryDirectory() as directory:
            local_dir = os.path.join(directory, "local")
            os.makedirs(local_dir)
            with open(os.path.join(local_dir, "CAMERA_local.json"), "w") as handle:
                json.dump([{
                    "query_fingerprint": {"ip": "1.2.3.4"},
                    "top_k": 5,
                    "similar_devices": [],
                    "confidence_score": 0.5,
                    "missing_perspectives": ["dns"],
                }], handle)

            with mock.patch.object(decision, "_QDB_PATH", directory), \
                    mock.patch.object(decision, "_dev_labels", ["CAMERA"]), \
                    mock.patch.object(decision, "_retrieval_runtime", None):
                section = decision._local_section("1.2.3.4")

            self.assertEqual(section["status"], "found")
            self.assertNotIn("candidate_dev", section)
            self.assertEqual(section["missing_perspectives"], ["dns"])

    def test_joint_vote_never_boosts_uncalibrated_confidence(self):
        gemini = {
            "device_type": "CAMERA", "vendor": "A", "confidence": 1.0, "llm": "G"
        }
        claude = {
            "device_type": "CAMERA", "vendor": "A", "confidence": 1.0, "llm": "C"
        }
        agreed = decision.DecisionAgent._joint_vote(gemini, claude)
        self.assertEqual(agreed["final_confidence"], 0.89)

        claude["device_type"] = "SCADA"
        disagreed = decision.DecisionAgent._joint_vote(gemini, claude)
        self.assertEqual(disagreed["final_confidence"], 0.69)

    def test_vendor_vote_uses_vendor_confidence_independently(self):
        gemini = {
            "device_type": "CAMERA", "vendor": "Vendor-G",
            "type_confidence": 0.9, "vendor_confidence": 0.2, "llm": "gemini",
        }
        claude = {
            "device_type": "ROUTER", "vendor": "Vendor-C",
            "type_confidence": 0.4, "vendor_confidence": 0.8, "llm": "claude",
        }

        result = decision.DecisionAgent._joint_vote(gemini, claude)

        self.assertEqual(result["final_device_type"], "CAMERA")
        self.assertEqual(result["winning_llm"], "gemini")
        self.assertEqual(result["final_vendor"], "Vendor-C")
        self.assertEqual(result["winning_vendor_llm"], "claude")
        self.assertLessEqual(result["final_vendor_confidence"], 0.69)

    def test_vendor_novelty_does_not_disable_type_drift_check(self):
        graph = agent_module.IoTDecisionGraph.__new__(agent_module.IoTDecisionGraph)
        graph.enable_first_stage = True
        update = graph._gate_node({
            "unseen_result": {
                "new_type_probability": 0.1,
                "new_vendor_probability": 0.99,
            }
        })
        self.assertTrue(update["first_stage"]["drift_checked"])


if __name__ == "__main__":
    unittest.main()
