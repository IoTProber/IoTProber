import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
if DEMO not in sys.path:
    sys.path.insert(0, DEMO)

from evaluate_fullflow import summarize  # noqa: E402


class DemoEvaluationTests(unittest.TestCase):
    def test_summary_reports_each_retrieval_level_and_integrity_gate(self):
        rows = [{
            "ip": "query",
            "true_type": "CAMERA",
            "predicted_type": "CAMERA",
            "decision_confidence": 0.8,
            "type_correct": True,
            "status": "completed",
            "retrieval_evidence": {
                "local": {"similar_devices": [
                    {"ip": "other", "device_type": "CAMERA", "similarity_score": 0.9}
                ]},
                "community": {"matched_clusters": [
                    {"device_type": "CAMERA", "similarity_score": 0.8}
                ]},
                "reasoning": {"path_matching_results": [
                    {"cluster_info": {"device_type": "CAMERA"}, "path_matching_score": 0.7}
                ]},
            },
        }]

        result = summarize(rows)

        self.assertEqual(result["final_accuracy"], 1.0)
        self.assertEqual(result["completion_rate"], 1.0)
        self.assertEqual(result["end_to_end_accuracy"], 1.0)
        self.assertEqual(result["local_recall_at_k"], 1.0)
        self.assertEqual(result["community_top1_accuracy"], 1.0)
        self.assertEqual(result["reasoning_top1_accuracy"], 1.0)
        self.assertEqual(result["self_match_violations"], 0)
        self.assertEqual(result["score_range_violations"], 0)

    def test_missing_layers_reduce_end_to_end_metrics(self):
        rows = [{
            "ip": "missing",
            "true_type": "CAMERA",
            "predicted_type": None,
            "decision_confidence": None,
            "type_correct": False,
            "status": "incomplete",
            "retrieval_evidence": {},
        }, {
            "ip": "done",
            "true_type": "CAMERA",
            "predicted_type": "CAMERA",
            "decision_confidence": 0.8,
            "type_correct": True,
            "status": "completed",
            "retrieval_evidence": {
                "local": {"similar_devices": []},
                "community": {"matched_clusters": []},
                "reasoning": {"path_matching_results": []},
            },
        }]

        result = summarize(rows)

        self.assertEqual(result["completion_rate"], 0.5)
        self.assertEqual(result["final_accuracy"], 1.0)
        self.assertEqual(result["end_to_end_accuracy"], 0.5)
        self.assertEqual(result["local_recall_at_k"], 0.0)
        self.assertEqual(result["local_availability"], 0.0)


if __name__ == "__main__":
    unittest.main()
