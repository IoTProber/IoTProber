"""
agent/unseen.py

Unseen IoT Device Detection Module (基于 LLama-3.1-8B-Instruct 的 unseen 设备识别模块)

Uses LLama-3.1-8B-Instruct (base or fine-tuned) to identify whether a query device
is "unseen" (not belonging to any known device type in rag_devices.json), and if so,
predict its possible type and vendor from the broader all_IoT_devices.json catalogue.

Input:
    key perspectives 和 non-key perspectives, 来自 reasoning_path_retrieval 的输出.
    Key perspectives and non-key perspectives from reasoning_path_retrieval output.

Output:
    - unseen_probability: float [0, 1], 设备属于 unseen 的概率
    - is_unseen: bool, 是否属于 unseen 设备
    - predicted_type: str, 预测的设备类型 (来自 all_IoT_devices.json)
    - predicted_vendor: str, 预测的设备厂商
    - confidence: float [0, 1], 整体预测置信度
    - reasoning: str, 推理链说明

Design:
    Stage 1 — Statistical Pre-screening (统计预筛选):
        从检索结果中提取数值型 unseen 指标 (路径匹配分数, 相似度差距, perspective 覆盖率).
    Stage 2 — LLM Reasoning (LLM 推理):
        构建结构化 prompt, 包含 key/non-key perspectives 数据和统计指标,
        送入 LLama 模型进行 chain-of-thought unseen 检测.
    Training Support (训练支持):
        生成 SFT 训练样本, 用于 LoRA 微调.

Usage:
    from unseen import UnseenDeviceDetector

    detector = UnseenDeviceDetector(gpu=0)
    result = detector.detect_unseen(
        reasoning_result=reasoning_result,   # from reasoning_path_retrieval
        local_result=local_result,           # from local_retrieval (optional)
    )
"""

import os
import sys
import json
import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import numpy as np
import torch

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import load_perspective_info

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "unseen.log"),
    filemode="a",
)

# ── Path Constants ───────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODEL_PATH = os.path.join(_BASE, "Meta-Llama-3.1-8B-Instruct")
_RAG_DEVICES_PATH = os.path.join(_BASE, "rag_devices.json")
_ALL_DEVICES_PATH = os.path.join(_BASE, "all_IoT_devices.json")

# 排除不参与局部检索和推理路径检索的 perspective
# Perspectives excluded from retrieval-level analysis
_EXCEPT_PERSPECTIVES = ["hpart", "http", "overall"]

# perspective 名称到 community report common_patterns 键的映射
# Mapping from perspective names to community report common_patterns keys
_PERSPECTIVE_TO_REPORT_KEY = {
    "as": "autonomous_system",
    "whois": "whois",
    "os": "os",
    "sw": "software",
    "hw": "hardware",
    "sd": "service_distribution",
    "body": "http_tags_favicons",
    "htags": "http_tags_favicons",
    "hfavicons": "http_tags_favicons",
    "certificate": "certificates",
    "dns": "dns",
}

# Unseen 概率阈值 / Unseen probability threshold
_UNSEEN_THRESHOLD = 0.5


# ═════════════════════════════════════════════════════════════════════════════
# §1  UnseenDeviceDetector
# ═════════════════════════════════════════════════════════════════════════════

class UnseenDeviceDetector:
    """
    基于 LLama-3.1-8B-Instruct 的 unseen IoT 设备检测器.
    Unseen IoT device detector based on LLama-3.1-8B-Instruct.

    Core idea:
        reasoning_path_retrieval 将 perspectives 分为 "key" (低 Shannon 熵, 高区分性)
        和 "non-key". 若查询设备的 key perspectives 与所有已知 RAG 设备 cluster 匹配度低,
        则该设备很可能是 "unseen".

        LLama 模型对结构化的 perspective 证据进行推理, 输出:
          1) P(unseen) — 设备不属于 rag_devices 的概率
          2) 预测类型 (来自 all_IoT_devices, 若 unseen)
          3) 预测厂商

    Supports:
        - Base LLama-3.1-8B-Instruct (零样本, 结构化 prompt)
        - Fine-tuned checkpoint (全量微调或 LoRA adapter)
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        gpu: int = -1,
        torch_dtype=None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 1024,
    ):
        """
        初始化 UnseenDeviceDetector
        Initialize UnseenDeviceDetector

        Args:
            model_path: LLama-3.1-8B-Instruct 基座模型路径
                        Path to LLama-3.1-8B-Instruct base model
            adapter_path: LoRA adapter 路径 (微调后的 PEFT 权重)
                          Path to LoRA adapter (fine-tuned with PEFT)
            gpu: GPU 设备索引, -1 表示 CPU
                 GPU device index, -1 for CPU
            torch_dtype: 模型精度, 默认 GPU 用 bfloat16, CPU 用 float32
                         Model dtype, default bfloat16 for GPU, float32 for CPU
            load_in_4bit: 是否使用 bitsandbytes 4-bit 量化
                          Use bitsandbytes 4-bit quantization
            load_in_8bit: 是否使用 bitsandbytes 8-bit 量化
                          Use bitsandbytes 8-bit quantization
            max_new_tokens: 生成的最大 token 数
                            Maximum tokens for generation
        """
        self.base_path = _BASE
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self.adapter_path = adapter_path
        self.gpu = gpu
        self.device_str = f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens

        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device_str != "cpu" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # 加载设备列表 / Load device lists
        self.rag_devices: List[str] = self._load_json(_RAG_DEVICES_PATH)["IoT"]
        self.all_devices: List[str] = self._load_json(_ALL_DEVICES_PATH)["IoT"]
        self.unseen_candidates: List[str] = [
            d for d in self.all_devices if d not in self.rag_devices
        ]

        # 加载 perspective 配置 / Load perspective config
        self.perspective_info = load_perspective_info()
        self.retrieval_perspectives = [
            p for p in self.perspective_info.keys() if p not in _EXCEPT_PERSPECTIVES
        ]

        # 加载模型 / Load model
        self._load_model(load_in_4bit, load_in_8bit)

        logging.info(
            f"UnseenDeviceDetector initialized: model={self.model_path}, "
            f"adapter={self.adapter_path}, device={self.device_str}, "
            f"known_types={len(self.rag_devices)}, "
            f"unseen_candidates={len(self.unseen_candidates)}"
        )

    # ── Initialization helpers ───────────────────────────────────────────

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """
        加载 LLama-3.1-8B-Instruct 模型和 tokenizer
        Load LLama-3.1-8B-Instruct model and tokenizer
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logging.info(f"Loading LLama tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 量化配置 / Quantization config
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }

        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = (
                self.device_str if self.device_str != "cpu" else "cpu"
            )

        logging.info(f"Loading LLama model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )

        # 加载 LoRA adapter (微调后的权重)
        # Load LoRA adapter (fine-tuned weights)
        if self.adapter_path and os.path.isdir(self.adapter_path):
            from peft import PeftModel

            logging.info(f"Loading LoRA adapter from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        logging.info("LLama model loaded successfully")

    # ═════════════════════════════════════════════════════════════════════
    # §2  Perspective Extraction (从 reasoning_path_retrieval 提取关键/非关键视角)
    # ═════════════════════════════════════════════════════════════════════

    def extract_perspectives(
        self, reasoning_result: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        从 reasoning_path_retrieval 的输出中提取关键和非关键 perspectives.
        Extract key and non-key perspectives from reasoning_path_retrieval output.

        关键 perspective: is_important=True (归一化 Shannon 熵 < 0.1)
        Key perspectives: those with is_important=True (normalized Shannon entropy < 0.1)

        非关键 perspective: 剩余的 retrieval perspectives
        Non-key perspectives: the remaining retrieval perspectives

        Args:
            reasoning_result: reasoning_path_retrieval 的输出
                              Output from reasoning_path_retrieval

        Returns:
            (key_perspectives, non_key_perspectives)
            每个映射 perspective 名称到包含特征值, 重要性分数, 匹配信息的 detail dict.
            Each maps perspective name to a detail dict with feature values,
            importance scores, and matching information.
        """
        if not reasoning_result or "path_matching_results" not in reasoning_result:
            return {}, {}

        query_fp = reasoning_result.get("query_fingerprint", {})
        pmrs = reasoning_result["path_matching_results"]

        # 聚合所有 cluster 的关键 perspective 名称和最佳匹配详情
        # Aggregate key perspective names and best matching details across all clusters
        key_perspective_names: set = set()
        best_perspective_detail: Dict[str, Dict] = {}

        for pmr in pmrs:
            # important_features 包含关键 perspective 名称
            # important_features contains key perspective names
            for feat in pmr.get("important_features", []):
                key_perspective_names.add(feat["feature_name"])

            # feature_matching_details 包含每个 perspective 的匹配分数
            # feature_matching_details contains per-perspective matching scores
            cluster_key = pmr.get("cluster_key", "")
            device_type = pmr.get("cluster_info", {}).get("device_type", "")

            for fmd in pmr.get("feature_matching_details", []):
                fname = fmd["feature_name"]

                best_sim = max(
                    (vs["similarity"] for vs in fmd.get("value_similarities", [])),
                    default=0.0,
                )

                entry = {
                    "importance_score": fmd.get("importance_score", 0.0),
                    "feature_matching_score": fmd.get("feature_matching_score", 0.0),
                    "weighted_feature_score": fmd.get("weighted_feature_score", 0.0),
                    "best_cluster_similarity": best_sim,
                    "matched_device_type": device_type,
                    "cluster_key": cluster_key,
                    "top_similarities": sorted(
                        [
                            {
                                "cluster_id": vs["cluster_id"],
                                "similarity": vs["similarity"],
                                "weight": vs["weight"],
                                "analysis_preview": vs.get("analysis_preview", "")[:200],
                            }
                            for vs in fmd.get("value_similarities", [])
                        ],
                        key=lambda x: x["similarity"],
                        reverse=True,
                    )[:3],
                }

                # 保留 feature_matching_score 最高的条目
                # Keep entry with highest feature_matching_score
                if (
                    fname not in best_perspective_detail
                    or entry["feature_matching_score"]
                    > best_perspective_detail[fname]["feature_matching_score"]
                ):
                    best_perspective_detail[fname] = entry

        # 构建包含查询指纹值的 perspective 字典
        # Build perspective dicts with query fingerprint values
        key_perspectives: Dict[str, Any] = {}
        non_key_perspectives: Dict[str, Any] = {}

        for p_name in self.retrieval_perspectives:
            cols = self.perspective_info[p_name]["cols"]
            values: Dict[str, str] = {}
            for col in cols:
                val = query_fp.get(col)
                if val is not None and str(val).strip() not in ("", "nan", "None"):
                    values[col] = str(val)

            detail = best_perspective_detail.get(p_name, {})
            entry = {
                "perspective_name": p_name,
                "description": self.perspective_info[p_name]["prompt_info"],
                "weight": self.perspective_info[p_name]["weight"],
                "feature_values": values,
                "importance_score": detail.get("importance_score", 0.0),
                "feature_matching_score": detail.get("feature_matching_score", 0.0),
                "best_cluster_similarity": detail.get("best_cluster_similarity", 0.0),
                "matched_device_type": detail.get("matched_device_type", ""),
                "cluster_key": detail.get("cluster_key", ""),
                "top_similarities": detail.get("top_similarities", []),
            }

            if p_name in key_perspective_names:
                key_perspectives[p_name] = entry
            else:
                non_key_perspectives[p_name] = entry

        return key_perspectives, non_key_perspectives

    # ═════════════════════════════════════════════════════════════════════
    # §3  Statistical Unseen Indicators (统计型 unseen 指标)
    # ═════════════════════════════════════════════════════════════════════

    def compute_unseen_indicators(
        self,
        reasoning_result: Dict[str, Any],
        local_result: Optional[Dict[str, Any]] = None,
        key_perspectives: Optional[Dict] = None,
        non_key_perspectives: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        计算统计型 unseen 指标, 辅助 LLM 判断.
        Compute statistical indicators that suggest the device may be unseen.

        Returns:
            包含数值特征的 dict: best_path_score, top_similarity,
            key_avg_similarity, dominant_type_ratio, heuristic_unseen_score 等.
            Dict containing numerical features for unseen assessment.
        """
        indicators: Dict[str, Any] = {}

        # ── 1. 路径匹配分数 / Path matching scores ──
        pmrs = reasoning_result.get("path_matching_results", [])
        if pmrs:
            indicators["best_path_score"] = pmrs[0]["path_matching_score"]
            indicators["best_device_type"] = (
                pmrs[0].get("cluster_info", {}).get("device_type", "UNKNOWN")
            )
            indicators["num_clusters_analyzed"] = len(pmrs)
            if len(pmrs) > 1:
                indicators["second_path_score"] = pmrs[1]["path_matching_score"]
                indicators["score_gap"] = (
                    pmrs[0]["path_matching_score"] - pmrs[1]["path_matching_score"]
                )
            # 分析 top cluster 中涉及的设备类型
            # Analyse device types involved in top clusters
            top_cluster_types = [
                p.get("cluster_info", {}).get("device_type", "") for p in pmrs[:5]
            ]
            indicators["top_cluster_types"] = list(set(top_cluster_types))
        else:
            indicators["best_path_score"] = 0.0
            indicators["best_device_type"] = "UNKNOWN"
            indicators["num_clusters_analyzed"] = 0
            indicators["top_cluster_types"] = []

        # ── 2. 局部检索指标 / Local retrieval indicators ──
        if local_result:
            indicators["local_confidence"] = local_result.get("confidence_score", 0.0)
            similar_devices = local_result.get("similar_devices", [])
            if similar_devices:
                indicators["top_similarity"] = similar_devices[0].get(
                    "similarity_score", 0.0
                )
                types = [d["device_type"] for d in similar_devices[:5]]
                type_counts = Counter(types)
                indicators["dominant_type"] = (
                    type_counts.most_common(1)[0][0] if types else "UNKNOWN"
                )
                indicators["dominant_type_ratio"] = (
                    type_counts.most_common(1)[0][1] / len(types) if types else 0.0
                )
                indicators["type_diversity"] = len(type_counts)
            else:
                indicators["top_similarity"] = 0.0
                indicators["dominant_type"] = "UNKNOWN"
                indicators["dominant_type_ratio"] = 0.0
                indicators["type_diversity"] = 0
            indicators["missing_perspectives"] = local_result.get(
                "missing_perspectives", []
            )
        else:
            indicators["local_confidence"] = 0.0
            indicators["top_similarity"] = 0.0
            indicators["dominant_type"] = "UNKNOWN"
            indicators["missing_perspectives"] = []

        # ── 3. Key perspective 平均相似度 / Key perspective avg similarity ──
        if key_perspectives:
            key_sims = [
                v["best_cluster_similarity"]
                for v in key_perspectives.values()
                if v["best_cluster_similarity"] > 0
            ]
            indicators["key_avg_similarity"] = (
                float(np.mean(key_sims)) if key_sims else 0.0
            )
            indicators["key_min_similarity"] = (
                float(np.min(key_sims)) if key_sims else 0.0
            )
            indicators["num_key_perspectives"] = len(key_perspectives)
        else:
            indicators["key_avg_similarity"] = 0.0
            indicators["key_min_similarity"] = 0.0
            indicators["num_key_perspectives"] = 0

        # ── 4. Non-key perspective 平均相似度 / Non-key perspective avg similarity ──
        if non_key_perspectives:
            nk_sims = [
                v["best_cluster_similarity"]
                for v in non_key_perspectives.values()
                if v["best_cluster_similarity"] > 0
            ]
            indicators["nonkey_avg_similarity"] = (
                float(np.mean(nk_sims)) if nk_sims else 0.0
            )
            indicators["num_nonkey_perspectives"] = len(non_key_perspectives)
        else:
            indicators["nonkey_avg_similarity"] = 0.0
            indicators["num_nonkey_perspectives"] = 0

        # ── 5. 启发式 unseen 分数 / Heuristic unseen score ──
        #    综合考虑路径匹配分数, key perspective 相似度, 局部检索置信度
        #    A quick heuristic combining path score, key similarity, local confidence
        path_s = indicators.get("best_path_score", 0.0)
        key_s = indicators.get("key_avg_similarity", 0.0)
        local_s = indicators.get("top_similarity", 0.0)
        # 加权平均: 路径分数 40%, key perspective 相似度 40%, 局部相似度 20%
        # Weighted mean: path score 40%, key similarity 40%, local similarity 20%
        combined = 0.4 * path_s + 0.4 * key_s + 0.2 * local_s
        # 映射到 unseen 概率: 越低越可能是 unseen
        # Map to unseen probability: lower combined → higher unseen probability
        indicators["heuristic_unseen_score"] = round(1.0 - combined, 4)

        return indicators

    # ═════════════════════════════════════════════════════════════════════
    # §4  Prompt Construction (构建检测 prompt)
    # ═════════════════════════════════════════════════════════════════════

    # ── Tavily web search for vendor attribution ────────────────────────

    def _tavily_vendor_search(
        self,
        key_perspectives: Dict[str, Any],
        query_fp: Dict,
    ) -> str:
        """
        使用 Tavily web search 搜索关键设备标识符, 辅助厂商归属.
        Query Tavily web search with discriminative device identifiers
        (e.g., DNS domains, OUI prefixes, HTTP headers) to aid vendor attribution.

        Returns:
            格式化的搜索结果摘要字符串. 若搜索不可用或无结果, 返回占位提示.
            Formatted search result summary. Returns placeholder if unavailable.
        """
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not _TAVILY_AVAILABLE or not api_key:
            return "  (Tavily web search unavailable – no API key or package not installed)"

        # 从 key perspectives 和 query fingerprint 中提取可搜索的关键词
        # Extract searchable keywords from key perspectives and query fingerprint
        keywords: List[str] = []

        # DNS 域名
        dns_vals = key_perspectives.get("dns", {}).get("feature_values", {})
        for v in dns_vals.values():
            # 取域名中的主域部分
            for token in str(v).split(";"):
                token = token.strip()
                if "." in token and len(token) > 4:
                    keywords.append(token)

        # MAC OUI 前缀
        mac = query_fp.get("mac", "")
        if mac and len(str(mac)) >= 8:
            keywords.append(str(mac)[:8])  # OUI prefix

        # HTTP User-Agent / Server 字段
        for ua_key in ("http_useragent", "http_server", "useragent"):
            ua = query_fp.get(ua_key, "")
            if ua and str(ua).strip() not in ("", "nan", "None"):
                keywords.append(str(ua)[:120])

        # certificate 中的 CN / O 字段
        cert_vals = key_perspectives.get("certificate", {}).get("feature_values", {})
        for v in cert_vals.values():
            if v and len(str(v)) > 3:
                keywords.append(str(v)[:120])

        if not keywords:
            return "  (No discriminative keywords found for web search)"

        # 去重, 最多取 3 个关键词构造查询
        seen = set()
        unique_kw: List[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_kw.append(kw)
            if len(unique_kw) >= 3:
                break

        query = "IoT device vendor manufacturer " + " ".join(unique_kw)
        logging.info(f"Tavily vendor search query: {query}")

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=3,
            )
            results = response.get("results", [])
            if not results:
                return "  (No web search results found)"

            lines: List[str] = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                snippet = r.get("content", "")[:200]
                url = r.get("url", "")
                lines.append(f"  {i}. **{title}** ({url})\n     {snippet}")
            return "\n".join(lines)

        except Exception as e:
            logging.warning(f"Tavily search failed: {e}")
            return f"  (Web search failed: {e})"

    def _extract_community_perspectives(
        self, community_result: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        从 community_retrieval 结果中提取各 cluster 的 perspective 描述.
        Extract per-perspective descriptions from community retrieval matched clusters.

        Returns:
            列表, 每个元素是一个 cluster 的信息, 包含 device_type, cluster_id,
            similarity_score, 以及 perspective_descriptions (perspective_name → text).
        """
        if not community_result:
            return []

        clusters_info: List[Dict[str, Any]] = []
        for mc in community_result.get("matched_clusters", []):
            report = mc.get("report", {})
            patterns = {}
            if isinstance(report, dict):
                patterns = report.get("common_patterns", {})

            # 将 community report key 反向映射到 perspective 名称
            # Reverse-map community report keys back to perspective names
            persp_desc: Dict[str, str] = {}
            for p_name, rkey in _PERSPECTIVE_TO_REPORT_KEY.items():
                if rkey in patterns:
                    persp_desc[p_name] = str(patterns[rkey])[:500]

            clusters_info.append({
                "device_type": mc.get("device_type", "UNKNOWN"),
                "cluster_id": mc.get("cluster_id", -1),
                "similarity_score": mc.get("similarity_score", 0.0),
                "matched_features": mc.get("matched_features", []),
                "unmatched_features": mc.get("unmatched_features", []),
                "perspective_descriptions": persp_desc,
            })

        # 按相似度降序排列
        clusters_info.sort(key=lambda x: x["similarity_score"], reverse=True)
        return clusters_info

    def _format_key_vs_community(
        self,
        key_perspectives: Dict[str, Any],
        community_clusters: List[Dict[str, Any]],
    ) -> str:
        """
        格式化 key perspectives 与 community cluster perspectives 的逐一对比.
        Format side-by-side comparison of key perspectives vs community cluster perspectives.
        """
        if not key_perspectives:
            return "  No key perspectives available.\n"

        lines: List[str] = []

        for p_name, detail in sorted(
            key_perspectives.items(),
            key=lambda x: x[1].get("importance_score", 0),
            reverse=True,
        ):
            # 查询设备的 key perspective 特征值
            # Query device key perspective feature values
            values = detail.get("feature_values", {})
            if not values:
                values_str = "(no data)"
            else:
                values_str = "; ".join(f"{k}={v[:200]}" for k, v in values.items())

            lines.append(
                f"### Perspective: {p_name} "
                f"(importance={detail['importance_score']:.3f}, "
                f"weight={detail['weight']}, "
                f"matching_score={detail['feature_matching_score']:.3f})"
            )
            lines.append(f"  **Query Device**: {values_str}")

            # 逐 cluster 对比 / Compare against each community cluster
            for ci in community_clusters[:3]:
                cluster_desc = ci["perspective_descriptions"].get(p_name, "(N/A)")
                sim = detail.get("best_cluster_similarity", 0.0)

                # 从 reasoning 的 feature_matching_details 中提取针对该 cluster 的相似度
                # Try to find cluster-specific similarity from top_similarities
                for ts in detail.get("top_similarities", []):
                    if ts.get("cluster_id") == ci["cluster_id"]:
                        sim = ts["similarity"]
                        break

                lines.append(
                    f"  **Community [{ci['device_type']}_{ci['cluster_id']}]** "
                    f"(community_sim={ci['similarity_score']:.2f}, "
                    f"perspective_sim={sim:.3f}):"
                )
                lines.append(f"    {cluster_desc}")

            lines.append("")

        return "\n".join(lines)

    def _build_detection_prompt(
        self,
        query_fp: Dict,
        key_perspectives: Dict,
        non_key_perspectives: Dict,
        indicators: Dict,
        community_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        构建 unseen 设备检测的 prompt.
        仅关注 key perspectives 与 community cluster 对应 perspective 的对比和相似度.
        Build the prompt for unseen device detection.
        Focus only on key perspectives vs community cluster perspectives with similarity scores.
        """
        # 提取 community cluster 的 perspective 描述
        # Extract community cluster perspective descriptions
        community_clusters = self._extract_community_perspectives(community_result)

        # 格式化 key perspectives 与 community 的逐一对比
        comparison_text = self._format_key_vs_community(
            key_perspectives, community_clusters
        )

        # community cluster 摘要
        # Community cluster summary
        cluster_summary_lines: List[str] = []
        for ci in community_clusters[:3]:
            matched = "; ".join(ci["matched_features"][:3]) if ci["matched_features"] else "none"
            unmatched = "; ".join(ci["unmatched_features"][:3]) if ci["unmatched_features"] else "none"
            cluster_summary_lines.append(
                f"  - {ci['device_type']}_{ci['cluster_id']} "
                f"(similarity={ci['similarity_score']:.2f}): "
                f"matched=[{matched}]; unmatched=[{unmatched}]"
            )
        cluster_summary = "\n".join(cluster_summary_lines) if cluster_summary_lines else "  No community clusters matched."

        # Tavily web search 辅助厂商归属
        # Tavily web search for vendor attribution
        vendor_search_text = self._tavily_vendor_search(key_perspectives, query_fp)

        prompt = f"""You are an expert IoT device classifier. Determine whether a query device is "unseen" (not belonging to any known RAG device type) based on the comparison between its key perspectives and the community-retrieved cluster perspectives.

## Known Device Types (RAG)
{', '.join(self.rag_devices)}

## Unseen Candidate Types
{', '.join(self.unseen_candidates)}

## Community Cluster Matches
{cluster_summary}

## Key Perspective Comparison (Query Device vs Community Clusters)
For each key perspective (high discriminative power, low Shannon entropy), the query device's feature values are compared against the community cluster's corresponding perspective description, with a similarity score.

{comparison_text}

## Summary Statistics
- Best path matching score: {indicators.get('best_path_score', 0.0):.4f}
- Best matching known type: {indicators.get('best_device_type', 'UNKNOWN')}
- Key perspectives avg similarity: {indicators.get('key_avg_similarity', 0.0):.4f}
- Key perspectives min similarity: {indicators.get('key_min_similarity', 0.0):.4f}

## Web Search Results (Vendor Attribution)
{vendor_search_text}

## Task
Reason step by step:

**Step 1 – Per-perspective verdict**: For each key perspective in the comparison above, judge whether the query device MATCHES or MISMATCHES the best community cluster on that perspective. A perspective_sim < 0.5 strongly indicates MISMATCH. State your verdict and a one-sentence reason.

**Step 2 – Aggregate**: Count how many key perspectives are MATCH vs MISMATCH. If the majority of key perspectives are MISMATCH, the device is very likely unseen.

**Step 3 – Vendor attribution**: Based on the web search results above and any identifiers from the key perspectives (e.g., DNS domains, OUI prefixes, certificate fields, HTTP headers), infer the most likely device vendor.

**Step 4 – Decision**: Based on Steps 1-3 and the summary statistics, estimate unseen_probability, decide is_unseen, and predict the device type (from RAG types if known, from unseen candidates if unseen) and vendor.

Output your step-by-step reasoning, then end with a JSON block:
```json
{{
    "unseen_probability": <float 0.0-1.0>,
    "is_unseen": <true if unseen_probability >= 0.5>,
    "predicted_type": "<device type>",
    "predicted_vendor": "<vendor or 'Unknown'>",
    "confidence": <float 0.0-1.0>
}}
```"""
        return prompt

    # ═════════════════════════════════════════════════════════════════════
    # §5  Model Inference (模型推理)
    # ═════════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def _generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        运行 LLama 推理, 返回生成的文本.
        Run LLama inference and return generated text.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert IoT network device classifier specializing in "
                    "unseen device detection. Always respond with valid JSON as instructed."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ═════════════════════════════════════════════════════════════════════
    # §6  Response Parsing (解析模型输出)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """
        从模型输出中提取 JSON, 支持多种格式的鲁棒解析.
        Extract JSON from model response with robust fallback parsing.
        """
        # 1. fenced JSON block
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 2. JSON containing "unseen_probability"
        for match in re.finditer(
            r"\{[^{}]*\"unseen_probability\"[^{}]*\}", text, re.DOTALL
        ):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        # 3. last { ... } block (可能包含嵌套对象)
        start, end = text.rfind("{"), text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        logging.warning(f"Failed to parse JSON from LLama response: {text[:500]}")
        return {}

    def _validate_predicted_type(self, predicted_type: str) -> str:
        """
        验证并规范化预测的设备类型.
        Validate and normalize the predicted device type.
        """
        if predicted_type in self.all_devices:
            return predicted_type

        # 尝试模糊匹配 / Fuzzy match
        pred_upper = predicted_type.upper().strip().replace(" ", "_")
        for dev in self.all_devices:
            if dev == pred_upper:
                return dev
            if dev.lower() == pred_upper.lower():
                return dev

        # 部分匹配 / Partial match
        for dev in self.all_devices:
            if pred_upper in dev or dev in pred_upper:
                return dev

        return predicted_type

    # ═════════════════════════════════════════════════════════════════════
    # §7  Main Detection Entry (主检测入口)
    # ═════════════════════════════════════════════════════════════════════

    def detect_unseen(
        self,
        reasoning_result: Dict[str, Any],
        local_result: Optional[Dict[str, Any]] = None,
        community_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        主入口: 检测查询设备是否为 unseen 设备.
        Main entry point: detect whether a query device is unseen.

        Args:
            reasoning_result: reasoning_path_retrieval 的输出 (必需).
                              包含 query_fingerprint, path_matching_results, summary.
                              Output from reasoning_path_retrieval (required).
            local_result: local_retrieval 的输出 (可选, 提供统计辅助证据).
                          Output from local_retrieval (optional, for statistical indicators).
            community_result: community_retrieval 的输出 (推荐, 提供 cluster perspective 描述用于对比).
                              Output from community_retrieval (recommended, provides
                              cluster perspective descriptions for key-perspective comparison).

        Returns:
            {
                "unseen_probability": float,   # 设备属于 unseen 的概率
                "is_unseen": bool,             # 是否属于 unseen 设备
                "predicted_type": str,         # 预测的设备类型
                "predicted_vendor": str,       # 预测的设备厂商
                "confidence": float,           # 整体预测置信度
                "key_perspectives": dict,      # 关键 perspective 详情
                "non_key_perspectives": dict,  # 非关键 perspective 详情
                "indicators": dict,            # 统计指标
                "raw_response": str,           # 模型原始输出
                "generation_time_sec": float,
                "total_time_sec": float,
            }
        """
        t0 = time.time()
        logging.info("Starting unseen device detection")

        # 1. 提取关键和非关键 perspectives
        # 1. Extract key and non-key perspectives
        key_perspectives, non_key_perspectives = self.extract_perspectives(
            reasoning_result
        )
        logging.info(
            f"Extracted {len(key_perspectives)} key perspectives, "
            f"{len(non_key_perspectives)} non-key perspectives"
        )

        # 2. 计算统计型 unseen 指标
        # 2. Compute statistical unseen indicators
        indicators = self.compute_unseen_indicators(
            reasoning_result, local_result, key_perspectives, non_key_perspectives
        )
        logging.info(f"Unseen indicators: {json.dumps(indicators, default=str)}")

        # 3. 构建 prompt (聚焦 key perspectives vs community cluster perspectives)
        # 3. Build prompt (focus on key perspectives vs community cluster perspectives)
        query_fp = reasoning_result.get("query_fingerprint", {})
        prompt = self._build_detection_prompt(
            query_fp, key_perspectives, non_key_perspectives, indicators,
            community_result=community_result,
        )

        # 4. LLama 推理
        # 4. Run LLama inference
        t_gen_start = time.time()
        raw_response = self._generate(prompt)
        t_gen = time.time() - t_gen_start
        logging.info(f"LLama generation took {t_gen:.2f}s")

        # 5. 解析响应
        # 5. Parse response
        parsed = self._parse_response(raw_response)

        # 6. 构建结果 (仅 5 个核心字段 + 辅助信息)
        # 6. Build result (5 core fields + auxiliary info)
        unseen_prob = float(parsed.get("unseen_probability", 0.5))
        predicted_type = parsed.get(
            "predicted_type", indicators.get("best_device_type", "UNKNOWN")
        )
        predicted_type = self._validate_predicted_type(predicted_type)

        result = {
            # ── 5 core output fields ──
            "unseen_probability": round(unseen_prob, 4),
            "is_unseen": parsed.get("is_unseen", unseen_prob >= _UNSEEN_THRESHOLD),
            "predicted_type": predicted_type,
            "predicted_vendor": parsed.get("predicted_vendor", "Unknown"),
            "confidence": round(float(parsed.get("confidence", 0.0)), 4),
            # ── auxiliary info ──
            "key_perspectives": key_perspectives,
            "non_key_perspectives": non_key_perspectives,
            "indicators": indicators,
            "raw_response": raw_response,
            "generation_time_sec": round(t_gen, 3),
            "total_time_sec": round(time.time() - t0, 3),
        }

        logging.info(
            f"Unseen detection complete: unseen_prob={result['unseen_probability']:.3f}, "
            f"is_unseen={result['is_unseen']}, type={result['predicted_type']}, "
            f"vendor={result['predicted_vendor']}, conf={result['confidence']:.3f}, "
            f"time={result['total_time_sec']:.1f}s"
        )

        return result

    # ═════════════════════════════════════════════════════════════════════
    # §8  Batch Detection (批量检测)
    # ═════════════════════════════════════════════════════════════════════

    def batch_detect(
        self,
        reasoning_results: List[Dict[str, Any]],
        local_results: Optional[List[Optional[Dict]]] = None,
        community_results: Optional[List[Optional[Dict]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量 unseen 检测.
        Batch unseen detection for multiple devices.

        Args:
            reasoning_results: 多个 reasoning_path_retrieval 输出的列表.
            local_results: 对应的 local_retrieval 输出列表 (可选).
            community_results: 对应的 community_retrieval 输出列表 (可选).

        Returns:
            检测结果列表.
        """
        n = len(reasoning_results)
        if local_results is None:
            local_results = [None] * n
        if community_results is None:
            community_results = [None] * n

        results = []
        for i, (rr, lr, cr) in enumerate(
            zip(reasoning_results, local_results, community_results)
        ):
            ip = rr.get("query_fingerprint", {}).get("ip", f"device_{i}")
            print(f"[{i+1}/{n}] Detecting unseen for {ip}...")
            try:
                result = self.detect_unseen(rr, lr, cr)
                result["ip"] = ip
                results.append(result)
                print(
                    f"  -> unseen_prob={result['unseen_probability']:.3f}, "
                    f"type={result['predicted_type']}, "
                    f"vendor={result['predicted_vendor']}"
                )
            except Exception as e:
                logging.error(f"Error detecting unseen for {ip}: {e}", exc_info=True)
                results.append({"ip": ip, "error": str(e)})
                print(f"  -> ERROR: {e}")

        return results

    # ═════════════════════════════════════════════════════════════════════
    # §9  Training Data Generation (训练数据生成, 用于 LoRA 微调)
    # ═════════════════════════════════════════════════════════════════════

    def generate_training_sample(
        self,
        reasoning_result: Dict[str, Any],
        local_result: Optional[Dict[str, Any]],
        ground_truth_type: str,
        ground_truth_vendor: str = "Unknown",
        community_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成一条训练样本, 用于 SFT / LoRA 微调.
        Generate a training sample for SFT / LoRA fine-tuning.

        创建 (instruction, input, output) 三元组.
        Creates an (instruction, input, output) triple suitable for
        instruction-tuning with LoRA.

        Args:
            reasoning_result: reasoning_path_retrieval 的输出.
            local_result: local_retrieval 的输出.
            ground_truth_type: 真实设备类型标签.
            ground_truth_vendor: 真实厂商标签.
            community_result: community_retrieval 的输出.

        Returns:
            Dict with keys: instruction, input, output, metadata
        """
        key_perspectives, non_key_perspectives = self.extract_perspectives(
            reasoning_result
        )
        indicators = self.compute_unseen_indicators(
            reasoning_result, local_result, key_perspectives, non_key_perspectives
        )
        query_fp = reasoning_result.get("query_fingerprint", {})

        prompt = self._build_detection_prompt(
            query_fp, key_perspectives, non_key_perspectives, indicators,
            community_result=community_result,
        )

        is_unseen = ground_truth_type not in self.rag_devices
        is_in_all = ground_truth_type in self.all_devices

        # 根据真实标签设定 unseen 概率
        # Set unseen probability based on ground truth
        if not is_unseen:
            unseen_prob = 0.05
            confidence = 0.90
        elif is_in_all:
            unseen_prob = 0.90
            confidence = 0.85
        else:
            unseen_prob = 0.95
            confidence = 0.70

        target_output = json.dumps(
            {
                "unseen_probability": unseen_prob,
                "is_unseen": is_unseen,
                "predicted_type": ground_truth_type,
                "predicted_vendor": ground_truth_vendor,
                "confidence": confidence,
            },
            indent=2,
            ensure_ascii=False,
        )

        return {
            "instruction": (
                "You are an expert IoT network device classifier specializing in "
                "unseen device detection. Always respond with valid JSON as instructed."
            ),
            "input": prompt,
            "output": f"```json\n{target_output}\n```",
            "metadata": {
                "ip": query_fp.get("ip", ""),
                "ground_truth_type": ground_truth_type,
                "ground_truth_vendor": ground_truth_vendor,
                "is_unseen": is_unseen,
                "is_in_extended_catalogue": is_in_all,
                "num_key_perspectives": len(key_perspectives),
                "num_nonkey_perspectives": len(non_key_perspectives),
                "heuristic_unseen_score": indicators.get("heuristic_unseen_score", 0.0),
            },
        }

    def batch_generate_training_data(
        self,
        samples: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成训练数据.
        Batch generate training data for fine-tuning.

        Args:
            samples: 样本列表, 每个包含:
                     - reasoning_result: reasoning_path_retrieval 的输出
                     - local_result: local_retrieval 的输出 (可选)
                     - ground_truth_type: 真实设备类型
                     - ground_truth_vendor: 真实厂商 (可选)
            output_path: 输出 JSONL 文件路径 (可选).

        Returns:
            训练样本列表.
        """
        training_data: List[Dict] = []

        for i, sample in enumerate(samples):
            try:
                train_sample = self.generate_training_sample(
                    reasoning_result=sample["reasoning_result"],
                    local_result=sample.get("local_result"),
                    ground_truth_type=sample["ground_truth_type"],
                    ground_truth_vendor=sample.get("ground_truth_vendor", "Unknown"),
                )
                training_data.append(train_sample)

                if (i + 1) % 50 == 0:
                    print(f"Generated {i+1}/{len(samples)} training samples")
            except Exception as e:
                logging.warning(f"Failed to generate sample {i}: {e}")
                continue

        print(f"Total training samples generated: {len(training_data)}")

        # 保存到 JSONL 文件 / Save to JSONL file
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Training data saved to {output_path}")
            logging.info(
                f"Training data saved: {len(training_data)} samples → {output_path}"
            )

        return training_data


# ═════════════════════════════════════════════════════════════════════════════
# §10  CLI Interface (命令行接口)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unseen IoT Device Detection (基于 LLama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--reasoning_file",
        type=str,
        required=True,
        help="Path to reasoning_path_retrieval JSON output file",
    )
    parser.add_argument(
        "--local_file",
        type=str,
        default=None,
        help="Path to local_retrieval JSON output file (optional)",
    )
    parser.add_argument(
        "--community_file",
        type=str,
        default=None,
        help="Path to community_retrieval JSON output file (recommended)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to LLama-3.1-8B-Instruct model (default: auto-detect)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (fine-tuned weights)",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device index (-1=CPU)")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--max_devices",
        type=int,
        default=None,
        help="Max number of devices to process (for debugging)",
    )

    args = parser.parse_args()

    # 加载数据 / Load data
    print(f"Loading reasoning results from {args.reasoning_file}")
    with open(args.reasoning_file, "r", encoding="utf-8") as f:
        reasoning_data = json.load(f)

    local_data = None
    if args.local_file:
        print(f"Loading local results from {args.local_file}")
        with open(args.local_file, "r", encoding="utf-8") as f:
            local_data = json.load(f)

    community_data = None
    if args.community_file:
        print(f"Loading community results from {args.community_file}")
        with open(args.community_file, "r", encoding="utf-8") as f:
            community_data = json.load(f)

    # 初始化检测器 / Initialize detector
    print("Initializing UnseenDeviceDetector...")
    detector = UnseenDeviceDetector(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        gpu=args.gpu,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # 处理数据 / Process data
    if isinstance(reasoning_data, list):
        # 多个设备 / Multiple devices
        max_n = args.max_devices or len(reasoning_data)
        reasoning_list = reasoning_data[:max_n]
        local_list = (
            local_data[:max_n] if isinstance(local_data, list) else [None] * max_n
        )
        community_list = (
            community_data[:max_n] if isinstance(community_data, list) else [None] * max_n
        )
        results = detector.batch_detect(reasoning_list, local_list, community_list)
    else:
        # 单个设备 / Single device
        lr = local_data if isinstance(local_data, dict) else None
        cr = community_data if isinstance(community_data, dict) else None
        results = [detector.detect_unseen(reasoning_data, lr, cr)]

    # 输出结果 (5 core fields + ip/error)
    # Output results (5 core fields + ip/error)
    output_results = []
    for r in results:
        output_results.append(
            {
                "ip": r.get("ip", r.get("indicators", {}).get("ip", "")),
                "unseen_probability": r.get("unseen_probability"),
                "is_unseen": r.get("is_unseen"),
                "predicted_type": r.get("predicted_type"),
                "predicted_vendor": r.get("predicted_vendor"),
                "confidence": r.get("confidence"),
                "error": r.get("error"),
            }
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n=== Detection Results ===")
        for r in output_results:
            print(json.dumps(r, indent=2, ensure_ascii=False))

    # 汇总 / Summary
    unseen_count = sum(1 for r in output_results if r.get("is_unseen"))
    known_count = sum(1 for r in output_results if r.get("is_unseen") is False)
    error_count = sum(1 for r in output_results if r.get("error"))
    print(f"\n=== Summary ===")
    print(f"Total: {len(output_results)}, Unseen: {unseen_count}, "
          f"Known: {known_count}, Errors: {error_count}")
s