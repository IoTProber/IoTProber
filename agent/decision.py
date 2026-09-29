"""
agent/decision.py

IoT Device Type Decision Agent.
Uses LangChain to build two independent classification agents (Gemini + Claude),
each equipped with three RAG retrieval tools. Joint voting by confidence score
determines the final device type and vendor prediction.
"""

import os
import sys
import json
import re
import logging
import time
from threading import RLock
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from util import load_all_dev_labels, convert_json_from_str
from path_config import (
    DECISION_LOG_FILE,
    LLM_CONFIG_FILE,
    LOCAL_DATA_DIR,
    PREDICTION_RESULT_DIR,
    QUERY_DB_DIR,
    ROOT_DIR,
    VALIDATION_DIR,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=DECISION_LOG_FILE,
    filemode="a",
)

# ── path constants ────────────────────────────────────────────────────────────
_BASE = ROOT_DIR
_LOCAL_RAW = LOCAL_DATA_DIR
_VAL_PATH = VALIDATION_DIR
_QDB_PATH = QUERY_DB_DIR
_RES_PATH = PREDICTION_RESULT_DIR
_CFG_PATH = LLM_CONFIG_FILE

# ── module-level state shared with @tool closures ─────────────────────────────
_dev_labels: List[str] = []


# ═════════════════════════════════════════════════════════════════════════════
# §1  Retrieval Tool Definitions
# ═════════════════════════════════════════════════════════════════════════════

def _lookup_qdb(sub: str, ip: str) -> Dict:
    """
    Search query_db/{sub}/{DEV}_{sub}.json for an entry whose
    query_fingerprint.ip matches *ip*.  Returns the raw entry on hit.
    """
    db_dir = os.path.join(_QDB_PATH, sub)
    runtime = globals().get("_retrieval_runtime")
    context = runtime._context.get(str(ip), {}) if runtime is not None else {}
    context_device = context.get("device_name")
    devices = [context_device] if context_device else list(_dev_labels)
    for dev in devices:
        fp = os.path.join(db_dir, f"{dev}_{sub}.json")
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            if str(entry.get("query_fingerprint", {}).get("ip", "")) == str(ip):
                if context:
                    from retrieval import cache_record_matches
                    if not cache_record_matches(
                        entry, context.get("fingerprint", {}), runtime.top_k
                    ):
                        continue
                return {"status": "found", "entry": entry}
    return {"status": "not_found", "ip": ip}


def _local_section(ip: str) -> Dict:
    """
    Local embedding-based retrieval section: top-k most similar known IoT devices
    to the query device, by comparing multi-perspective network fingerprint
    embeddings stored in the vector database.
    Returns device_type + cosine similarity scores.
    """
    hit = _lookup_qdb("local", ip)
    if hit["status"] == "not_found":
        return hit

    entry = hit["entry"]
    return {
        "status": "found",
        "top_k": entry.get("top_k", 5),
        "similar_devices": entry.get("similar_devices", []),
        "confidence_score": entry.get("confidence_score"),
        "missing_perspectives": entry.get("missing_perspectives", []),
        "total_compared": entry.get("total_compared", 0),
    }


@tool
def local_retrieval(ip: str) -> str:
    """
    Local embedding-based retrieval: retrieve the top-k most similar known IoT
    devices to the query device by comparing multi-perspective network fingerprint
    embeddings stored in the vector database.

    Returns top-k similar devices with device_type and cosine similarity scores.
    Call this tool FIRST to identify strong candidate device types.

    Args:
        ip: IP address of the query device.
    """
    return json.dumps(_local_section(ip), ensure_ascii=False, indent=2)


def _community_section(ip: str) -> Dict:
    """
    Community / cluster-level retrieval section: which device behaviour clusters
    the query device belongs to (based on the similar IPs from local retrieval).
    Returns matched cluster common-pattern reports and per-cluster similarity scores.
    """
    hit = _lookup_qdb("community", ip)
    if hit["status"] == "not_found":
        return hit

    entry = hit["entry"]
    matched = entry.get("matched_clusters", [])
    trimmed = []
    for c in matched[:5]:
        report = c.get("report", {})
        if isinstance(report, dict):
            raw_patterns = report.get("common_patterns", {})
            patterns = (
                {k: str(v)[:300] for k, v in list(raw_patterns.items())[:8]}
                if isinstance(raw_patterns, dict)
                else str(raw_patterns)[:500]
            )
        else:
            patterns = str(report)[:500]

        trimmed.append(
            {
                "device_type":       c.get("device_type"),
                "cluster_id":        c.get("cluster_id"),
                "similarity_score":  c.get("similarity_score"),
                "report_source":     c.get("report_source"),
                "common_patterns":   patterns,
                "matched_features":  c.get("matched_features", [])[:3],
                "unmatched_features": c.get("unmatched_features", [])[:3],
            }
        )

    return {
        "status":         "found",
        "total_clusters": entry.get("total_clusters", len(matched)),
        "matched_clusters": trimmed,
        "unavailable_clusters": entry.get("unavailable_clusters", [])[:5],
    }


@tool
def community_retrieval(ip: str) -> str:
    """
    Community / cluster-level retrieval: identifies which device behaviour clusters
    the query device belongs to, based on the similar IPs found in local retrieval.
    Returns matched cluster common-pattern reports and per-cluster similarity scores.

    Call this tool SECOND to obtain cluster-level contextual evidence.

    Args:
        ip: IP address of the query device.
    """
    return json.dumps(_community_section(ip), ensure_ascii=False, indent=2)


def _reasoning_section(ip: str) -> Dict:
    """
    Reasoning path retrieval section: the key discriminative features that place
    the query device into a particular cluster, via Shannon entropy-based feature
    importance. Returns path matching scores and weighted per-feature similarity
    breakdowns.
    """
    hit = _lookup_qdb("reasoning", ip)
    if hit["status"] == "not_found":
        return hit

    entry = hit["entry"]
    path_results = entry.get("path_matching_results", [])
    trimmed = []
    for pr in path_results[:3]:
        ci = pr.get("cluster_info", {})
        details = sorted(
            pr.get("feature_matching_details", []),
            key=lambda x: x.get("importance_score", 0),
            reverse=True,
        )[:4]
        trimmed.append(
            {
                "cluster_key":        pr.get("cluster_key"),
                "device_type":        ci.get("device_type"),
                "path_matching_score": pr.get("path_matching_score"),
                "evidence_coverage": pr.get("evidence_coverage"),
                "important_features": pr.get("important_features", [])[:5],
                "top_feature_scores": [
                    {
                        "feature":        fd.get("feature_name"),
                        "importance":     fd.get("importance_score"),
                        "matching_score": fd.get("feature_matching_score"),
                    }
                    for fd in details
                ],
            }
        )

    summary = entry.get("summary") or {}
    top_cluster = summary.get("top_cluster") or {}
    return {
        "status":               "found",
        "path_matching_results": trimmed,
        "top_cluster_key":      top_cluster.get("cluster_key"),
    }


@tool
def reasoning_path_retrieval(ip: str) -> str:
    """
    Reasoning path retrieval: analyses the key discriminative features that place
    the query device into a particular cluster, using Shannon entropy-based feature
    importance scoring.  Returns path matching scores and weighted per-feature
    similarity breakdowns.

    Call this tool LAST to understand the most discriminative evidence.

    Args:
        ip: IP address of the query device.
    """
    return json.dumps(_reasoning_section(ip), ensure_ascii=False, indent=2)


_TOOLS = [local_retrieval, community_retrieval, reasoning_path_retrieval]


# ═════════════════════════════════════════════════════════════════════════════
# §1b  Configurable Retrieval Tool Runtime (for the LangGraph decision branches)
# ═════════════════════════════════════════════════════════════════════════════
#
# The LangGraph decision LLM nodes call ONE unified retrieval tool. Which of the
# three retrieval algorithms it actually runs is controlled by the CLI flags
# (--local / --community / --reasoning); when none are supplied the default is
# "all". The runtime lazily runs the live MultiLevelRetrieval algorithms for the
# enabled levels (skipping any already cached in query_db), then returns the same
# trimmed evidence sections consumed by the legacy tools.

class RetrievalToolRuntime:
    """
    可配置的检索工具运行时: 按 (--local / --community / --reasoning) 选择要运行的检索算法, 默认全部.
    Configurable retrieval runtime shared by the LangGraph decision branches.
    """

    def __init__(
        self,
        whether_local: bool = True,
        whether_community: bool = True,
        whether_reasoning: bool = True,
        top_k: int = 5,
        llm_type: str = "CLAUDE",
        retrieval_agent: Any = None,
    ):
        # 若三者都未开启, 默认运行全部 / default to all when nothing is enabled
        if not (whether_local or whether_community or whether_reasoning):
            whether_local = whether_community = whether_reasoning = True
        self.whether_local = whether_local
        self.whether_community = whether_community
        self.whether_reasoning = whether_reasoning
        self.top_k = top_k
        self.llm_type = llm_type
        self.retrieval_agent = retrieval_agent
        self._context: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    # ── per-IP context (device label + fingerprint) ──
    def register(self, ip: str, device_name: str, fingerprint: Dict[str, Any]) -> None:
        with self._lock:
            self._context[str(ip)] = {
                "device_name": device_name,
                "fingerprint": fingerprint,
            }
            global _dev_labels
            if device_name and device_name not in _dev_labels:
                _dev_labels = list(_dev_labels) + [device_name]

    def _ensure_retrieved(self, ip: str, ctx: Dict[str, Any]) -> None:
        """Run the enabled retrieval algorithms for levels not yet cached in query_db."""
        if self.retrieval_agent is None:
            return
        device_name = ctx["device_name"]
        fingerprint = ctx["fingerprint"]
        existing_local, existing_community, existing_reasoning = (
            self.retrieval_agent.load_retrieval_result_by_type(
                ip, device_name, query_fingerprint=fingerprint, top_k=self.top_k
            )
        )
        need_local = self.whether_local and existing_local is None
        need_community = self.whether_community and existing_community is None
        need_reasoning = self.whether_reasoning and existing_reasoning is None
        if not (need_local or need_community or need_reasoning):
            return
        # community / reasoning cascade requires local + community to exist
        run_local = need_local
        run_community = need_community or (need_reasoning and existing_community is None)
        with self._lock:
            self.retrieval_agent.run_retrieval_algorithm(
                test_fingerprint=fingerprint,
                top_k=self.top_k,
                whether_local=run_local,
                whether_community=run_community,
                whether_reasoning=need_reasoning,
                local_result=existing_local,
                community_result=existing_community,
                llm_type=self.llm_type,
                device_name=device_name,
            )
            self.retrieval_agent.clear_history()

    def ensure(self, ip: str) -> None:
        """Public helper: run enabled retrieval algorithms for *ip* if not cached."""
        ctx = self._context.get(str(ip))
        if ctx is not None:
            self._ensure_retrieved(str(ip), ctx)

    def run(self, ip: str) -> Dict[str, Any]:
        ip = str(ip)
        ctx = self._context.get(ip)
        if ctx is not None:
            try:
                self._ensure_retrieved(ip, ctx)
            except Exception as exc:  # pragma: no cover - live retrieval is best-effort
                logging.warning("Live retrieval failed for %s: %s", ip, exc)

        combined: Dict[str, Any] = {"ip": ip, "levels_run": []}
        if self.whether_local:
            combined["local_entity_retrieval"] = _local_section(ip)
            combined["levels_run"].append("local")
        if self.whether_community:
            combined["community_retrieval"] = _community_section(ip)
            combined["levels_run"].append("community")
        if self.whether_reasoning:
            combined["reasoning_path_retrieval"] = _reasoning_section(ip)
            combined["levels_run"].append("reasoning")
        return combined


# module-level runtime consulted by the configurable tool closure
_retrieval_runtime: Optional[RetrievalToolRuntime] = None


def set_retrieval_runtime(runtime: Optional[RetrievalToolRuntime]) -> None:
    """Register the runtime that the configurable retrieval tool should use."""
    global _retrieval_runtime
    _retrieval_runtime = runtime


@tool
def configurable_multi_level_retrieval(ip: str) -> str:
    """
    Multi-Level Retrieval — the single unified RAG tool for the Decision Agent.
    It runs the enabled retrieval algorithms (local-entity, community, and/or
    reasoning-path, selected via the --local/--community/--reasoning flags,
    defaulting to all) for the query device and returns their combined evidence.

    You MUST call this tool exactly once, before any analysis, and base your
    reasoning on its result.

    Args:
        ip: IP address of the query device.
    """
    runtime = _retrieval_runtime
    if runtime is not None:
        combined = runtime.run(ip)
    else:
        combined = {
            "ip": ip,
            "local_entity_retrieval": _local_section(ip),
            "community_retrieval": _community_section(ip),
            "reasoning_path_retrieval": _reasoning_section(ip),
        }
    return json.dumps(combined, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# §1c  Shared helpers for the LangGraph decision branches
# ═════════════════════════════════════════════════════════════════════════════

def extract_decision_json(text: str) -> Dict[str, Any]:
    """Module-level wrapper around the robust JSON extractor."""
    return DecisionAgent._extract_json(text)


def normalize_decision(parsed: Dict[str, Any], llm_name: str, raw_output: str) -> Dict[str, Any]:
    """Normalise a parsed LLM answer into the canonical classification dict."""
    parsed = dict(parsed or {})
    parsed.setdefault("device_type", "UNKNOWN")
    parsed.setdefault("device_type_reason", "")
    parsed.setdefault("vendor", "Unknown")
    parsed.setdefault("vendor_reason", "")
    def bounded(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(np.clip(number, 0.0, 1.0)) if np.isfinite(number) else 0.0

    legacy_confidence = bounded(parsed.get("confidence", 0.0))
    parsed["type_confidence"] = bounded(parsed.get("type_confidence", legacy_confidence))
    parsed["vendor_confidence"] = bounded(parsed.get("vendor_confidence", legacy_confidence))
    parsed["confidence"] = parsed["type_confidence"]  # backward-compatible alias
    parsed["device_type"] = str(parsed["device_type"]).upper().strip()
    parsed["vendor"] = str(parsed.get("vendor", "Unknown"))
    parsed["device_type_reason"] = str(parsed.get("device_type_reason", ""))
    parsed["vendor_reason"] = str(parsed.get("vendor_reason", ""))
    parsed["llm"] = llm_name
    parsed["full_response"] = raw_output
    return parsed


def joint_vote(gemini: Dict[str, Any], claude: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level wrapper around the DecisionAgent joint-voting strategy."""
    return DecisionAgent._joint_vote(gemini, claude)


# ═════════════════════════════════════════════════════════════════════════════
# §2  Prompt Templates
# ═════════════════════════════════════════════════════════════════════════════

def _candidate_device_types() -> str:
    """Candidate types for the decision prompt, from config/rag_devices.json.

    The list used to be hardcoded to 9 types, which silently excluded ALARM
    and CONTROLLER — the LLM could then never classify them correctly no
    matter how strong the retrieval evidence was.
    """
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "rag_devices.json",
    )
    try:
        with open(cfg_path) as fh:
            types = json.load(fh)["IoT"]
    except (OSError, ValueError, KeyError):
        types = [
            "ALARM", "BUILDING_AUTOMATION", "CAMERA", "CONTROLLER", "MEDICAL",
            "NAS", "NVR", "POWER_METER", "PRINTER", "ROUTER", "SCADA",
        ]
    return " | ".join(types)


_AGENT_SYSTEM = """\
You are an expert IoT network device classifier with deep knowledge of network
traffic fingerprinting. Your mission is to identify the **device type** and
**manufacturer/vendor** of an unknown IoT device.

## Candidate Device Types
{candidate_types}

## Your Workflow
Depending on the runtime configuration, you either have ONE unified retrieval
tool (configurable_multi_level_retrieval) that returns all enabled retrieval
levels in a single call, or THREE separate tools to call in order:
  1. local_retrieval(ip)           – get top-k similar known devices
  2. community_retrieval(ip)       – get cluster-level behavioral patterns
  3. reasoning_path_retrieval(ip)  – get discriminative feature importance

Collect ALL retrieval evidence for the query IP (local-entity, community and
reasoning-path levels) before answering. After collecting the tool results,
reason step by step through the Chain-of-Thought framework below, then emit
your final JSON answer.

## Chain-of-Thought Reasoning Framework

**[Step 1] Hardware & Software Fingerprint**
Analyse hw-vendors, hw-products, sw-vendors, sw-products, sw-versions.
What manufacturer/product identifiers are present?

**[Step 2] Network Service Profile**
Examine service-distribution (open ports). Characteristic patterns:
  - NVR / Camera → RTSP-554, HTTP-8000, ONVIF
  - Router       → Telnet-23, SSH-22, HTTP-80/443
  - Printer      → IPP-631, JetDirect-9100, LPD-515
  - NAS          → SMB-445, NFS-2049, FTP-21
  - SCADA        → Modbus-502, DNP3-20000, BACnet-47808
  - Medical      → HL7/DICOM, HTTP-8080
  - Power Meter  → DLMS/COSEM, IEC 61850
  - Alarm        → intrusion/burglar-alarm panels (Bosch, Honeywell, DMP,
                   Axis, Intruder), embedded web UI on 80/443/8080
  - Controller   → PLC / building controllers / I/O units (BACnet, KNX,
                   Niagara, Tridium), vendor firmware web frontends

**[Step 3] TLS / Certificate Evidence**
Analyse cert-subjects, cert-issuers, tls-versions.
Self-signed vs CA-signed? Subject organisation name?

**[Step 4] HTTP Behavioural Signature**
Analyse http-bodys (HTML content), http-tags (title/meta), http-favicon-hashes.
What type of web interface does this suggest?

**[Step 5] ASN & Location Context**
Examine as-name, as-country_code, whois-network-name.
Consumer ISP → likely endpoint device; Cloud ASN → possible scanner/honeypot.

**[Step 6] Local Retrieval Evidence**
Review top-k similar devices. Dominant device type? Similarity scores?
`similarity_score` is a linear weighted average of per-perspective cosine
similarities and is always in [-1, 1]. Also inspect `confidence_score` and
`missing_perspectives`; sparse evidence must reduce confidence, not be treated as
semantic contradiction. (>0.85 = very strong, 0.70–0.85 = moderate, <0.70 = weak)

**[Step 7] Community Cluster Evidence**
Review matched clusters: similarity scores, common patterns, matched/unmatched
features. Entries listed under `unavailable_clusters` are explicitly NOT
evidence and must not affect the classification.

**[Step 8] Reasoning Path Evidence**
Review path matching scores and feature importance. Which features are most
discriminative and do they align with the query device? Reduce confidence when
`evidence_coverage` is low.

**[Step 9] Synthesis & Confidence**
Weigh all evidence. Assign confidence:
  0.90–1.00 – reserved for independently calibrated, overwhelming evidence
  0.75–0.89 – strong evidence with minor gaps
  0.60–0.74 – moderate evidence, some conflicting signals
  0.40–0.59 – weak evidence, notable uncertainty
  0.00–0.39 – very uncertain, mostly noise

## Required Final Output
After your chain-of-thought, output a **single JSON block** (no trailing text):
```json
{
  "step1_hardware":      "<analysis>",
  "step2_services":      "<analysis>",
  "step3_certs":         "<analysis>",
  "step4_http":          "<analysis>",
  "step5_asn":           "<analysis>",
  "step6_local":         "<analysis>",
  "step7_community":     "<analysis>",
  "step8_reasoning":     "<analysis>",
  "step9_synthesis":     "<analysis>",
  "device_type":         "<ONE type from the candidate list>",
  "device_type_reason":  "<Concise explanation of the key evidence that led to this device type decision>",
  "vendor":              "<manufacturer name or 'Unknown'>",
  "vendor_reason":       "<Concise explanation of the key evidence that identified this specific vendor>",
  "type_confidence":     0.0,
  "vendor_confidence":   0.0
}
```
"""

_AGENT_HUMAN = """\
Classify the IoT device with IP address: {ip}

Raw network fingerprint (excluding null fields):
```json
{fingerprint}
```

Call the available retrieval tool(s) for this IP to collect all retrieval
evidence, then apply the 9-step reasoning framework, and output your final JSON
classification.
"""


# ═════════════════════════════════════════════════════════════════════════════
# §3  DecisionAgent
# ═════════════════════════════════════════════════════════════════════════════

class DecisionAgent:
    """
    LangChain-based IoT device classification agent.

    Two independent LangChain AgentExecutors (Gemini + Claude) each:
      - call three RAG retrieval tools (local / community / reasoning_path)
      - perform 9-step chain-of-thought reasoning
      - output {device_type, vendor, confidence}

    Joint voting selects the result with the higher confidence score.
    When both LLMs agree on device_type, confidence is boosted slightly.
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self):
        os.makedirs(_RES_PATH, exist_ok=True)

        with open(_CFG_PATH, "r") as fh:
            self.cfg = json.load(fh)

        self.dev_labels: List[str] = load_all_dev_labels() or []
        global _dev_labels
        _dev_labels = self.dev_labels

        # LangChain LLM clients (both accessed via OpenAI-compatible endpoints)
        self.gemini_llm = ChatOpenAI(
            api_key=self.cfg["GEMINI"]["API_KEY"],
            base_url=self.cfg["GEMINI"]["BASE_URL"],
            model=self.cfg["GEMINI"]["MODEL"],
            temperature=0.3,
            max_tokens=4096,
        )
        self.claude_llm = ChatOpenAI(
            api_key=self.cfg["CLAUDE"]["API_KEY"],
            base_url=self.cfg["CLAUDE"]["BASE_URL"],
            model=self.cfg["CLAUDE"]["MODEL"],
            temperature=0.3,
            max_tokens=4096,
        )

        # Build independent LangChain agents for each LLM
        self._gemini_executor = self._build_agent(self.gemini_llm)
        self._claude_executor  = self._build_agent(self.claude_llm)

        logging.info("DecisionAgent initialised (Gemini + Claude).")

    # ── agent construction ────────────────────────────────────────────────────

    def _build_agent(self, llm: ChatOpenAI) -> AgentExecutor:
        """
        Wrap *llm* in a LangChain AgentExecutor with the three retrieval tools.
        Uses create_openai_tools_agent which supports function-calling APIs.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",  _AGENT_SYSTEM),
                ("human",   _AGENT_HUMAN),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_openai_tools_agent(llm=llm, tools=_TOOLS, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=_TOOLS,
            verbose=False,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    # ── fingerprint loader ────────────────────────────────────────────────────

    def _load_fingerprint(self, ip: str, dev_type: str) -> Optional[Dict]:
        """
        Load the raw fingerprint row for *ip* from
        evaluation/validation/test_{dev_type}_1.csv.
        Returns None if not found.
        """
        csv_path = os.path.join(_VAL_PATH, f"test_{dev_type}_1.csv")
        if not os.path.exists(csv_path):
            logging.warning("Test CSV not found: %s", csv_path)
            return None
        df = pd.read_csv(csv_path, low_memory=False)
        rows = df[df["ip"].astype(str) == str(ip)]
        if rows.empty:
            logging.warning("IP %s not found in %s", ip, csv_path)
            return None
        raw = rows.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in raw.items()}

    # ── fingerprint formatter ─────────────────────────────────────────────────

    @staticmethod
    def _fmt_fp(fp: Dict) -> str:
        """Return a compact JSON representation of the fingerprint (no nulls/IP)."""
        display = {
            k: v
            for k, v in fp.items()
            if k != "ip" and v is not None and str(v) not in ("nan", "None", "")
        }
        return json.dumps(display, indent=2, ensure_ascii=False)

    # ── single LLM agent run ──────────────────────────────────────────────────

    def _run_agent(
        self,
        executor: AgentExecutor,
        llm_name: str,
        ip: str,
        fingerprint: Dict,
    ) -> Dict:
        """
        Invoke the LangChain AgentExecutor for one LLM.
        The agent calls all three retrieval tools, then emits a JSON answer.
        Returns a normalised classification dict.
        """
        try:
            response = executor.invoke(
                {"ip": ip, "fingerprint": self._fmt_fp(fingerprint)}
            )
            raw_output: str = response.get("output", "")
            parsed = self._extract_json(raw_output)
            parsed.setdefault("device_type", "UNKNOWN")
            parsed.setdefault("device_type_reason", "")
            parsed.setdefault("vendor", "Unknown")
            parsed.setdefault("vendor_reason", "")
            parsed["confidence"]        = float(parsed.get("confidence", 0.0))
            parsed["device_type"]       = str(parsed["device_type"]).upper().strip()
            parsed["vendor"]            = str(parsed.get("vendor", "Unknown"))
            parsed["device_type_reason"] = str(parsed.get("device_type_reason", ""))
            parsed["vendor_reason"]      = str(parsed.get("vendor_reason", ""))
            parsed["llm"]               = llm_name
            parsed["full_response"]     = raw_output
            return parsed

        except Exception as exc:
            logging.error("%s agent error: %s", llm_name, exc, exc_info=True)
            return {
                "llm":          llm_name,
                "device_type":  "UNKNOWN",
                "vendor":       "Unknown",
                "confidence":   0.0,
                "step9_synthesis": f"Agent error: {exc}",
                "error":        str(exc),
            }

    # ── JSON extractor ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """
        Robustly extract the last JSON object containing 'device_type' from *text*.
        Tries (in order):
          1. ```json ... ``` fenced block
          2. Innermost { } containing "device_type"
          3. Last { ... } block in the text
        """
        # 1. fenced code block
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 2. { ... } containing "device_type"
        for match in re.finditer(r"\{[^{}]*\"device_type\"[^{}]*\}", text, re.DOTALL):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        # 3. last { ... } block (may be a nested object)
        start, end = text.rfind("{"), text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")

    # ── joint voting ──────────────────────────────────────────────────────────

    @staticmethod
    def _joint_vote(gemini: Dict, claude: Dict) -> Dict:
        """
        Select the higher-confidence result while applying a conservative policy.

        LLM self-reported confidence is not calibrated. Agreement therefore does
        not receive an artificial boost, and disagreement cannot be presented as
        high-confidence evidence.
        """
        g_conf = float(np.clip(float(gemini.get("type_confidence", gemini.get("confidence", 0.0))), 0.0, 1.0))
        c_conf = float(np.clip(float(claude.get("type_confidence", claude.get("confidence", 0.0))), 0.0, 1.0))

        winner, loser = (gemini, claude) if g_conf >= c_conf else (claude, gemini)
        agree = winner.get("device_type") == loser.get("device_type")

        if agree:
            agreement_cap = 0.35 if winner.get("device_type") == "UNKNOWN" else 0.89
            final_conf = min(agreement_cap, (g_conf + c_conf) / 2)
            confidence_policy = "agreement_mean_capped"
        else:
            final_conf = min(0.69, max(g_conf, c_conf))
            confidence_policy = "disagreement_capped"

        g_vendor_conf = float(np.clip(float(gemini.get("vendor_confidence", g_conf)), 0.0, 1.0))
        c_vendor_conf = float(np.clip(float(claude.get("vendor_confidence", c_conf)), 0.0, 1.0))
        vendor_winner, vendor_loser = (
            (gemini, claude) if g_vendor_conf >= c_vendor_conf else (claude, gemini)
        )
        vendor_agree = (
            str(vendor_winner.get("vendor", "")).strip().casefold()
            == str(vendor_loser.get("vendor", "")).strip().casefold()
        )
        if vendor_agree:
            vendor_cap = (
                0.35
                if str(vendor_winner.get("vendor", "")).strip().casefold()
                in {"", "unknown"}
                else 0.89
            )
            final_vendor_conf = min(vendor_cap, (g_vendor_conf + c_vendor_conf) / 2)
            vendor_confidence_policy = "agreement_mean_capped"
        else:
            final_vendor_conf = min(0.69, max(g_vendor_conf, c_vendor_conf))
            vendor_confidence_policy = "disagreement_capped"

        return {
            "final_device_type":        winner["device_type"],
            "final_device_type_reason": winner.get("device_type_reason", ""),
            "final_vendor":             vendor_winner["vendor"],
            "final_vendor_reason":      vendor_winner.get("vendor_reason", ""),
            "final_confidence":         round(final_conf, 4),
            "final_vendor_confidence":  round(final_vendor_conf, 4),
            "winning_llm":              winner["llm"],
            "llm_agreement":            agree,
            "confidence_policy":        confidence_policy,
            "winning_vendor_llm":       vendor_winner["llm"],
            "vendor_llm_agreement":     vendor_agree,
            "vendor_confidence_policy": vendor_confidence_policy,
            "gemini": {
                "device_type":        gemini.get("device_type"),
                "device_type_reason": gemini.get("device_type_reason", ""),
                "vendor":             gemini.get("vendor"),
                "vendor_reason":      gemini.get("vendor_reason", ""),
                "confidence":         g_conf,
                "type_confidence":    g_conf,
                "vendor_confidence":  g_vendor_conf,
                "synthesis":          gemini.get("step9_synthesis", ""),
            },
            "claude": {
                "device_type":        claude.get("device_type"),
                "device_type_reason": claude.get("device_type_reason", ""),
                "vendor":             claude.get("vendor"),
                "vendor_reason":      claude.get("vendor_reason", ""),
                "confidence":         c_conf,
                "type_confidence":    c_conf,
                "vendor_confidence":  c_vendor_conf,
                "synthesis":          claude.get("step9_synthesis", ""),
            },
        }

    # ── public: classify one IP ───────────────────────────────────────────────

    def classify(self, ip: str, dev_type: str) -> Dict:
        """
        Classify a single device IP using Gemini + Claude joint voting.

        Args:
            ip:       IP address of the query device.
            dev_type: True device type label (used to locate test CSV).

        Returns:
            Classification result dict including predicted_device_type,
            predicted_vendor, final_confidence, per-LLM outputs, and metadata.
        """
        logging.info("classify  ip=%s  dev_type=%s", ip, dev_type)
        t0 = time.time()

        fp = self._load_fingerprint(ip, dev_type)
        if fp is None:
            return {"error": f"Fingerprint not found for ip={ip}, dev_type={dev_type}"}

        # Run both LangChain agents independently (each calls all 3 tools + reasons)
        gemini_result = self._run_agent(self._gemini_executor, "GEMINI", ip, fp)
        logging.info(
            "Gemini → %s  vendor=%s  conf=%.3f",
            gemini_result["device_type"], gemini_result["vendor"],
            gemini_result["confidence"],
        )

        claude_result = self._run_agent(self._claude_executor, "CLAUDE", ip, fp)
        logging.info(
            "Claude → %s  vendor=%s  conf=%.3f",
            claude_result["device_type"], claude_result["vendor"],
            claude_result["confidence"],
        )

        voting = self._joint_vote(gemini_result, claude_result)

        result = {
            "ip":                        ip,
            "true_device_type":          dev_type,
            "predicted_device_type":     voting["final_device_type"],
            "device_type_reason":        voting["final_device_type_reason"],
            "predicted_vendor":          voting["final_vendor"],
            "vendor_reason":             voting["final_vendor_reason"],
            "final_confidence":          voting["final_confidence"],
            "final_vendor_confidence":   voting["final_vendor_confidence"],
            "winning_llm":               voting["winning_llm"],
            "llm_agreement":             voting["llm_agreement"],
            "confidence_policy":         voting.get("confidence_policy"),
            "winning_vendor_llm":        voting.get("winning_vendor_llm"),
            "vendor_llm_agreement":      voting.get("vendor_llm_agreement"),
            "vendor_confidence_policy":  voting.get("vendor_confidence_policy"),
            "gemini":                    voting["gemini"],
            "claude":                    voting["claude"],
            "elapsed_sec":               round(time.time() - t0, 2),
        }

        logging.info(
            "Decision  %s  conf=%.3f  winner=%s  agree=%s",
            voting["final_device_type"], voting["final_confidence"],
            voting["winning_llm"], voting["llm_agreement"],
        )
        return result

    # ── public: batch classify ────────────────────────────────────────────────

    def run_batch(
        self,
        dev_type: str,
        max_samples: Optional[int] = None,
        ip_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Classify a batch of test devices for *dev_type*.

        Reads IPs from evaluation/validation/test_{dev_type}_1.csv unless
        *ip_list* is provided.  Results are auto-saved after processing.

        Args:
            dev_type:    Device type label (e.g. 'NVR', 'ROUTER').
            max_samples: Maximum number of IPs to process (None = all).
            ip_list:     Explicit IP list (overrides test CSV).

        Returns:
            List of per-IP classification result dicts.
        """
        logging.info("run_batch  dev_type=%s  max=%s", dev_type, max_samples)

        if ip_list is None:
            csv_path = os.path.join(_VAL_PATH, f"test_{dev_type}_1.csv")
            if not os.path.exists(csv_path):
                logging.error("Test CSV not found: %s", csv_path)
                return []
            ip_list = pd.read_csv(csv_path, low_memory=False)["ip"].astype(str).tolist()

        if max_samples:
            ip_list = ip_list[:max_samples]

        print(f"[DecisionAgent] Classifying {len(ip_list)} IPs for {dev_type} …")
        results: List[Dict] = []
        for i, ip in enumerate(ip_list, 1):
            print(f"  [{i}/{len(ip_list)}] {ip}")
            results.append(self.classify(ip, dev_type))

        self._save(dev_type, results)
        return results

    # ── public: convenience entry-point ──────────────────────────────────────

    def run(
        self,
        dev_type: Optional[str] = None,
        ip: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Any:
        """
        Convenience entry-point:

        - run(dev_type='NVR', ip='1.2.3.4') → classify one IP, save result.
        - run(dev_type='NVR')               → classify all IPs in test CSV.
        - run()                             → classify all available device types.
        - run(max_samples=5)                → classify first 5 IPs per type.
        """
        if dev_type and ip:
            result = self.classify(ip, dev_type)
            self._save(dev_type, [result])
            return result
        elif dev_type:
            return self.run_batch(dev_type, max_samples=max_samples)
        else:
            all_results: Dict[str, List[Dict]] = {}
            for dev in self.dev_labels:
                csv_path = os.path.join(_VAL_PATH, f"test_{dev}_1.csv")
                if os.path.exists(csv_path):
                    all_results[dev] = self.run_batch(dev, max_samples=max_samples)
            return all_results

    # ── result persistence ────────────────────────────────────────────────────

    def _save(self, dev_type: str, results: List[Dict]) -> None:
        """
        Serialise *results* into two separate JSON files:
          - {dev_type}_type_prediction.json   – device type predictions + reasoning
          - {dev_type}_vendor_prediction.json – vendor predictions + reasoning
        """
        type_records = [
            {
                "ip":                    r.get("ip"),
                "true_device_type":      r.get("true_device_type"),
                "predicted_device_type": r.get("predicted_device_type"),
                "device_type_reason":    r.get("device_type_reason", ""),
                "confidence":            r.get("final_confidence"),
                "winning_llm":           r.get("winning_llm"),
                "llm_agreement":         r.get("llm_agreement"),
                "gemini_device_type":    r.get("gemini", {}).get("device_type"),
                "gemini_reason":         r.get("gemini", {}).get("device_type_reason", ""),
                "gemini_confidence":     r.get("gemini", {}).get("confidence"),
                "claude_device_type":    r.get("claude", {}).get("device_type"),
                "claude_reason":         r.get("claude", {}).get("device_type_reason", ""),
                "claude_confidence":     r.get("claude", {}).get("confidence"),
                "elapsed_sec":           r.get("elapsed_sec"),
            }
            for r in results
        ]

        vendor_records = [
            {
                "ip":                r.get("ip"),
                "true_device_type":  r.get("true_device_type"),
                "predicted_vendor":  r.get("predicted_vendor"),
                "vendor_reason":     r.get("vendor_reason", ""),
                "confidence":        r.get("final_confidence"),
                "winning_llm":       r.get("winning_llm"),
                "llm_agreement":     r.get("llm_agreement"),
                "gemini_vendor":     r.get("gemini", {}).get("vendor"),
                "gemini_reason":     r.get("gemini", {}).get("vendor_reason", ""),
                "gemini_confidence": r.get("gemini", {}).get("confidence"),
                "claude_vendor":     r.get("claude", {}).get("vendor"),
                "claude_reason":     r.get("claude", {}).get("vendor_reason", ""),
                "claude_confidence": r.get("claude", {}).get("confidence"),
                "elapsed_sec":       r.get("elapsed_sec"),
            }
            for r in results
        ]

        for suffix, records in (
            ("type_prediction",   type_records),
            ("vendor_prediction", vendor_records),
        ):
            out = os.path.join(_RES_PATH, f"{dev_type}_{suffix}.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
            logging.info("Saved %d records → %s", len(records), out)
            print(f"  ✓  Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# §4  CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IoT Device Decision Agent (Gemini + Claude joint voting)"
    )
    parser.add_argument("--dev",  type=str, default=None,
                        help="Device type, e.g. NVR, ROUTER (omit to run all)")
    parser.add_argument("--ip",   type=str, default=None,
                        help="Specific IP address to classify")
    parser.add_argument("--max",  type=int, default=None,
                        help="Max samples per device type")
    args = parser.parse_args()

    agent = DecisionAgent()
    result = agent.run(dev_type=args.dev, ip=args.ip, max_samples=args.max)

    # Pretty-print a preview
    preview = result if isinstance(result, dict) else (result[:3] if result else [])
    print(json.dumps(preview, indent=2, ensure_ascii=False, default=str))
