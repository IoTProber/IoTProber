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
from typing import Any, Dict, List, Optional

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from util import load_all_dev_labels, convert_json_from_str

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision.log"),
    filemode="a",
)

# ── path constants ────────────────────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_RAW = os.path.join(_BASE, "platform_data", "csv", "local", "1")
_VAL_PATH  = os.path.join(_BASE, "evaluation", "validation")
_QDB_PATH  = os.path.join(_BASE, "agent", "query_db")
_RES_PATH  = os.path.join(_BASE, "evaluation", "predict", "result")
_CFG_PATH  = os.path.join(_BASE, "llm_config.json")

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
    for dev in _dev_labels:
        fp = os.path.join(db_dir, f"{dev}_{sub}.json")
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            if str(entry.get("query_fingerprint", {}).get("ip", "")) == str(ip):
                return {"status": "found", "candidate_dev": dev, "entry": entry}
    return {"status": "not_found", "ip": ip}


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
    hit = _lookup_qdb("local", ip)
    if hit["status"] == "not_found":
        return json.dumps(hit, ensure_ascii=False)

    entry = hit["entry"]
    return json.dumps(
        {
            "status": "found",
            "candidate_dev": hit["candidate_dev"],
            "top_k": entry.get("top_k", 5),
            "similar_devices": entry.get("similar_devices", []),
        },
        ensure_ascii=False,
        indent=2,
    )


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
    hit = _lookup_qdb("community", ip)
    if hit["status"] == "not_found":
        return json.dumps(hit, ensure_ascii=False)

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
                "common_patterns":   patterns,
                "matched_features":  c.get("matched_features", [])[:3],
                "unmatched_features": c.get("unmatched_features", [])[:3],
            }
        )

    return json.dumps(
        {
            "status":         "found",
            "total_clusters": entry.get("total_clusters", len(matched)),
            "matched_clusters": trimmed,
        },
        ensure_ascii=False,
        indent=2,
    )


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
    hit = _lookup_qdb("reasoning", ip)
    if hit["status"] == "not_found":
        return json.dumps(hit, ensure_ascii=False)

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
    return json.dumps(
        {
            "status":               "found",
            "path_matching_results": trimmed,
            "top_cluster_key":      top_cluster.get("cluster_key"),
        },
        ensure_ascii=False,
        indent=2,
    )


_TOOLS = [local_retrieval, community_retrieval, reasoning_path_retrieval]


# ═════════════════════════════════════════════════════════════════════════════
# §2  Prompt Templates
# ═════════════════════════════════════════════════════════════════════════════

_AGENT_SYSTEM = """\
You are an expert IoT network device classifier with deep knowledge of network
traffic fingerprinting. Your mission is to identify the **device type** and
**manufacturer/vendor** of an unknown IoT device.

## Candidate Device Types
CAMERA | NVR | ROUTER | NAS | PRINTER | MEDICAL | SCADA | BUILDING_AUTOMATION | POWER_METER

## Your Workflow
You have access to three retrieval tools. Call them in order:
  1. local_retrieval(ip)           – get top-k similar known devices
  2. community_retrieval(ip)       – get cluster-level behavioral patterns
  3. reasoning_path_retrieval(ip)  – get discriminative feature importance

After collecting all tool results, reason step by step through the
Chain-of-Thought framework below, then emit your final JSON answer.

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
(>0.85 = very strong, 0.70–0.85 = moderate, <0.70 = weak)

**[Step 7] Community Cluster Evidence**
Review matched clusters: similarity scores, common patterns, matched/unmatched
features. Do they corroborate the fingerprint analysis?

**[Step 8] Reasoning Path Evidence**
Review path matching scores and feature importance. Which features are most
discriminative and do they align with the query device?

**[Step 9] Synthesis & Confidence**
Weigh all evidence. Assign confidence:
  0.90–1.00 – overwhelming, consistent evidence across all sources
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
  "confidence":          0.0
}
```
"""

_AGENT_HUMAN = """\
Classify the IoT device with IP address: {ip}

Raw network fingerprint (excluding null fields):
```json
{fingerprint}
```

Call all three retrieval tools for this IP, then apply the 9-step reasoning
framework, and output your final JSON classification.
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
        Select the result with the higher confidence score.
        When both LLMs agree on device_type, boost the final confidence slightly
        (average of both + 0.05, capped at 1.0).
        """
        g_conf = float(gemini.get("confidence", 0.0))
        c_conf = float(claude.get("confidence", 0.0))

        winner, loser = (gemini, claude) if g_conf >= c_conf else (claude, gemini)
        agree = winner.get("device_type") == loser.get("device_type")

        final_conf = winner["confidence"]
        if agree:
            final_conf = min(1.0, round((g_conf + c_conf) / 2 + 0.05, 4))

        return {
            "final_device_type":        winner["device_type"],
            "final_device_type_reason": winner.get("device_type_reason", ""),
            "final_vendor":             winner["vendor"],
            "final_vendor_reason":      winner.get("vendor_reason", ""),
            "final_confidence":         round(final_conf, 4),
            "winning_llm":              winner["llm"],
            "llm_agreement":            agree,
            "gemini": {
                "device_type":        gemini.get("device_type"),
                "device_type_reason": gemini.get("device_type_reason", ""),
                "vendor":             gemini.get("vendor"),
                "vendor_reason":      gemini.get("vendor_reason", ""),
                "confidence":         g_conf,
                "synthesis":          gemini.get("step9_synthesis", ""),
            },
            "claude": {
                "device_type":        claude.get("device_type"),
                "device_type_reason": claude.get("device_type_reason", ""),
                "vendor":             claude.get("vendor"),
                "vendor_reason":      claude.get("vendor_reason", ""),
                "confidence":         c_conf,
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
            "winning_llm":               voting["winning_llm"],
            "llm_agreement":             voting["llm_agreement"],
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
