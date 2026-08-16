import binascii
import os
import sys
import json
import ast
import re
import time
import math
import hashlib
import threading
import unicodedata
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subprocess import check_output, STDOUT

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_all_dev_labels():
    """
    Load all device labels that will be configured into the graph
    """
    device_label_list = []
    label_file_path = os.path.join(BASE_PATH, "rag_devices.json")
    if not os.path.exists(label_file_path):
        print(f"No device labels file found!")
        return None

    with open(label_file_path, "r") as f:
        dt = json.load(f)

    for device_type in dt.values():
        device_label_list.extend(device_type)

    return device_label_list


def load_new_dev_labels():
    device_label_list = []
    label_file_path = os.path.join(BASE_PATH, "new_devices.json")
    if not os.path.exists(label_file_path):
        print(f"No device labels file found!")
        return None

    with open(label_file_path, "r") as f:
        dt = json.load(f)

    for device_type in dt.values():
        device_label_list.extend(device_type)

    return device_label_list

def load_perspective_info():
    """加载 perspective 信息 """
    perspective_info_path = os.path.join(BASE_PATH, "perspective_info.json")
    if not os.path.exists(perspective_info_path):
        raise FileNotFoundError(f"perspective 信息文件不存在: {perspective_info_path}")
    
    with open(perspective_info_path, "r") as f:
        perspective_info = json.load(f)
    
    return perspective_info

def load_perspective_cluster_info():
    """加载 perspective 信息 """
    perspective_info_path = os.path.join(BASE_PATH, "perspective_name.json")
    if not os.path.exists(perspective_info_path):
        raise FileNotFoundError(f"perspective 信息文件不存在: {perspective_info_path}")
    
    with open(perspective_info_path, "r") as f:
        perspective_cluster_info = json.load(f)
    
    return perspective_cluster_info

def load_local_used_features():
    """加载本地使用的特征列表"""
    local_feature_path = os.path.join(BASE_PATH, "local_used_feature.txt")
    if not os.path.exists(local_feature_path):
        raise FileNotFoundError(f"本地特征文件不存在: {local_feature_path}")

    with open(local_feature_path, "r", encoding="utf-8") as f:
        local_used_features = [line.strip() for line in f.readlines() if line.strip()]

    return local_used_features


# ═══════════════════════════════════════════════════════════════════════════════
# Unseen-device-detection SHARED prompt (train == inference, "Method B" alignment)
# ═══════════════════════════════════════════════════════════════════════════════
# Only the per-perspective "info" columns are fed to the model — NOT the raw
# vendor/product/version sub-columns (e.g. hw-vendors/hw-products/hw-versions are
# dropped, only hw-info is kept; http-bodys/http-tags/http-part-info are dropped,
# only http-info is kept). The SAME builder is used by
#   - evaluation/unseen/llama3/prepare_data.py  (SFT training-data generation)
#   - agent/unseen.py                            (fine-tuned-adapter inference)
# so the prompt the adapter sees is byte-identical at train and inference time.
UNSEEN_INFO_COLS = [
    "as-info",        # Autonomous System perspective
    "whois-info",     # WHOIS perspective
    "os-info",        # Operating System perspective
    "sw-info",        # Software perspective
    "hw-info",        # Hardware perspective
    "service-distribution",  # Service distribution perspective
    "http-info",      # HTTP perspective (favicons+tags+body aggregated)
    "cert-info",      # Certificate perspective
    "dns-reverse",    # DNS perspective
]

UNSEEN_SYSTEM = (
    "You are an expert IoT network device classifier specializing in "
    "unseen device detection. Always respond with valid JSON as instructed."
)
CLASSIFIER_SYSTEM = (
    "You are an IoT network device classifier. Predict the concrete device type "
    "and vendor from the supplied fingerprint and return valid JSON only."
)
CLASSIFICATION_CONTRACT_VERSION = 3
SUMMARY_PROMPT_VERSION = 1
UNSEEN_TYPE_ALIASES = {
    "NAS": {"network attached storage", "network storage"},
    "NVR": {"network video recorder", "video recorder"},
    "POWER_METER": {"power meter", "electricity meter", "energy meter"},
    "BUILDING_AUTOMATION": {
        "building automation",
        "building management system",
        "bms",
    },
    "MEDICAL": {"medical device", "healthcare device"},
    "ROUTER": {"network router", "wireless router", "home router"},
    "PRINTER": {"network printer", "office printer"},
    "SCADA": {"scada", "industrial control system"},
    "CAMERA": {"network camera", "ip camera", "security camera"},
    "ALARM": {"alarm", "alarm system", "security alarm"},
    "CONTROLLER": {
        "controller",
        "smart home controller",
        "home automation controller",
        "industrial controller",
    },
}
UNSEEN_VENDOR_SUFFIXES = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "limited", "ltd", "llc", "nv", "plc", "pte", "sa",
    "spa", "srl", "technology", "technologies",
}


def build_fingerprint_info_text(values):
    """Render the per-perspective info columns as ``col: value`` lines.

    Only the columns in :data:`UNSEEN_INFO_COLS` are emitted, and only when they
    carry a non-empty value. ``values`` may be a pandas Series, a dict, or any
    column-name → value mapping.
    """
    parts = []
    for col in UNSEEN_INFO_COLS:
        v = values.get(col) if hasattr(values, "get") else getattr(values, col, None)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            parts.append(f"{col}: {s}")
    return "\n".join(parts)


def build_unseen_type_vendor_classification_prompt(
    fingerprint_text, known_types, known_vendors_by_type, web_search_results=None
):
    """Build the type/vendor prompt used only by the unseen-model pipeline.

    ``web_search_results`` is optional: when provided (non-empty), a vendor-attribution
    section is injected so the fine-tuned adapter can refine its prediction from web
    evidence. The JSON output contract (device_type / device_vendor) is unchanged.
    """
    vendor_lines = []
    for device_type in known_types:
        vendors = known_vendors_by_type.get(device_type, [])
        vendor_lines.append(
            f"- {device_type}: {', '.join(vendors) if vendors else '(none)'}"
        )
    if web_search_results:
        web_section = (
            "\n## Web Search Results (Vendor Attribution)\n"
            f"{web_search_results}\n\n"
            "If web search results are present above, use them to refine "
            "the device_vendor and device_type.\n"
        )
    else:
        web_section = ""
    return f"""Classify the IoT device represented by the network fingerprint.

## Known RAG Device Types
{', '.join(known_types)}

## Known RAG Vendors by Device Type
{chr(10).join(vendor_lines)}

The references above are used for downstream novelty detection. Always predict
the most specific concrete device type and vendor supported by the fingerprint,
even when a label is outside the references. Use "UNKNOWN" only when the fingerprint
contains no usable evidence. Do not decide whether the device is new.

## Device Fingerprint
{fingerprint_text}
{web_section}
Return ONLY this JSON object:
{{
  "device_type": "<concrete device type>",
  "device_vendor": "<concrete vendor>"
}}"""


def normalize_unseen_label(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(
        "".join(char if char.isalnum() else " " for char in value).split()
    )


def normalize_unseen_vendor(value):
    tokens = normalize_unseen_label(value).split()
    while tokens and tokens[-1] in UNSEEN_VENDOR_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def match_known_unseen_type(predicted_type, known_types):
    predicted = normalize_unseen_label(predicted_type)
    if not predicted:
        return None
    predicted_tokens = set(predicted.split())
    for known_type in known_types:
        forms = {
            normalize_unseen_label(known_type),
            *{
                normalize_unseen_label(alias)
                for alias in UNSEEN_TYPE_ALIASES.get(known_type, set())
            },
        }
        for form in forms:
            form_tokens = set(form.split())
            if predicted == form:
                return known_type
            if form_tokens and (
                form_tokens.issubset(predicted_tokens)
                or predicted_tokens.issubset(form_tokens)
            ):
                return known_type
            if SequenceMatcher(None, predicted, form).ratio() >= 0.86:
                return known_type
    return None


def match_known_unseen_vendor(
    predicted_vendor,
    matched_rag_type,
    known_vendors_by_type,
):
    if not matched_rag_type:
        return None
    predicted = normalize_unseen_vendor(predicted_vendor)
    if not predicted:
        return None
    for known_vendor in known_vendors_by_type.get(matched_rag_type, []):
        known = normalize_unseen_vendor(known_vendor)
        if predicted == known:
            return known_vendor
        if SequenceMatcher(None, predicted, known).ratio() >= 0.95:
            return known_vendor
    return None


class DeepSeekFingerprintSummarizer:
    """Shared DeepSeek fingerprint fitting for SFT preparation and inference."""

    def __init__(
        self,
        cfg_path,
        cache_path,
        token_budget,
        tokenizer,
        max_workers=1,
        provider="DEEPSEEK",
    ):
        with open(cfg_path, "r", encoding="utf-8") as file:
            config = json.load(file)[provider]
        self.provider = provider
        self.model = config["MODEL"]
        self.base_url = config["BASE_URL"]
        self.api_key = config["API_KEY"]
        self.cache_path = cache_path
        self.token_budget = token_budget
        self.tokenizer = tokenizer
        self.max_workers = max(1, max_workers)
        self.cache = self._load_cache()
        self._lock = threading.Lock()
        self._client = None
        self._disabled = False
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_hard_truncates = 0
        self.n_errors = 0

    @property
    def contract(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "token_budget": self.token_budget,
        }

    def _load_cache(self):
        cache = {}
        if not os.path.exists(self.cache_path):
            return cache
        with open(self.cache_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)
                    cache[item["key"]] = item["summary"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return cache

    def _cache_key(self, text):
        material = (
            f"{self.provider}/{self.model}/v{SUMMARY_PROMPT_VERSION}/"
            f"{self.token_budget}\0{text}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _append_cache(self, key, summary):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(
                {"key": key, "summary": summary},
                ensure_ascii=False,
            ) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                base_url = self.base_url.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url += "/v1"
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=base_url,
                )
        return self._client

    def _prompt(self, fingerprint_text):
        return (
            "You compress IoT device fingerprints for a classifier. Summarize the "
            "fingerprint below into a keyword summary.\n"
            "Rules:\n"
            "- Keep the SAME `col: value` line format and ONLY these field labels, in "
            "this order: as-info, whois-info, os-info, sw-info, hw-info, "
            "service-distribution, http-info, cert-info, dns-reverse. Drop any field "
            "with no useful content.\n"
            "- For each field keep ONLY discriminative vendor, product, version, "
            "domain, certificate, OUI, model, and service:port tokens.\n"
            "- Remove boilerplate, repeated tokens, and generic words.\n"
            "- Output the summary ONLY, without prose or code fences.\n"
            f"- The result MUST fit under {self.token_budget} tokens.\n\n"
            f"Fingerprint:\n{fingerprint_text}"
        )

    def _hard_truncate(self, text):
        token_ids = self.tokenizer(
            text, add_special_tokens=False
        ).input_ids[:self.token_budget]
        with self._lock:
            self.n_hard_truncates += 1
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _strip_fences(text):
        text = str(text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        return text.strip()

    def _summarize(self, text):
        if self._disabled:
            return None
        max_tokens = min(max(self.token_budget, 1024), 4096)
        for attempt in range(2):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self._prompt(text)}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                summary = self._strip_fences(
                    response.choices[0].message.content
                )
                if summary:
                    return summary
            except Exception as exc:
                with self._lock:
                    self.n_errors += 1
                error = repr(exc).casefold()
                if any(marker in error for marker in (
                    "401", "402", "authentication", "balance",
                    "insufficient", "quota",
                )):
                    self._disabled = True
                    return None
                if attempt == 0:
                    time.sleep(2)
        return None

    def fit_one(self, text):
        token_count = len(
            self.tokenizer(text, add_special_tokens=False).input_ids
        )
        if token_count <= self.token_budget:
            return text, False

        key = self._cache_key(text)
        with self._lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.n_cache_hits += 1
        if cached is not None:
            return cached, True

        summary = self._summarize(text)
        if summary is None:
            return self._hard_truncate(text), True
        if len(self.tokenizer(
            summary, add_special_tokens=False
        ).input_ids) > self.token_budget:
            summary = self._hard_truncate(summary)

        with self._lock:
            if key not in self.cache:
                self.cache[key] = summary
                self._append_cache(key, summary)
                self.n_calls += 1
            else:
                summary = self.cache[key]
        return summary, True

    def fit_many(self, texts):
        if not texts:
            return []
        if self.max_workers == 1:
            return [self.fit_one(text) for text in texts]
        fitted = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fit_one, text): index
                for index, text in enumerate(texts)
            }
            for future in as_completed(futures):
                fitted[futures[future]] = future.result()
        return fitted


def geometric_mean_log_probs(log_probs):
    if not log_probs:
        return 0.0
    return float(math.exp(sum(log_probs) / len(log_probs)))


def field_generation_confidences(
    tokenizer,
    text,
    generated_ids,
    token_log_probs,
    fields=("device_type", "device_vendor"),
):
    """Map generated-token log probabilities to JSON string value fields."""
    prefix_lengths = [0]
    for end in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(
            generated_ids[:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prefix_lengths.append(len(prefix))

    confidences = {
        "overall": geometric_mean_log_probs(token_log_probs)
    }
    for field in fields:
        match = re.search(
            rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"',
            text,
            re.DOTALL,
        )
        if match is None:
            confidences[field] = 0.0
            continue
        value_start, value_end = match.span(1)
        field_log_probs = [
            token_log_probs[index]
            for index in range(min(
                len(token_log_probs),
                len(generated_ids),
            ))
            if prefix_lengths[index] < value_end
            and prefix_lengths[index + 1] > value_start
        ]
        confidences[field] = geometric_mean_log_probs(field_log_probs)
    return confidences


def build_unseen_detection_prompt(fingerprint_text, known_types, unseen_types):
    """Build the unified unseen-detection prompt.

    Used verbatim for both SFT data generation and fine-tuned-adapter inference,
    so that training and inference prompts are aligned.
    """
    return f"""Determine whether the following IoT device is a NEW device *type* and/or a NEW *vendor*, based on its network fingerprint.

## Known Device Types (RAG)
{', '.join(known_types)}

## Unseen Candidate Types
{', '.join(unseen_types)}

## Device Fingerprint
{fingerprint_text}

## Task
Estimate two INDEPENDENT probabilities:
- new_type_probability: how likely the device is a NEW type not in the Known Device Types.
- new_vendor_probability: how likely the device is from a NEW vendor.
If information is insufficient to decide, keep the probability BELOW 0.5.
If new_type_probability > 0.5 output the new type, else predicted_type = "none".
If new_vendor_probability > 0.5 output the new vendor, else predicted_vendor = "none".

Respond with ONLY a JSON block:
```json
{{
    "new_type_probability": <float 0.0-1.0>,
    "new_vendor_probability": <float 0.0-1.0>,
    "is_unseen": <bool>,
    "predicted_type": "<type or 'none'>",
    "predicted_vendor": "<vendor or 'none'>",
    "confidence": <float 0.0-1.0>
}}
```"""

def preprocess_vector(vec, weights):
    """
    核心步骤：对向量进行加权，并进行 L2 归一化
    这样在使用 Inner Product (IP) 检索时，结果等同于加权余弦相似度
    """
    weighted_vec = vec * weights
    norm = np.linalg.norm(weighted_vec)
    return weighted_vec / norm if norm > 0 else weighted_vec

def convert_json_from_str(text):
    """将LLM输出转换成 json """
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL)
    
    clean_content = match.group(1) if match else text
    return json.loads(clean_content)

def hex_to_bit_list(hex_value):
    # 将十六进制数转换为二进制，并去掉前缀 '0b'
    binary_value = bin(hex_value)[2:].zfill(16)  # zfill(16) 确保输出为16位

    # 将二进制字符串转换为比特位列表
    bit_list = [int(bit) for bit in binary_value]

    return bit_list


def write_list_to_file(filepath, data_list):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(f"{item}\n")

def execute(command):
    """
    Executes a command on the local host.
    :param str command: the command to be executedi
    :return: returns the output of the STDOUT or STDERR
    """
    print("Shell command : {}".format(command))
    # command = "{}; exit 0".format(command)
    return check_output(command, stderr=STDOUT, shell=True).decode("utf-8")


def list_files_in_folder(directory: str):
    all_file_path = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_file_path.append(os.path.join(root, file))
    return all_file_path


def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def read_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def read_tuple_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [ast.literal_eval(line.strip()) for line in f.readlines()]


def check_extension(file_path: str, extension: str) -> bool:
    # 获取文件的后缀名
    _, ext = os.path.splitext(file_path)

    # 检查是否是 .csv 后缀
    if ext.lower() == extension:
        return True
    else:
        return False


def split_list_by_size(lst, size=2000):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def decode_mixed_logs(raw: bytes) -> str:
    results = []

    try:
        results.append(raw.decode('utf-8'))
    except UnicodeDecodeError:
        pass  # 继续按字节扫描

    # 查找 UTF-8 解码失败点
    for i in range(len(raw)):
        try:
            prefix = raw[:i].decode('utf-8')
            suffix = raw[i:].decode('gbk')
            return prefix + suffix
        except UnicodeDecodeError:
            continue

    return repr(raw)


def process_labels(labels):
    processed = []
    for label in labels:
        # 去除末尾的 \r 和 \n
        clean_label = label.rstrip('\r\n')
        # 如果长度超过30，按每30字符加一个换行
        if len(clean_label) > 30:
            # 将字符串按每30字符分段并插入换行
            chunks = [clean_label[i:i+30] for i in range(0, len(clean_label), 30)]
            clean_label = '\n'.join(chunks)
        processed.append(clean_label)
    return processed


def parse_banner_hex(banner_hex: str):
    hex_str = banner_hex.strip().replace(" ", "").replace("\n", "")

    try:
        # banner_hex 转换成字符串
        raw_bytes = binascii.unhexlify(hex_str)
        http_text = raw_bytes.decode("utf-8", errors="ignore")
        return http_text

    except Exception as e:
        print(f"[ERROR] Transform hex banner to string fail!: {e}")
        return None

def chunk_text(text, max_length=500, overlap=50):
    """
    The chunk_text function is invoked in cluster.py as part of a resilience strategy during the generation of embeddings for IoT device traffic features. 
    When the embedding model fails to process a full text input (typically due to GPU memory limits), 
    the system attempts to break the text into smaller segments using chunk_text, embed each segment, and then average the resulting vectors.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

if __name__ == "__main__":
    pass
