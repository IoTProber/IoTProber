from typing import List, Dict, Optional, Any
import re
import os
import sys
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
import time
import heapq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import torch, gc

from typing import List, Optional

from llm import LLM
from util import *

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='a',
    filename="retrieval.log"
)

class MultiLevelRetrieval:
    """
    负责设备检索, 包括局部检索和全局社区检索
    Responsible for device retrieval, including local retrieval and global community retrieval
    """
    
    def __init__(self, llm: str = "gemini", gpu: int = -1, whether_milvus: bool = False):
        """
        初始化MultiLevelRetrieval
        Initialize MultiLevelRetrieval
        
        Args:
            llm: LLM类型, 可选 "gemini", "deepseek", "openai"
                 LLM type, options: "gemini", "deepseek", "openai"
            csv_base_path: CSV文件基础路径, 默认为 iot-classification/rag_data/csv
                           CSV file base path, default: iot-classification/rag_data/csv
        """
        # 设置路径
        # Set paths
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.csv_base_path = os.path.join(self.base_path, "platform_data", "csv")
        
        self.local_path = os.path.join(self.csv_base_path, "local/1")
        self.single_view_path = os.path.join(self.local_path, "community/single")
        self.com_view_path = os.path.join(self.local_path, "community/embedding_overall")

        self.agent_path = os.path.join(self.base_path, "agent")

        self.device_label_list = load_all_dev_labels()
        
        # Initialize embedding model and llm model
        self.gpu = gpu
        self.initialize_embedding_model()
        
        self.used_llm_model = llm
        self.llm = LLM()

        self.perspective_info_config = load_perspective_info()
        self.perspective_names = list(self.perspective_info_config.keys())

        # 排除不参与局部检索和推理路径检索的perspective
        self.except_perspective = ["hpart", "http", "overall"]
        self.retrieval_perspective_names = [
            p for p in self.perspective_names if p not in self.except_perspective
        ]

        self.perspective_weights_dict = {p: self.perspective_info_config[p]["weight"] for p in self.retrieval_perspective_names}
        self.perspective_weights = list(self.perspective_weights_dict.values())
        
        # 初始化Milvus向量数据库客户端
        self.vector_log_path = os.path.join(self.agent_path, "store_vector.log")
        self.vector_db_path = os.path.join(self.local_path, "vectorDB")
        os.makedirs(self.vector_db_path, exist_ok=True)

        self.milvus_db_path = os.path.join(self.vector_db_path, "milvus.db")
        if whether_milvus:
            from pymilvus import MilvusClient
            self.milvus_client = MilvusClient(uri=self.milvus_db_path)
            logging.info(f"=== Milvus客户端已连接: {self.milvus_db_path} ===\n")
        else:
            self.milvus_client = None
            logging.info("=== Milvus客户端未启用 ===\n")
            
        self.embedding_dim = 1024
        self.embedding_overall_dim = 1024 * len(self.retrieval_perspective_names)

        self.max_ip_length = 64
        self.max_device_length = 64
        self.batch_size = 1000  # 每批插入条数 / rows per insert batch

        # numpy-based local vector storage (避免 Milvus Lite 高维向量崩溃)
        self.local_npz_dir = os.path.join(self.vector_db_path, "local_npz")
        os.makedirs(self.local_npz_dir, exist_ok=True)
        self._local_vectors_cache = None  # lazy-loaded: {embeddings, ips, device_types}

        self.retrieval_history: List[Dict[str, Any]] = []
        self.reasoning_use_llm = True
        self._graph_db = None  # lazy Neo4j connection, initialized on first graph_neighbor use
    
    def initialize_embedding_model(self):
        """
        初始化embedding模型
        Initialize embedding model
        """
        print("=== 初始化Embedding模型 ===")
        embedding_model_path = os.path.join(self.base_path, "qwen3_embedding_06b")
        
        if self.gpu != -1:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": self.gpu},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
                query_encode_kwargs={"normalize_embeddings": True, "batch_size": 1}
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
                query_encode_kwargs={"normalize_embeddings": True, "batch_size": 1}
            )
        # 初始化PCA降维器 (1024 -> 256)
        # Initialize PCA for dimensionality reduction (1024 -> 256)
        self.pca = PCA(n_components=256)
        
        print("=== Embedding模型初始化完成 ===\n")
    
    def get_device_csv_files(self, perspective: str) -> List[dict]:
        """
        根据 rag_domain.json 中的设备名称, 获取某个 perspective 下各设备的 embedding CSV 文件(排除 PCA 版本)
        Get embedding CSV files for each device under a perspective directory, excluding PCA versions.
        Device names are sourced from rag_domain.json.
        Returns list of {device, filepath}
        """
        emb_dir = os.path.join(self.local_path, f"embedding_{perspective}")
        if not os.path.isdir(emb_dir):
            return []

        results = []
        for device in self.device_label_list:
            fname = f"ipraw_{device}_embedding_{perspective}.csv"
            fpath = os.path.join(emb_dir, fname)
            if os.path.isfile(fpath):
                results.append({
                    "device": device,
                    "filepath": fpath,
                })
        return results
    
    def create_collection(self, perspective: str, drop: bool = False, col_name: str = None, embedding_dim: int = 1024):
        """
        为某个 perspective 创建 Milvus collection
        Schema: id(int64 pk auto), ip(varchar), device_type(varchar), embedding(float_vector 1024)
        """
        if col_name is None:
            col_name = f"embedding_{perspective}"

        if drop and self.milvus_client.has_collection(col_name):
            logging.info(f"Dropping existing collection: {col_name}")
            self.milvus_client.drop_collection(col_name)

        if self.milvus_client.has_collection(col_name):
            logging.info(f"Collection already exists: {col_name}")
            return col_name

        schema = self.milvus_client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("ip", DataType.VARCHAR, max_length=self.max_ip_length)
        schema.add_field("device_type", DataType.VARCHAR, max_length=self.max_device_length)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim)

        # 创建 collection
        self.milvus_client.create_collection(
            collection_name=col_name,
            schema=schema,
        )

        # 创建向量索引 (IVF_FLAT for local use)
        index_params = self.milvus_client.prepare_index_params()
        if perspective == "overall":
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
        else:
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
        self.milvus_client.create_index(
            collection_name=col_name,
            index_params=index_params,
        )

        logging.info(f"Created collection: {col_name} (dim={EMBEDDING_DIM})")
        return col_name

    def insert_csv_to_collection(self, col_name: str, device: str, filepath: str):
        """
        读取 CSV 文件并批量插入到 Milvus collection.
        当 col_name == "local" 时, filepath 为 embedding_local/ipraw_{device}_embedding.csv,
        其特征列为各 perspective 的 1024 维 embedding 拼接, 列名格式为 {perspective_name}{1..1024}.
        插入前对各 perspective 块乘以对应权重并做整体 L2 归一化.
        """
        logging.info(f"Loading CSV: {os.path.basename(filepath)} (device={device})")

        # 分块读取大文件 / read large files in chunks
        # local collection 向量维度极高, 用更小的 batch 避免 Milvus Lite 崩溃
        chunksize = 500 if col_name == "local" else self.batch_size
        total_inserted = 0
        for chunk_df in pd.read_csv(filepath, chunksize=chunksize):
            ip_list = chunk_df["ip"].astype(str).tolist()

            if col_name == "local":
                # 按 perspective 顺序构建有序列名, 保证各 perspective 的 1024 维正确对齐
                ordered_cols = []
                for p_name in self.retrieval_perspective_names:
                    ordered_cols.extend(
                        [f"{p_name}{i}" for i in range(1, self.embedding_dim + 1)]
                    )
                embeddings = chunk_df[ordered_cols].values  # (batch, overall_dim)

                num_perspectives = len(self.retrieval_perspective_names)
                weights = np.array(self.perspective_weights)  # (num_perspectives,)

                # reshape -> (batch, num_perspectives, 1024), 逐 perspective 乘权重
                emb_reshaped = embeddings.reshape(
                    embeddings.shape[0], num_perspectives, self.embedding_dim
                )
                weighted = emb_reshaped * weights.reshape(1, -1, 1)

                # 展平回 (batch, overall_dim) 并 L2 归一化
                weighted_flat = weighted.reshape(embeddings.shape[0], -1)
                norms = np.linalg.norm(weighted_flat, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                normalized = weighted_flat / norms

                embeddings_list = normalized.tolist()
            else:
                embedding_cols = [c for c in chunk_df.columns if c != "ip"]
                embeddings_list = chunk_df[embedding_cols].values.tolist()

            data = [
                {
                    "ip": ip_list[i],
                    "device_type": device,
                    "embedding": embeddings_list[i],
                }
                for i in range(len(ip_list))
            ]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.milvus_client.insert(collection_name=col_name, data=data)
                    break
                except Exception as e:
                    if "Connection refused" in str(e) or "UNAVAILABLE" in str(e):
                        logging.warning(
                            f"  Milvus connection lost (attempt {attempt+1}/{max_retries}), "
                            f"reconnecting..."
                        )
                        time.sleep(5)
                        self.milvus_client = MilvusClient(uri=self.milvus_db_path)
                        logging.info("  Milvus client reconnected.")
                    else:
                        raise
            else:
                logging.error(
                    f"  Failed to insert batch after {max_retries} retries, skipping."
                )
                continue

            total_inserted += len(data)
            logging.info(
                f"  Inserted batch: {len(data)} rows "
                f"(total so far: {total_inserted})"
            )

        logging.info(
            f"Finished {os.path.basename(filepath)}: "
            f"{total_inserted} rows inserted into {col_name}"
        )
        return total_inserted

    def insert_csv_to_local_npz(self, device: str, filepath: str) -> int:
        """
        读取 CSV 文件, 对各 perspective 加权 + L2 归一化后, 保存为 numpy npz 文件.
        替代 Milvus Lite 存储, 避免高维向量导致嵌入式服务器崩溃.
        每个 device 保存为 local_npz/{device}.npz, 包含 embeddings, ips 两个数组.
        """
        logging.info(f"Loading CSV: {os.path.basename(filepath)} (device={device})")

        all_embeddings = []
        all_ips = []
        total_rows = 0

        for chunk_df in pd.read_csv(filepath, chunksize=self.batch_size):
            ip_list = chunk_df["ip"].astype(str).tolist()

            # 按 perspective 顺序构建有序列名
            ordered_cols = []
            for p_name in self.retrieval_perspective_names:
                ordered_cols.extend(
                    [f"{p_name}{i}" for i in range(1, self.embedding_dim + 1)]
                )
            embeddings = chunk_df[ordered_cols].values  # (batch, overall_dim)

            num_perspectives = len(self.retrieval_perspective_names)
            weights = np.array(self.perspective_weights)  # (num_perspectives,)

            # reshape -> (batch, num_perspectives, 1024), 逐 perspective 乘权重
            emb_reshaped = embeddings.reshape(
                embeddings.shape[0], num_perspectives, self.embedding_dim
            )
            weighted = emb_reshaped * weights.reshape(1, -1, 1)

            # 展平回 (batch, overall_dim) 并 L2 归一化
            weighted_flat = weighted.reshape(embeddings.shape[0], -1)
            norms = np.linalg.norm(weighted_flat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized = weighted_flat / norms

            all_embeddings.append(normalized.astype(np.float32))
            all_ips.extend(ip_list)
            total_rows += len(ip_list)
            logging.info(
                f"  Processed batch: {len(ip_list)} rows "
                f"(total so far: {total_rows})"
            )

        # 拼接所有 chunk 并保存为 .npy (支持 mmap 读取, 避免 OOM)
        embeddings_array = np.vstack(all_embeddings)  # (N, overall_dim)
        ips_array = np.array(all_ips, dtype=str)

        emb_path = os.path.join(self.local_npz_dir, f"{device}_embeddings.npy")
        ips_path = os.path.join(self.local_npz_dir, f"{device}_ips.npy")
        np.save(emb_path, embeddings_array)
        np.save(ips_path, ips_array)
        logging.info(
            f"Finished {os.path.basename(filepath)}: "
            f"{total_rows} rows saved to {device}_embeddings.npy"
        )
        # 清除缓存, 下次检索时重新加载
        self._local_vectors_cache = None
        return total_rows

    def _search_local_vectors(self, query_vec: np.ndarray, top_k: int = 10):
        """
        逐文件流式搜索 local npy, 使用 mmap 避免将整个数组加载到 RAM, 防止 OOM.
        返回 list of {ip, device_type, score}, 按 score 降序, 长度 <= top_k.
        """
        # 用 min-heap 维护 top_k (score, idx) — heapq 是最小堆, 堆顶是最小值
        heap = []  # elements: (score, global_counter, ip, device_type)
        counter = 0
        total_compared = 0

        # 扫描 {device}_embeddings.npy 文件
        emb_files = sorted(
            f for f in os.listdir(self.local_npz_dir)
            if f.endswith("_embeddings.npy")
        )
        if not emb_files:
            return [], 0

        for emb_fname in emb_files:
            device = emb_fname.replace("_embeddings.npy", "")
            emb_path = os.path.join(self.local_npz_dir, emb_fname)
            ips_path = os.path.join(self.local_npz_dir, f"{device}_ips.npy")

            # mmap_mode='r': 内存映射只读, OS 按需分页, 不占用物理 RAM
            emb = np.load(emb_path, mmap_mode='r')  # (n, dim), float32
            ips = np.load(ips_path, allow_pickle=True)  # (n,)

            # 批量计算 IP scores
            scores = emb @ query_vec  # (n,)
            total_compared += len(scores)

            # 取当前文件的 top_k 候选, 减少 heap 操作
            if len(scores) > top_k:
                part_idx = np.argpartition(scores, -top_k)[-top_k:]
            else:
                part_idx = np.arange(len(scores))

            for i in part_idx:
                s = float(scores[i])
                if len(heap) < top_k:
                    heapq.heappush(heap, (s, counter, str(ips[i]), device))
                    counter += 1
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, counter, str(ips[i]), device))
                    counter += 1

            del emb, ips, scores  # 释放引用

        # 从 heap 中提取结果, 按 score 降序
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [
            {"ip": r[2], "device_type": r[3], "similarity_score": r[0]}
            for r in results
        ], total_compared

    def check_existing_count(self, col_name: str, device: str) -> int:
        """检查 collection 中某个 device 已有的记录数"""
        if not self.milvus_client.has_collection(col_name):
            return 0
        try:
            results = self.milvus_client.query(
                collection_name=col_name,
                filter=f'device_type == "{device}"',
                output_fields=["ip"],
                limit=1,
            )
            # 如果能查到至少一条，说明已经有数据
            return len(results)
        except Exception:
            return 0

    def parse_log_for_resume(self) -> Optional[dict]:
        """
        解析 store_vector.log, 找到最后一个未完成的 (perspective, device)
        Parse store_vector.log to find the last incomplete (perspective, device).
        
        逻辑: 从日志尾部向前搜索, 找到最后一条 "Loading CSV" 行,
        如果之后没有对应的 "Finished" 行, 说明该文件处理中断了。
        
        Returns:
            dict with keys {perspective, device, collection, inserted_so_far} or None if all complete.
        """
        if not os.path.exists(self.vector_log_path):
            logging.warning(f"Log file not found: {log_path}")
            return None

        # 匹配常规 perspective: ipraw_{device}_embedding_{perspective}.csv
        loading_pattern = re.compile(
            r"Loading CSV: ipraw_(.+)_embedding_(.+)\.csv \(device=(.+)\)"
        )
        finished_pattern = re.compile(
            r"Finished ipraw_(.+)_embedding_(.+)\.csv: (\d+) rows inserted into (.+)"
        )
        # 匹配 local collection: ipraw_{device}_embedding.csv (无 perspective 后缀)
        loading_local_pattern = re.compile(
            r"Loading CSV: ipraw_(.+)_embedding\.csv \(device=(.+)\)"
        )
        finished_local_pattern = re.compile(
            r"Finished ipraw_(.+)_embedding\.csv: (\d+) rows (?:inserted into|saved to) (.+)"
        )
        inserted_pattern = re.compile(
            r"(?:Inserted|Processed) batch: \d+ rows \(total so far: (\d+)\)"
        )

        last_loading = None
        last_loading_line_num = -1  # log中记录的最后加载csv文件的行号
        finished_after_last_loading = False  # 加载后是否完成
        last_inserted_total = 0  # 最后插入的总行数

        with open(self.vector_log_path, "r") as f:
            for line_num, line in enumerate(f):
                # 1a. 匹配常规 perspective 的 Loading CSV
                m_load = loading_pattern.search(line)
                if m_load:
                    collection_name = "embedding_{}".format(m_load.group(2))
                    last_loading = {
                        "device": m_load.group(1),
                        "perspective": m_load.group(2),
                        "collection": collection_name,
                    }
                    last_loading_line_num = line_num
                    finished_after_last_loading = False
                    last_inserted_total = 0

                # 1b. 匹配 local collection 的 Loading CSV
                m_load_local = loading_local_pattern.search(line)
                if m_load_local:
                    last_loading = {
                        "device": m_load_local.group(1),
                        "perspective": "overall",
                        "collection": "local",
                    }
                    last_loading_line_num = line_num
                    finished_after_last_loading = False
                    last_inserted_total = 0
                
                # 2a. 匹配常规 perspective 的 Finished
                m_fin = finished_pattern.search(line)
                if m_fin and last_loading and line_num > last_loading_line_num:
                    if (m_fin.group(1) == last_loading["device"]
                            and m_fin.group(2) == last_loading["perspective"]):
                        finished_after_last_loading = True

                # 2b. 匹配 local collection 的 Finished
                m_fin_local = finished_local_pattern.search(line)
                if m_fin_local and last_loading and line_num > last_loading_line_num:
                    if (m_fin_local.group(1) == last_loading["device"]
                            and last_loading["collection"] == "local"):
                        finished_after_last_loading = True

                # 3. 匹配已经插入的行数
                m_ins = inserted_pattern.search(line)
                if m_ins and last_loading and line_num > last_loading_line_num:
                    last_inserted_total = int(m_ins.group(1))

        if last_loading and not finished_after_last_loading:
            last_loading["inserted_so_far"] = last_inserted_total
            return last_loading

        return None
    
    def delete_device_from_collection(self, col_name: str, device: str) -> int:
        """
        删除 collection 中某个 device 的所有记录
        Delete all records for a specific device from a collection.
        
        Returns: 删除的记录数 (approximate)
        """
        if not self.milvus_client.has_collection(col_name):
            logging.warning(f"Collection {col_name} does not exist, nothing to delete")
            return 0

        logging.info(f"Deleting all records for device '{device}' from {col_name}...")

        # Milvus Lite delete by filter
        try:
            result = self.milvus_client.delete(
                collection_name=col_name,
                filter=f'device_type == "{device}"',
            )
            deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
            logging.info(f"Deleted records for device '{device}' from {col_name} (result: {result})")
            return deleted_count
        except Exception as e:
            logging.error(f"Error deleting device '{device}' from {col_name}: {e}")
            return 0
    
    def vector_store_embedding(self, whether_resume: bool = False, whether_drop: bool = False, whether_skip: bool = True):

        resume_perspective = None
        resume_device = None

        if whether_resume:
            log_path = os.path.join(self.agent_path, "store_vector.log")
            interrupted = self.parse_log_for_resume()
            if interrupted:
                resume_perspective = interrupted["perspective"]  # 中断任务的 perspective
                resume_device = interrupted["device"]  # 中断任务的 device
                col_name = interrupted["collection"]  # 中断任务的 collection
                logging.info(
                    f"[Resume] Detected interrupted task: "
                    f"perspective={resume_perspective}, device={resume_device}, "
                    f"collection={col_name}, inserted_so_far={interrupted['inserted_so_far']}"
                )
                # 删除该 device 的不完整数据 / delete partial data for this device
                if col_name == "local":
                    # local collection 使用 numpy npy 存储
                    for suffix in ("_embeddings.npy", "_ips.npy"):
                        p = os.path.join(self.local_npz_dir, f"{resume_device}{suffix}")
                        if os.path.exists(p):
                            os.remove(p)
                    logging.info(
                        f"[Resume] Partial npy for {resume_device} has been deleted. "
                        f"Will re-process from scratch."
                    )
                else:
                    self.delete_device_from_collection(col_name, resume_device)
                    logging.info(
                        f"[Resume] Partial data for {resume_device} in {col_name} has been deleted. "
                        f"Will re-insert from scratch."
                    )
            else:
                logging.info("[Resume] No interrupted task found in log. All tasks completed.")

        total_rows = 0
        for perspective in self.perspective_names:

            csv_files = self.get_device_csv_files(perspective)
            if not csv_files:
                logging.warning(f"No CSV files found for perspective '{perspective}'")
                continue

            if not csv_files:
                logging.info(f"No matching device CSVs for perspective '{perspective}'")
                continue

            # 创建 collection
            col_name = self.create_collection(perspective, drop=whether_drop, embedding_dim=self.embedding_dim)

            # 逐文件插入 / insert each CSV
            for csv_info in csv_files:
                device = csv_info["device"]
                filepath = csv_info["filepath"]

                is_resume_target = (
                    whether_resume
                    and resume_perspective == perspective
                    and resume_device == device
                )

                # 跳过已存在的数据 / skip if already loaded
                if whether_skip and not whether_drop and not is_resume_target:
                    existing = self.check_existing_count(col_name, device)
                    if existing > 0:
                        logging.info(
                            f"Skipping {device} in {col_name} (already has data)"
                        )
                        continue

                if is_resume_target:
                    logging.info(f"[Resume] Re-inserting {device} into {col_name}")

                rows = self.insert_csv_to_collection(col_name, device, filepath)
                total_rows += rows

        logging.info("Processing embedding_overall files (numpy storage)...")
        local_dir = os.path.join(self.local_path, "embedding_local")
        
        if os.path.isdir(local_dir):
            if whether_drop:
                # 清空 local npy 目录
                for f in os.listdir(self.local_npz_dir):
                    if f.endswith(".npy") or f.endswith(".npz"):
                        os.remove(os.path.join(self.local_npz_dir, f))
                logging.info("Dropped all local npy files.")

            # 遍历所有设备的 ipraw_{dev}_embedding.csv 文件
            for device in self.device_label_list:
                fname = f"ipraw_{device}_embedding.csv"
                fpath = os.path.join(local_dir, fname)
                if os.path.isfile(fpath):
                    emb_npy_path = os.path.join(self.local_npz_dir, f"{device}_embeddings.npy")
                    is_resume_target = (
                        whether_resume
                        and resume_perspective == "overall"
                        and resume_device == device
                    )

                    # resume 时删除不完整的 npy 文件
                    if is_resume_target:
                        for suffix in ("_embeddings.npy", "_ips.npy"):
                            p = os.path.join(self.local_npz_dir, f"{device}{suffix}")
                            if os.path.exists(p):
                                os.remove(p)
                        logging.info(f"[Resume] Deleted partial npy for {device}")

                    # 检查是否跳过已存在的数据
                    if whether_skip and not whether_drop and not is_resume_target:
                        if os.path.exists(emb_npy_path):
                            logging.info(
                                f"Skipping {device} in local_npz (already has data)"
                            )
                            continue

                    if is_resume_target:
                        logging.info(f"[Resume] Re-processing {device} into local_npz")
                    
                    rows = self.insert_csv_to_local_npz(device, fpath)
                    total_rows += rows
                else:
                    logging.warning(f"File not found: {fname}")
        else:
            logging.warning(f"Directory not found: {local_dir}")

        logging.info(f"All done. Total rows inserted: {total_rows}")
        logging.info(f"Vector DB saved to: {VECTOR_DB_DIR}")   

    def _get_graph_db(self):
        """
        懒加载 Neo4j ProtocolGraph 连接
        Lazily initialize the Neo4j ProtocolGraph connection.
        """
        if self._graph_db is None:
            try:
                graph_path = os.path.join(os.path.dirname(self.base_path), "graph") \
                    if not os.path.isdir(os.path.join(self.base_path, "graph")) \
                    else os.path.join(self.base_path, "graph")
                sys.path.insert(0, graph_path)
                from api import ProtocolGraph
                self._graph_db = ProtocolGraph("neo4j://localhost:7687", "neo4j", "12345678")
                logging.info("Neo4j ProtocolGraph connected for graph_neighbor queries.")
            except Exception as e:
                logging.warning(f"Failed to connect to Neo4j for graph_neighbor: {e}")
                self._graph_db = None
        return self._graph_db

    def _get_graph_neighbors(
        self,
        ip_list: List[str],
        top_k: int,
        q: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        对 top-K 中每个 IP, 在第一层设备实体图中查询邻居节点:
        两个 Device 节点间共享 Feature 节点数 > q 则视为相邻.
        按共享特征数降序排列, 去重后返回至多 top_k 个新邻居.

        For each IP in ip_list, query the Layer-1 entity graph for adjacent Device nodes
        (sharing more than q Feature nodes). Returns at most top_k unique new neighbors,
        sorted by shared feature count descending.
        """
        graph_db = self._get_graph_db()
        if graph_db is None:
            return []

        ip_set = set(ip_list)
        neighbor_map: Dict[str, Dict[str, Any]] = {}  # ip -> best entry

        cypher = (
            "MATCH (a:Device {ip: $p1})-[]->(f:Feature)<-[]-(b:Device) "
            "WHERE b.ip <> $p1 "
            "WITH b.ip AS neighbor_ip, b.device_type AS device_type, count(DISTINCT f) AS shared_count "
            "WHERE shared_count > $p2 "
            "RETURN neighbor_ip, device_type, shared_count "
            "ORDER BY shared_count DESC"
        )

        for ip in ip_list:
            try:
                results = graph_db.CypherQuery(cypher, ip, q)
            except Exception as e:
                logging.warning(f"graph_neighbor Cypher failed for ip={ip}: {e}")
                continue
            if not results:
                continue
            for row in results:
                nip = row.get("neighbor_ip")
                if nip is None or nip in ip_set:
                    continue
                sc = int(row.get("shared_count", 0))
                if nip not in neighbor_map or sc > neighbor_map[nip]["shared_count"]:
                    neighbor_map[nip] = {
                        "ip": nip,
                        "device_type": row.get("device_type", ""),
                        "similarity_score": 0.0,
                        "shared_count": sc,
                        "source": "graph_neighbor",
                    }

        neighbors = sorted(neighbor_map.values(), key=lambda x: x["shared_count"], reverse=True)
        logging.info(f"graph_neighbor: found {len(neighbors)} unique neighbors (q>{q}), returning top {top_k}")
        return neighbors[:top_k]

    def local_retrieval(self, query_fingerprint: Dict[str, Any], top_k: int = 5,
                        graph_neighbor: bool = False, neighbor_q: int = 1) -> Dict[str, Any]:
        """
        局部检索: 从local文件夹中检索与输入指纹最相似的设备
        Local retrieval: retrieve devices most similar to input fingerprint from local folder

        Args:
            query_fingerprint: 查询设备的指纹信息(字典格式)
                               Query device fingerprint information (dictionary format)
            top_k: 返回最相似的设备数量, 默认5
                   Number of most similar devices to return, default 5
            graph_neighbor: 是否在第一层设备实体图中扩展邻居节点, 默认False
                            Whether to expand results with Layer-1 entity graph neighbors, default False
            neighbor_q: 邻居节点共享特征数阈值 (共享 > neighbor_q 才视为相邻), 默认1
                        Minimum shared feature count threshold (> neighbor_q to be adjacent), default 1

        Returns:
            包含最相似设备信息的JSON结果; 若 graph_neighbor=True, similar_devices 中包含
            向量检索结果 (source='vector_search') 和图邻居结果 (source='graph_neighbor'), 总数 ≤ 2K
            JSON result; when graph_neighbor=True, similar_devices contains both
            vector-search hits (source='vector_search') and graph neighbors (source='graph_neighbor'), total ≤ 2K
        """
        print(f"=== 开始局部检索 (top_k={top_k}, graph_neighbor={graph_neighbor}) ===")
        logging.info(f"=== 开始局部检索 (top_k={top_k}, graph_neighbor={graph_neighbor}) ===")

        try:
            t_start = time.time()

            # 1. 对query_fingerprint进行embedding
            # 1. Embed query_fingerprint
            query_embedding = self._embed_fingerprint(query_fingerprint)
            t_embed = time.time()
            logging.info(f"Query embedding shape: {query_embedding.shape}")

            # 2. 计算各 perspective 的可见性掩码 (m_i ∈ {0,1})
            # 2. Compute visibility mask (m_i ∈ {0,1}) for each perspective
            #    m_i = 1 若该 perspective 在 query_fingerprint 中存在非空特征, 否则 0
            raw_weights = np.array(self.perspective_weights, dtype=np.float32)  # (num_perspectives,)
            masks = np.zeros(len(self.retrieval_perspective_names), dtype=np.float32)
            for idx, p_name in enumerate(self.retrieval_perspective_names):
                feature_cols = self.perspective_info_config[p_name]["cols"]
                for col in feature_cols:
                    if col in query_fingerprint and col.lower() != 'ip':
                        val = query_fingerprint[col]
                        try:
                            is_missing = val is None or pd.isna(val) or str(val).strip() == ""
                        except (TypeError, ValueError):
                            is_missing = False
                        if not is_missing:
                            masks[idx] = 1.0
                            break

            # W_A = Σ w_i · m_i  (公式 1 分母 / denominator of Eq. 1)
            W_A = float(np.sum(raw_weights * masks))
            alpha = 2.0  # 指数衰减系数 / confidence decay exponent

            missing_perspectives = [p for p, m in zip(self.retrieval_perspective_names, masks) if m == 0.0]
            if missing_perspectives:
                logging.info(f"缺失 perspectives: {missing_perspectives}, W_A={W_A:.4f}")

            if W_A <= 0:
                # 所有 perspective 均缺失, 退化为均匀权重, 置信度为 0
                logging.warning("所有 perspectives 均缺失, 退化为均匀权重且置信度为 0")
                effective_weights = raw_weights / (raw_weights.sum() if raw_weights.sum() > 0 else 1.0)
                confidence = 0.0
            else:
                # w_i^rel = w_i / W_A  (公式 1); 缺失 perspective 权重归零
                effective_weights = (raw_weights * masks) / W_A
                # P_confidence = (W_A)^alpha  (公式 2 置信度因子 / confidence factor in Eq. 2)
                confidence = float(W_A ** alpha)

            # 将缺失 perspective 的 embedding 置零, 再按有效权重缩放
            # Zero out missing perspective embeddings, then scale by effective weights
            query_embedding = query_embedding * masks.reshape(-1, 1)  # (num_perspectives, 1024)
            weighted_embedding = query_embedding * effective_weights.reshape(-1, 1)
            weighted_flat = weighted_embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(weighted_flat)
            if norm > 0:
                weighted_flat = weighted_flat / norm

            # 逐文件流式搜索, 每次只加载一个设备的 npy, 避免 OOM
            t_search_start = time.time()
            search_limit = max(top_k, 10)
            all_similarities, total_compared = self._search_local_vectors(
                weighted_flat, top_k=search_limit
            )
            t_search = time.time()

            # 3. 应用置信度衰减因子 (W_A)^alpha 到所有相似度分数 (公式 2)
            # 3. Apply confidence decay factor (W_A)^alpha to all similarity scores (Eq. 2)
            if confidence != 1.0:
                for item in all_similarities:
                    item["similarity_score"] *= confidence

            if not all_similarities:
                logging.warning("Local vectors not found (no npz files)")
                return {"query_fingerprint": query_fingerprint, "top_k": top_k,
                        "similar_devices": [], "total_compared": 0}

            hit_count = len(all_similarities)
            print(f"  search返回 {hit_count} 个候选 (共比较 {total_compared} 条)")
            logging.info(f"  search返回 {hit_count} 个候选 (共比较 {total_compared} 条)")

            # 4. 按相似度排序并取top_k, 标记来源为向量检索
            # 4. Sort by similarity, take top_k, mark source as vector_search
            all_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_devices = all_similarities[:top_k]
            for d in top_devices:
                d.setdefault("source", "vector_search")

            # 5. 图邻居扩展: 在第一层设备实体图中查找邻居节点, 合并后总数不超过 2K
            # 5. Graph neighbor expansion: find Layer-1 entity graph neighbors, total ≤ 2K
            if graph_neighbor and top_devices:
                top_ips = [d["ip"] for d in top_devices]
                neighbors = self._get_graph_neighbors(top_ips, top_k=top_k, q=neighbor_q)
                top_devices = top_devices + neighbors  # total ≤ 2K (top_k vector + top_k graph)
                print(f"  graph_neighbor: 合并后共 {len(top_devices)} 个设备 "
                      f"(向量检索={len(top_ips)}, 图邻居={len(neighbors)})")
                logging.info(f"graph_neighbor扩展: 向量检索={len(top_ips)}, "
                             f"图邻居={len(neighbors)}, 合并={len(top_devices)}")

            print(f"总共比较了 {len(all_similarities)} 个候选设备")

            t_end = time.time()
            timing = {
                "embedding_sec": round(t_embed - t_start, 4),
                "search_sec": round(t_search - t_search_start, 4),
                "total_sec": round(t_end - t_start, 4),
            }
            print(f"  耗时: embedding={timing['embedding_sec']}s, search={timing['search_sec']}s, total={timing['total_sec']}s")
            logging.info(f"  耗时: {timing}")

            result = {
                "query_fingerprint": query_fingerprint,
                "top_k": top_k,
                "similar_devices": top_devices,
                "total_compared": len(all_similarities),
                "timing": timing,
                "confidence_score": round(confidence, 6),
                "missing_perspectives": missing_perspectives,
                "graph_neighbor": graph_neighbor,
            }

            self.retrieval_history.append({
                "type": "local_retrieval",
                "timestamp": pd.Timestamp.now().isoformat(),
                "result": result
            })

            print(f"=== 局部检索完成, 找到 {len(top_devices)} 个相似设备 ===\n")
            logging.info(f"=== 局部检索完成, 找到 {len(top_devices)} 个相似设备 ===\n")
            logging.info(f"总共比较了 {len(all_similarities)} 个设备")

            return result

        except Exception as e:
            print(f"局部检索错误: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _embed_fingerprint(self, fingerprint: Dict[str, Any]) -> np.ndarray:
        """
        将设备指纹转换为embedding向量
        Convert device fingerprint to embedding vector
        
        Args:
            fingerprint: 设备指纹字典
                        Device fingerprint dictionary
        
        Returns:
            1024*len(self.retrieval_perspective_names)的embedding向量
            1024*len(self.retrieval_perspective_names)-dimensional embedding vector
        """
        # 按照self.perspective_names的顺序拼接特征
        # Concatenate features in the order of self.perspective_names
        combined_parts = []
        
        for perspective_name in self.retrieval_perspective_names:
            feature_cols = self.perspective_info_config[perspective_name]["cols"]
            feature_values = []
            
            for col in feature_cols:
                if col in fingerprint and col.lower() != 'ip':
                    value = fingerprint[col]
                    # 处理NaN和None值
                    if pd.isna(value) or value is None:
                        value = ""
                    feature_values.append(str(value))
            
            # 使用|拼接该特征组的所有列
            combined_parts.append("|".join(feature_values))
        
        # 对每个特征组分别进行embedding和降维
        # Embed and reduce dimensions for each feature group separately
        embeddings_list = []

        for part in combined_parts:
            # 使用embedding model获取1024维向量
            # Get 1024-dimensional vector using embedding model
            try:
                embedding_1024 = self.embedding_model.embed_query(part)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                logging.warning(f"Embedding OOM, 尝试chunk_text分段处理: {oom_err}")
                gc.collect()
                torch.cuda.empty_cache()
                chunks = chunk_text(part, max_length=500, overlap=50)
                chunk_embeddings = []
                for chunk in chunks:
                    try:
                        chunk_emb = self.embedding_model.embed_query(chunk)
                        chunk_embeddings.append(chunk_emb)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as inner_err:
                        logging.warning(f"chunk embedding失败, 跳过该chunk: {inner_err}")
                        gc.collect()
                        torch.cuda.empty_cache()
                if not chunk_embeddings:
                    raise
                embedding_1024 = np.mean(chunk_embeddings, axis=0).tolist()

            embeddings_list.append(embedding_1024)
        
        # 拼接所有1024维向量成1024*n维向量
        # Concatenate all 1024-dimensional vectors into 1024*n dimensional vector
        # final_embedding = np.concatenate(embeddings_list)
        
        return np.array(embeddings_list)
    
    def _calculate_similarity_with_llm(self, fingerprint1: Dict[str, Any], 
                                       fingerprint2: Dict[str, Any],
                                       langchain_version: str = "1.2") -> float:
        """
        使用LLM Agent计算两个指纹之间的相似度
        Calculate similarity between two fingerprints using LLM Agent
        
        Args:
            fingerprint1: 第一个设备指纹
                          First device fingerprint
            fingerprint2: 第二个设备指纹
                          Second device fingerprint
            langchain_version: langchain版本
                               langchain version
            
        Returns:
            相似度分数 (0-1之间)
            Similarity score (between 0-1)
        """
        try:
            if langchain_version != "1.2":
                # 使用Langchain Agent v0.3.27
                from langchain.agents import initialize_agent, AgentType
                from langchain.prompts import ChatPromptTemplate
                
                prompt = f"""
                You are an expert in IoT device fingerprint analysis.
                Please calculate the similarity score between two device fingerprints.
                
                Fingerprint 1:
                {json.dumps(fingerprint1, indent=2, ensure_ascii=False)}
                
                Fingerprint 2:
                {json.dumps(fingerprint2, indent=2, ensure_ascii=False)}
                
                Please return ONLY a similarity score between 0 and 1, where:
                - 1.0 means identical fingerprints
                - 0.0 means completely different fingerprints
                
                Consider all features except IP address.
                Return only the numeric score, no explanation.
                """
                
                agent_prompt = ChatPromptTemplate.from_template(template=prompt)
                messages = agent_prompt.format_messages()
                
                self.agent = initialize_agent(
                    tools=[],
                    llm=self.llm,
                    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                    verbose=False
                )
                
                response = self.agent.invoke(messages)
                score_str = response["output"].strip()
                
            else:
                # 使用Langchain v1.2+
                # Use Langchain v1.2+
                from pydantic import BaseModel
                from langchain.agents import create_agent
                from langchain.messages import SystemMessage
                from langchain.agents.structured_output import ProviderStrategy
                
                class SimilarityScore(BaseModel):
                    score: float
                    
                    def to_dict(self) -> dict:
                        return {"score": self.score}
                
                prompt = f"""
                Calculate the similarity score between two IoT device fingerprints.
                
                Fingerprint 1:
                {json.dumps(fingerprint1, indent=2, ensure_ascii=False)}
                
                Fingerprint 2:
                {json.dumps(fingerprint2, indent=2, ensure_ascii=False)}
                
                Return a similarity score between 0 and 1.
                Consider all features except IP address.
                """
                
                question = {"messages": [{"role": "user", "content": prompt}]}
                
                agent = create_agent(
                    model=self.llm,
                    system_prompt=SystemMessage(
                        content=[{
                            "type": "text",
                            "text": "You are an expert in IoT device fingerprint analysis."
                        }]
                    ),
                    response_format=ProviderStrategy(SimilarityScore)
                )
                
                response = agent.invoke(question)
                score_str = str(response["structured_response"].score)
            
            # 解析分数
            score = float(score_str)
            return max(0.0, min(1.0, score))  # Ensure within 0-1 range
            
        except Exception as e:
            print(f"LLM相似度计算错误: {e}")
            # 降级为简单的特征匹配
            # Fallback to simple feature matching
            return self._simple_similarity(fingerprint1, fingerprint2)
    
    def _simple_similarity(self, fingerprint1: Dict[str, Any], 
                          fingerprint2: Dict[str, Any]) -> float:
        """
        简单的相似度计算(降级方案)
        Simple similarity calculation (fallback solution)
        
        Args:
            fingerprint1: 第一个设备指纹
                          First device fingerprint
            fingerprint2: 第二个设备指纹
                          Second device fingerprint
            
        Returns:
            相似度分数 (0-1之间)
            Similarity score (between 0-1)
        """
        # 获取共同的特征键
        common_keys = set(fingerprint1.keys()) & set(fingerprint2.keys())
        
        if not common_keys:
            return 0.0
        
        # 计算匹配的特征数量
        matches = 0
        for key in common_keys:
            if key.lower() == 'ip':
                continue
            if fingerprint1[key] == fingerprint2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def community_retrieval(self, query_fingerprint: Dict[str, Any], 
                           similar_ips: List[Dict[str, Any]],
                           langchain_version: str = "1.2") -> Dict[str, Any]:
        """
        综合视角全局检索: 基于相似设备IP找到所属cluster, 并匹配cluster报告
        Comprehensive global retrieval: find clusters based on similar device IPs and match cluster reports
        
        Args:
            query_fingerprint: 查询设备的指纹信息
                               Query device fingerprint information
            similar_ips: 局部检索返回的相似设备列表
                         List of similar devices returned by local retrieval
            langchain_version: langchain版本
                               langchain version
            
        Returns:
            包含cluster匹配结果的字典
            Dictionary containing cluster matching results
        """
        print("=== 开始社区检索 ===")
        logging.info("=== 开始社区检索 ===")
        
        try:
            t_start = time.time()

            # 找到相似IP对应的cluster
            # Find clusters corresponding to similar IPs
            cluster_matches = {}
            
            for similar_device in similar_ips:
                device_type = similar_device.get("device_type")
                ip = similar_device.get("ip")
                
                if device_type is None or ip is None:
                    print(f"警告: 设备缺少device_type或ip信息, 跳过")
                    continue
                
                # 读取对应device_type的cluster文件
                # Read cluster file for corresponding device_type
                cluster_file = os.path.join(
                    self.com_view_path,
                    f"ipraw_{device_type}_embedding_overall_pca.csv"
                )
                
                if not os.path.exists(cluster_file):
                    print(f"警告: Cluster文件不存在 {cluster_file}, 跳过")
                    continue
                
                # 读取cluster数据
                # Read cluster data 
                cluster_df = pd.read_csv(cluster_file, low_memory=False)
                
                print("cluster数据读取完毕")

                # 根据IP定位到该IP所属的cluster
                # Locate the cluster for this IP using IP address
                ip_rows = cluster_df[cluster_df['ip'] == ip]
                if ip_rows.empty:
                    print(f"警告: IP {ip} 在cluster文件中未找到, 跳过")
                    continue
                
                cluster_id = ip_rows.iloc[0].get('cluster', -1)
                print(f"相似设备 {ip} {device_type} 所属cluster: {cluster_id}")
                
                # 使用(dev, cluster_id)作为唯一键, 避免重复
                # Use (dev, cluster_id) as unique key to avoid duplicates
                key = f"{device_type}_{cluster_id}"
                
                if key not in cluster_matches:
                    # 读取cluster报告 (只在第一次遇到该device_type时读取)
                    # Read cluster report (only when first encountering this device_type)
                    report_file = os.path.join(
                        self.com_view_path,
                        f"{device_type}_cluster_summaries.json"
                    )
                    
                    if os.path.exists(report_file):
                        with open(report_file, 'r', encoding='utf-8') as f:
                            cluster_reports = json.load(f)
                        
                        # 创建cluster报告字典
                        # Create cluster report dictionary
                        report_dict = {
                            report["cluster_id"]: report["analysis"] 
                            for report in cluster_reports
                        }
                        report = report_dict.get(int(cluster_id), "No report available")
                    else:
                        print(f"警告: 报告文件不存在 {report_file}")
                        report = "No report available"
                    
                    cluster_matches[key] = {
                        "device_type": device_type,
                        "cluster_id": int(cluster_id),
                        "ips": [ip],
                        "report": report
                    }
                else:
                    cluster_matches[key]["ips"].append(ip)
            
            t_cluster_lookup = time.time()

            print(f"找到 {len(cluster_matches)} 个唯一的cluster")
            logging.info(f"找到 {len(cluster_matches)} 个唯一的cluster")
            
            # 使用LLM匹配每个cluster报告与查询指纹
            # Use LLM to match each cluster report with query fingerprint
            matched_clusters = []
            
            for key, cluster_info in cluster_matches.items():
                print(f"匹配cluster {cluster_info['cluster_id']}...")
                
                # 调用LLM来匹配query_fingerprint和每个cluster报告。
                match_result = self._match_fingerprint_with_cluster(
                    query_fingerprint,
                    cluster_info["report"],
                    langchain_version
                )
                
                matched_clusters.append({
                    "device_type": cluster_info["device_type"],
                    "cluster_id": cluster_info["cluster_id"],
                    "similarity_score": match_result["similarity_score"],
                    "report": cluster_info["report"],
                    "matched_features": match_result["matched_features"],
                    "unmatched_features": match_result["unmatched_features"],
                    "related_ips": cluster_info["ips"]
                })
            
            # 按相似度排序
            matched_clusters.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            t_end = time.time()
            timing = {
                "cluster_lookup_sec": round(t_cluster_lookup - t_start, 4),
                "llm_matching_sec": round(t_end - t_cluster_lookup, 4),
                "total_sec": round(t_end - t_start, 4),
            }
            print(f"  耗时: cluster_lookup={timing['cluster_lookup_sec']}s, llm_matching={timing['llm_matching_sec']}s, total={timing['total_sec']}s")
            logging.info(f"  耗时: {timing}")

            result = {
                "query_fingerprint": query_fingerprint,
                "device_type": device_type,
                "total_clusters": len(matched_clusters),
                "matched_clusters": matched_clusters,
                "timing": timing
            }
            
            # 记录检索历史
            self.retrieval_history.append({
                "type": "community_retrieval",
                "timestamp": pd.Timestamp.now().isoformat(),
                "result": result
            })
            
            print(f"=== 社区检索完成, 匹配 {len(matched_clusters)} 个cluster ===\n")
            logging.info(f"=== 社区检索完成, 匹配 {len(matched_clusters)} 个cluster ===")
            
            return result
            
        except Exception as e:
            print(f"社区检索错误: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _match_fingerprint_with_cluster(self, query_fingerprint: Dict[str, Any],
                                       cluster_report: str,
                                       langchain_version: str = "1.2") -> Dict[str, Any]:
        """
        使用LLM Agent匹配设备指纹与cluster报告
        Match device fingerprint with cluster report using LLM Agent
        
        Args:
            query_fingerprint: 查询设备指纹
                               Query device fingerprint
            cluster_report: cluster报告文本
                            cluster report text
            langchain_version: langchain版本
                               langchain version
            
        Returns:
            包含相似度分数, 匹配特征和不匹配特征的字典
            Dictionary containing similarity score, matched features and unmatched features
        """
        try:
            # 过滤掉IP字段
            # Filter out IP field
            filtered_fingerprint = {
                k: v for k, v in query_fingerprint.items()
                if k.lower() != 'ip'
            }
            fingerprint_str = json.dumps(filtered_fingerprint, indent=2, ensure_ascii=False)
            report_str = cluster_report if cluster_report else "No report available"

            # system 角色提示词: 定义专家身份、评分标准和分析维度
            # system role prompt: define expert identity, scoring criteria, and analysis dimensions
            system_prompt = (
                "You are a senior IoT network security analyst specializing in device fingerprinting "
                "and traffic-based device identification. Your task is to evaluate how well an unknown "
                "device's network fingerprint matches a known device cluster profile.\n\n"
                "## Scoring Rubric\n"
                "Assign a similarity_score between 0.0 and 1.0 using this rubric:\n"
                "  - 0.9 - 1.0 : Almost all perspectives match; strong evidence of same device type\n"
                "  - 0.7 - 0.9 : Majority of perspectives match; likely same device category\n"
                "  - 0.4 - 0.7 : Partial match; some shared traits but notable differences\n"
                "  - 0.1 - 0.4 : Weak match; only a few superficial similarities\n"
                "  - 0.0 - 0.1 : No meaningful similarity\n\n"
                "## Output Requirements\n"
                "Return ONLY a JSON object with exactly three keys:\n"
                "  - similarity_score (float): overall similarity per the rubric above\n"
                "  - matched_features (list[str]): each string describes ONE matching perspective, "
                "e.g. 'OS: both run Linux-based firmware'\n"
                "  - unmatched_features (list[str]): each string describes ONE mismatching perspective, "
                "e.g. 'Cert: device uses self-signed cert while cluster uses Let\'s Encrypt'\n"
                "Do NOT include any explanation outside the JSON object."
            )

            # user 角色提示词: 提供具体数据
            # user role prompt: provide concrete data
            user_prompt = (
                "Please analyze the following device fingerprint against the cluster report "
                "and return the JSON result.\n\n"
                "### Device Fingerprint\n"
                f"```json\n{fingerprint_str}\n```\n\n"
                "### Cluster Report\n"
                f"```\n{report_str}\n```"
            )

            messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
            analysis_json = self.llm.chat_with_llm(self.used_llm_model, messages, whether_json=True)

            matched_features = analysis_json.get("matched_features", "")
            unmatched_features = analysis_json.get("unmatched_features", "")

            if not matched_features or not unmatched_features:
                logging.error(f"{self.used_llm_model} Failed to generate result in correct format!")
                raise ValueError(f"LLM错误生成结果: {analysis_json}")

            # if langchain_version != "1.2":
            #     # 使用Langchain Agent v0.3.27
            #     from langchain.agents import initialize_agent, AgentType
            #     from langchain.output_parsers import StructuredOutputParser, ResponseSchema
            #     from langchain.prompts import ChatPromptTemplate
            #     from langchain.schema import SystemMessage, HumanMessage

            #     # 定义输出格式
            #     # Define output format
            #     score_schema = ResponseSchema(
            #         name="similarity_score",
            #         description="Similarity score between 0 and 1, following the rubric"
            #     )
            #     matched_schema = ResponseSchema(
            #         name="matched_features",
            #         description="List of features that match the cluster pattern, one string per perspective"
            #     )
            #     unmatched_schema = ResponseSchema(
            #         name="unmatched_features",
            #         description="List of features that do not match the cluster pattern, one string per perspective"
            #     )

            #     response_schemas = [score_schema, matched_schema, unmatched_schema]
            #     output_parser = StructuredOutputParser(response_schemas=response_schemas)
            #     format_instructions = output_parser.get_format_instructions()

            #     messages = [
            #         SystemMessage(content=system_prompt_text),
            #         HumanMessage(content=(
            #             user_prompt_text + "\n\n"
            #             "Return your answer in the following JSON format:\n"
            #             f"{format_instructions}"
            #         ))
            #     ]

            #     self.agent = initialize_agent(
            #         tools=[],
            #         llm=self.llm,
            #         agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #         handle_parsing_errors=True,
            #         verbose=False
            #     )

            #     response = self.agent.invoke(messages)
            #     result = output_parser.parse(response["output"])
                
            # else:
            #     # 使用Langchain v1.2+
            #     # Use Langchain v1.2+
            #     from pydantic import BaseModel
            #     from langchain.agents import create_agent
            #     from langchain.messages import SystemMessage
            #     from langchain.agents.structured_output import ProviderStrategy

            #     class MatchResult(BaseModel):
            #         similarity_score: float
            #         matched_features: List[str]
            #         unmatched_features: List[str]

            #         def to_dict(self) -> dict:
            #             return {
            #                 "similarity_score": self.similarity_score,
            #                 "matched_features": self.matched_features,
            #                 "unmatched_features": self.unmatched_features
            #             }

            #     question = {"messages": [
            #         {"role": "system", "content": system_prompt_text},
            #         {"role": "user", "content": user_prompt_text}
            #     ]}

            #     agent = create_agent(
            #         model=self.llm,
            #         system_prompt=SystemMessage(
            #             content=[{
            #                 "type": "text",
            #                 "text": system_prompt_text
            #             }]
            #         ),
            #         response_format=ProviderStrategy(MatchResult)
            #     )

            #     response = agent.invoke(question)
            #     result = response["structured_response"].to_dict()
            
            return analysis_json
            
        except Exception as e:
            print(f"LLM匹配错误: {e}")
            # 降级方案
            # Fallback solution
            return {
                "similarity_score": 0.5,
                "matched_features": ["Unable to analyze with LLM"],
                "unmatched_features": []
            }
    
    def _extract_cluster_value_text(self, analysis: Any, feature_name: str = "") -> str:
        """
        从 analysis.common_patterns 中逐字段提取内容，构建 cluster_value_text。
        每个字段的 key 对应该 perspective 的某个特征维度，value 为该 cluster 的共同模式描述。
        Extract field contents from analysis.common_patterns field-by-field to build cluster_value_text.
        Each field key corresponds to a feature dimension of the perspective,
        and the value describes the common pattern of that cluster.

        Args:
            analysis: cluster 的 analysis 字典（应为 dict 类型）
                      cluster's analysis dict

        Returns:
            cluster_value_text: 拼接后的特征文本，格式为 "key: value\\n..."
                                Concatenated feature text in "key: value\\n..." format
        """
        if not isinstance(analysis, dict):
            return ''

        common_patterns = analysis.get('common_patterns', {})
        if not common_patterns:
            return ''
        
        # print("feature_name:", feature_name)
        # print("common_patterns:", common_patterns)

        parts = []

        if isinstance(common_patterns, dict):
            for key, value in common_patterns.items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_str = "; ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    value_str = str(value)
                parts.append(f"{key}: {value_str}")
        elif isinstance(common_patterns, list):
            parts = common_patterns

        return "\n".join(parts)

    def _compute_similarity_llm(self, query_feature: Dict[str, str], cluster_value_text: str) -> float:
        """
        使用 LLM 计算 query_feature 与 cluster_value_text 之间的相似度，仅返回 similarity 分数。
        Use LLM to compute similarity between query_feature and cluster_value_text,
        returning only the similarity score.

        Args:
            query_feature: 查询设备的特征字典 {col_name: value}
                           Query device feature dict {col_name: value}
            cluster_value_text: 从 common_patterns 提取的 cluster 特征文本
                                Cluster feature text extracted from common_patterns

        Returns:
            similarity: 相似度分数 (0.0 ~ 1.0)
                        Similarity score (0.0 ~ 1.0)
        """
        query_feature_str = "\n".join(
            f"{col}: {val}" for col, val in query_feature.items() if val and val != ''
        )

        system_prompt = (
            "You are an IoT device fingerprint analysis expert. "
            "Evaluate how similar a device's observed feature values are to a cluster's common patterns. "
            "Return ONLY a valid JSON object with a single key 'similarity' (a float between 0.0 and 1.0).\n"
            "Scoring guide:\n"
            "  0.9 - 1.0: Feature values almost perfectly match the cluster pattern\n"
            "  0.7 - 0.9: Strong match with minor differences\n"
            "  0.4 - 0.7: Partial match with notable differences\n"
            "  0.1 - 0.4: Weak match, mostly different\n"
            "  0.0 - 0.1: No meaningful similarity\n"
            "Do NOT include any explanation. Return ONLY: {\"similarity\": <float>}"
        )

        user_prompt = (
            "Compare the device's feature values against the cluster's common patterns "
            "and return only the similarity score.\n\n"
            f"### Device Feature Values\n{query_feature_str}\n\n"
            f"### Cluster Common Patterns\n{cluster_value_text}\n\n"
            "Return ONLY: {\"similarity\": <float>}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.llm.chat_with_llm(self.used_llm_model, messages, whether_json=True)
        return float(result.get('similarity', 0.0))

    def calculate_importance(self, matched_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算特征重要性：分析综合视角cluster中的设备在单特征聚类下的聚集程度
        Calculate feature importance: analyze how devices in comprehensive clusters 
        aggregate in single-feature clustering
        
        Args:
            matched_clusters: 社区检索返回的匹配cluster列表
                            List of matched clusters from community retrieval
        
        Returns:
            特征重要性分析结果
            Feature importance analysis results
        """
        print("=== 开始计算特征重要性 ===")
        
        importance_results = {}
        
        # 对每个matched cluster进行分析
        # Analyze each matched cluster
        for cluster_info in matched_clusters:
            cluster_key = f"{cluster_info['device_type']}_{cluster_info['cluster_id']}"
            cluster_ips = set(cluster_info.get('related_ips', []))
            
            print(f"\n分析cluster: {cluster_key}, 包含 {len(cluster_ips)} 个IP")
            
            feature_importance = {}
            
            # 遍历每个单特征perspective
            # Iterate through each single-feature perspective
            for perspective_name in self.retrieval_perspective_names:
                try:
                    # 读取单特征聚类文件
                    # Read single-feature clustering file
                    single_feature_file = os.path.join(
                        self.single_view_path,
                        f"embedding_{perspective_name}",
                        f"ipraw_{cluster_info['device_type']}_embedding_{perspective_name}_pca.csv"
                    )
                    
                    if not os.path.exists(single_feature_file):
                        print(f"  警告: 文件不存在 {single_feature_file}")
                        continue
                    
                    # 读取cluster分配
                    # Read cluster assignments
                    df = pd.read_csv(single_feature_file, usecols=['ip', 'cluster'], low_memory=False)
                    
                    # 在single-perspective cluster中筛选出综合视角cluster中的IP
                    # Filter IPs of comprehensive cluster in single-perspective cluster
                    df_filtered = df[df['ip'].isin(cluster_ips)]
                    
                    if len(df_filtered) == 0:
                        continue
                    
                    # 统计comprehensive-view cluster中的IP 在 single-perspective cluster中的分布 {cluster_id: count}
                    # Count distribution of these IPs in single-feature clusters
                    cluster_counts = df_filtered['cluster'].value_counts().to_dict()
                    
                    # 移除噪声点 (cluster = -1)
                    # Remove noise points (cluster = -1)
                    if -1 in cluster_counts:
                        del cluster_counts[-1]
                    
                    if len(cluster_counts) == 0:
                        continue
                    
                    # 计算Shannon Entropy
                    # Calculate Shannon Entropy
                    M = df['cluster'].nunique()  # 单特征cluster的unique数量,也就是总数
                    total_count = sum(cluster_counts.values())  # comprehensive-view cluster中的IP总数
                    
                    # 计算 cluster i(cid)的权重分数 w_i
                    # Calculate probability distribution w_i
                    cluster_weight = {cid: count / total_count for cid, count in cluster_counts.items()}

                    #  # 对 cluster_weight 进行 Softmax 归一化
                    # # Softmax normalization for cluster_weight
                    # cw_arr = np.array(list(cluster_weight.values()))
                    # cw_exp = np.exp(cw_arr - np.max(cw_arr))  # 减去最大值保证数值稳定
                    # cw_softmax = cw_exp / cw_exp.sum()
                    # cluster_weight = {cid: float(sw) for cid, sw in zip(cluster_weight.keys(), cw_softmax)}

                    # 计算归一化Shannon熵 S_f
                    # Calculate normalized Shannon entropy S_f
                    entropy = 0.0
                    for w_i in cluster_weight.values():
                        if w_i > 0:
                            entropy -= w_i * np.log(w_i)
                    
                    if M > 1:
                        normalized_entropy = entropy / np.log(M)
                    else:
                        normalized_entropy = 0.0
                    
                    important_score = 1 - normalized_entropy
                    
                    print(f"  特征 {perspective_name}: 未归一化重要性分数={important_score:.4f}, 归一化熵={normalized_entropy:.4f}, cluster总数={M}")

                    # 读取single-perspective cluster报告以提取特征值
                    # Read single-perspective cluster reports to extract feature values
                    summary_file = os.path.join(
                        self.single_view_path,
                        f"embedding_{perspective_name}",
                        f"{cluster_info['device_type']}_cluster_summaries.json"
                    )
                    
                    feature_values = []
                    if os.path.exists(summary_file):
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summaries = json.load(f)
                        
                        # 为每个cluster提取特征值
                        # Extract feature values for each cluster
                        for cid, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
                            # 找到对应的cluster summary
                            # Find corresponding cluster summary
                            cluster_summary = next((s for s in summaries if s['cluster_id'] == cid), None)
                            if cluster_summary:
                                feature_values.append({
                                    "cluster_id": cid,
                                    "weight": cluster_weight[cid],
                                    "device_count": count,
                                    "analysis": cluster_summary.get('analysis', '')
                                })
                    
                    # 保存特征重要性信息
                    # Save feature importance information
                    feature_importance[perspective_name] = {
                        "importance_score": important_score,  # 重要性分数，越高越重要
                        "is_important": normalized_entropy < 0.1,  # 阈值0.1，低于此值认为重要
                        "num_clusters": M,  # 该 single-perspective有多少个clusters
                        "cluster_distribution": cluster_counts, # IP分布于各个cluster中的数量
                        "cluster_weight": cluster_weight,  # cluster counts中每个该single-perspective cluster的权重
                        "feature_values": feature_values  # cluster代表的特征值，由该single-perspective cluster的report体现
                    }
                    
                except Exception as e:
                    print(f"  处理特征 {perspective_name} 时出错: {e}")
                    continue
            
            # 对 importance_score 进行 L1 归一化（使所有 perspective 的重要性分数之和为 1）
            # L1 normalization for importance_score (make all perspective importance scores sum to 1)
            raw_scores = {name: info['importance_score'] for name, info in feature_importance.items()}
            total_score = sum(raw_scores.values())
            if total_score > 0:
                for name in feature_importance:
                    feature_importance[name]['importance_score'] = raw_scores[name] / total_score
                    # print(f"  特征 {name}: 重要性分数={feature_importance[name]['importance_score']:.4f}, M={feature_importance[name]['num_clusters']}")

            # 按重要性分数排序
            # Sort by importance score
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1]['importance_score']
            )
            
            importance_results[cluster_key] = {
                "cluster_info": cluster_info,
                "feature_importance": dict(sorted_features),
                "important_features": [f for f, info in sorted_features if info['is_important']]
            }
        
        print("\n=== 特征重要性计算完成 ===")
        return importance_results
    
    def reasoning_path_retrieval(self, local_result: Dict[str, Any], 
                                community_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理路径检索: 分析相似设备因哪些关键特征落入cluster，并计算路径匹配分数
        Reasoning path retrieval: analyze which key features cause similar devices 
        to fall into clusters and calculate path matching scores
        
        Args:
            local_result: 局部检索结果
                        Local retrieval result
            community_result: 社区检索结果
                            Community retrieval result
        
        Returns:
            推理路径分析结果
            Reasoning path analysis results
        """
        print("\n=== 开始推理路径检索 ===")
        
        try:
            # 提取参数
            # Extract parameters
            query_fingerprint = local_result.get('query_fingerprint', {})
            matched_clusters = community_result.get('matched_clusters', [])
            
            if not matched_clusters:
                return {"error": "缺少必要的输入数据"}
            
            # 1. 计算特征重要性
            # 1. Calculate feature importance
            importance_results = self.calculate_importance(matched_clusters)

            # print(importance_results)
            
            # 2. 对每个cluster计算路径匹配分数
            # 2. Calculate path matching score for each cluster
            path_matching_results = []
            
            for cluster_key, cluster_data in importance_results.items():
                print(f"\n计算cluster {cluster_key} 的路径匹配分数")
                
                cluster_info = cluster_data['cluster_info']
                feature_importance = cluster_data['feature_importance']
                important_features = cluster_data['important_features']
                
                if not important_features:
                    print(f"  cluster {cluster_key} 没有重要特征，跳过")
                    continue
                
                # 计算路径匹配分数
                # Calculate path matching score
                path_score = 0.0
                feature_matching_details = []
                
                # 根据每个关键特征角度(perspective) （符号表示: f(a) ∈ A'， A'是关键特征角度集合）,
                # 计算其中（A) 每个single-perspective cluster （符号表示: C_i^𝑓(𝑎)）所表示的特征值 （符号表示: 𝑣_i^𝑓(𝑎)）
                # 和(B）待查询设备对应perspective的特征值 （符号表示: 𝑓_d^(𝑎)）之间的相似度，
                # 得到的相似度 𝑠𝑖𝑚(𝑓_d^(𝑎) , 𝑣_i^𝑓(𝑎)))乘以 𝑣_i^𝑓(𝑎) 的权重 𝑤_𝑖，并计算加权相似度和
                #  

                for feature_name in important_features:
                    feature_info = feature_importance[feature_name]
                    S_f = feature_info['importance_score']
                    feature_values = feature_info['feature_values']
                    
                    if not feature_values:
                        continue
                    
                    # 获取query_fingerprint中该特征的值
                    # Get feature value from query_fingerprint
                    feature_cols = self.perspective_info_config[feature_name]["cols"]
                    
                    query_feature = {
                        col: str(query_fingerprint.get(col, ""))
                        for col in feature_cols
                        if col in query_fingerprint and str(query_fingerprint.get(col, "")) != ""
                    }

                    # 构建 query 特征文本：逐字段 "col: value" 格式，与 common_patterns 结构对齐
                    # Build query feature text: "col: value" per field, aligned with common_patterns structure
                    query_feature_text = "\n".join(
                        f"{col}: {val}" for col, val in query_feature.items()
                    )

                    if not query_feature or not query_feature_text:
                        continue

                    # 在 embedding 模式下对 query 特征进行 embedding（每个 feature_name 仅计算一次）
                    # In embedding mode, embed query feature once per feature_name
                    if not self.reasoning_use_llm:
                        query_embedding = np.array(self.embedding_model.embed_query(query_feature_text))

                    # 计算与各个特征值的相似度加权和
                    # Calculate weighted sum of similarities with feature values
                    weighted_similarity_sum = 0.0
                    value_similarities = []


                    for fv in feature_values:
                        # 从 analysis.common_patterns 中逐字段提取 cluster_value_text
                        # Extract cluster_value_text field-by-field from analysis.common_patterns
                        analysis = fv.get('analysis') or {}
                        cluster_id = fv.get('cluster_id')

                        cluster_value_text = self._extract_cluster_value_text(analysis, feature_name)

                        if not cluster_value_text:
                            continue

                        if self.reasoning_use_llm:
                            # LLM 相似度计算策略：传入 query_feature dict 和 cluster_value_text，仅获取 similarity
                            # LLM-based similarity strategy: pass query_feature dict and cluster_value_text, get similarity only
                            try:
                                similarity = self._compute_similarity_llm(query_feature, cluster_value_text)
                            except Exception as llm_e:
                                print(f"  LLM 相似度计算失败: {llm_e}, 回退到 embedding 方法")
                                q_emb = np.array(self.embedding_model.embed_query(query_feature_text))
                                v_emb = np.array(self.embedding_model.embed_query(cluster_value_text))
                                similarity = cosine_similarity(
                                    q_emb.reshape(1, -1), v_emb.reshape(1, -1)
                                )[0][0]
                        else:
                            # Embedding 余弦相似度计算策略：对 cluster_value_text 进行 embedding 后计算余弦相似度
                            # Embedding cosine similarity strategy: embed cluster_value_text and compute cosine similarity
                            value_embedding = np.array(self.embedding_model.embed_query(cluster_value_text))
                            similarity = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                value_embedding.reshape(1, -1)
                            )[0][0]

                        # 加权
                        # Weight by probability
                        weighted_similarity = fv['weight'] * similarity  # singe-perspective 的某个cluster和待查询设备的相似度
                        weighted_similarity_sum += weighted_similarity

                        value_similarities.append({
                            "cluster_id": fv['cluster_id'],
                            "weight": fv['weight'],
                            "similarity": float(similarity),  # 待查询指纹和该single-perspective cluster的相似度
                            "weighted_similarity": float(weighted_similarity),
                            "device_count": fv['device_count'],
                            "analysis_preview": cluster_value_text[:300]
                        })
                    
                    # 特征角度级别的匹配分数 B
                    # Feature pespective-level matching score B  某个singe-perspective 的所有cluster和待查询设备的相似度总和
                    feature_matching_score = weighted_similarity_sum
                    
                    # 乘以特征重要性分数
                    # Multiply by feature importance score
                    weighted_feature_score = S_f * feature_matching_score
                    path_score += weighted_feature_score
                    
                    feature_matching_details.append({
                        "feature_name": feature_name,
                        "importance_score": float(S_f),
                        "feature_matching_score": float(feature_matching_score),
                        "weighted_feature_score": float(weighted_feature_score),
                        "value_similarities": sorted(value_similarities, 
                                                    key=lambda x: x['weight'], 
                                                    reverse=True)
                    })
                    
                    print(f"  特征 {feature_name}: 重要性={S_f:.4f}, 匹配分数={feature_matching_score:.4f}")
                
                # 保存该cluster的路径匹配结果
                # Save path matching result for this cluster
                path_matching_results.append({
                    "cluster_key": cluster_key,
                    "cluster_info": cluster_info,
                    "path_matching_score": float(path_score),
                    "important_features": [
                        {
                            "feature_name": f,
                            "importance_score": float(feature_importance[f]['importance_score']),
                            "num_clusters": feature_importance[f]['num_clusters']
                        }
                        for f in important_features
                    ],
                    "feature_matching_details": sorted(
                        feature_matching_details,
                        key=lambda x: x['importance_score']
                    )
                })
            
            # 按路径匹配分数排序
            # Sort by path matching score
            path_matching_results.sort(key=lambda x: x['path_matching_score'], reverse=True)
            
            result = {
                "query_fingerprint": query_fingerprint,
                "path_matching_results": path_matching_results,
                "summary": {
                    "total_clusters_analyzed": len(path_matching_results),
                    "top_cluster": path_matching_results[0] if path_matching_results else None
                }
            }
            
            # 记录到历史
            # Record to history
            self.retrieval_history.append({
                "type": "reasoning_path_retrieval",
                "timestamp": pd.Timestamp.now().isoformat(),
                "result": result
            })
            
            print(f"\n=== 推理路径检索完成, 分析了 {len(path_matching_results)} 个cluster ===")
            
            return result
            
        except Exception as e:
            print(f"推理路径检索错误: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def get_retrieval_history(self) -> List[Dict[str, Any]]:
        """
        获取检索历史记录
        Get retrieval history
        
        Returns:
            历史记录列表
            History record list
        """
        return self.retrieval_history
    
    def clear_history(self):
        """清空检索历史记录
        Clear retrieval history"""
        self.retrieval_history.clear()
    
    def save_history(self, filename: str = "retrieval_history.json"):
        """保存检索历史到单个文件 (兼容旧方式)
        Save retrieval history to a single file (backward compatible)"""
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_db"
        )
        os.makedirs(history_path, exist_ok=True)
        # 读取旧历史文件内容
        # Read old history file content
        if os.path.exists(os.path.join(history_path, filename)):
            with open(os.path.join(history_path, filename), 'r', encoding='utf-8') as f:
                old_history = json.load(f)
            # 新的历史拼接到旧的历史之后
            # Append new history to old history
            combined_history = old_history + self.retrieval_history
        else:
            combined_history = self.retrieval_history
        
        with open(os.path.join(history_path, filename), "w", encoding='utf-8') as f:
            json.dump(combined_history, f, indent=4, ensure_ascii=False)

    def save_history_by_type(self, device_name: str, 
                             local_result: Dict = None, 
                             community_result: Dict = None, 
                             reasoning_result: Dict = None):
        """
        按检索类型分别保存到 query_db/local/, query_db/community/, query_db/reasoning/
        Save retrieval results by type into separate folders under query_db.
        Each device saves one JSON file per retrieval type.
        
        Args:
            device_name: 设备类型名称, 如 "CAMERA" / Device type name
            local_result: 局部检索结果 / Local retrieval result
            community_result: 社区检索结果 / Community retrieval result
            reasoning_result: 推理路径检索结果 / Reasoning path retrieval result
        """
        query_db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_db"
        )
        
        type_map = {
            "local": local_result,
            "community": community_result,
            "reasoning": reasoning_result,
        }
        
        for type_name, result_data in type_map.items():
            if result_data is None:
                continue
            
            type_dir = os.path.join(query_db_path, type_name)
            os.makedirs(type_dir, exist_ok=True)
            
            filename = f"{device_name}_{type_name}.json"
            filepath = os.path.join(type_dir, filename)
            
            # 读取旧文件并追加
            # Read old file and append
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        old_data = json.load(f)
                    if not isinstance(old_data, list):
                        old_data = [old_data]
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"JSON文件损坏 {filepath}: {e}, 备份后重建")
                    print(f"警告: JSON文件损坏 {filepath}, 备份后重建")
                    import shutil
                    shutil.copy(filepath, filepath + ".bak")
                    old_data = []
            else:
                old_data = []
            
            # result_data可以是单条记录或列表
            # result_data can be a single record or a list
            if isinstance(result_data, list):
                old_data.extend(result_data)
            else:
                old_data.append(result_data)
            
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(old_data, f, indent=4, ensure_ascii=False)
            
            logging.info(f"已保存 {type_name} 检索结果到 {filepath}")

    def load_history_by_type(self, device_name: str, retrieval_type: str) -> Optional[List[Dict]]:
        """
        从分类保存的文件中加载指定设备和检索类型的历史结果
        Load history results for a specific device and retrieval type from categorized files
        
        Args:
            device_name: 设备类型名称 / Device type name
            retrieval_type: 检索类型 "local" / "community" / "reasoning"
        
        Returns:
            检索结果列表, 不存在则返回None
            List of retrieval results, or None if not found
        """
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_db",
            retrieval_type,
            f"{device_name}_{retrieval_type}.json"
        )
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_retrieval_result_by_type(self, ip: str, device_name: str):
        """
        从分类保存的文件中根据IP加载各类型检索结果
        Load retrieval results by IP from categorized files
        
        Args:
            ip: 要查询的IP地址 / IP address to query
            device_name: 设备类型名称 / Device type name
        
        Returns:
            (local_result, community_result, reasoning_result) 元组
        """
        local_result = None
        community_result = None
        reasoning_result = None
        
        for rtype, setter in [("local", "local_result"), 
                               ("community", "community_result"), 
                               ("reasoning", "reasoning_result")]:
            records = self.load_history_by_type(device_name, rtype)
            if records is None:
                continue
            for record in records:
                fp = record.get("query_fingerprint", {})
                if isinstance(fp, dict) and fp.get("ip") == ip:
                    if rtype == "local":
                        local_result = record
                    elif rtype == "community":
                        community_result = record
                    else:
                        reasoning_result = record
                    break
        
        return local_result, community_result, reasoning_result
    
    def load_history(self, filename: str = "retrieval_history.json"):
        """加载检索历史
        Load retrieval history"""
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_db",
            filename
        )
        
        if os.path.exists(history_path):
            with open(history_path, "r", encoding='utf-8') as f:
                self.retrieval_history = json.load(f)

    def load_retrieval_result(self, ip: str, filename: str = "retrieval_history.json"):
        """
        根据IP地址从历史记录中加载检索结果
        Load retrieval results from history based on IP address
        
        Args:
            ip: 要查询的IP地址 / IP address to query
            filename: 历史记录文件名 / History file name
            
        Returns:
            包含local_result, community_result, reasoning_result的元组
            Tuple containing local_result, community_result, reasoning_result
            如果未找到匹配记录，返回(None, None, None)
            Returns (None, None, None) if no matching record found
        """
        history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_db",
            filename
        )
        
        if not os.path.exists(history_path):
            print(f"警告: 历史记录文件不存在 {history_path}")
            return None, None, None
        
        try:
            with open(history_path, "r", encoding='utf-8') as f:
                history_data = json.load(f)
            
            # 查找type为"local_retrieval"且IP匹配的记录
            # Find record with type "local_retrieval" and matching IP
            local_result = None
            community_result = None
            reasoning_result = None
            local_index = -1
            
            for i, record in enumerate(history_data):
                if (record.get("type") == "local_retrieval" and 
                    record.get("result", {}).get("query_fingerprint", {}).get("ip") == ip):
                    local_result = record.get("result")
                    local_index = i
                    print(f"找到IP {ip} 的local_retrieval记录，索引: {i}")
                    break
            
            if local_result is None:
                print(f"未找到IP {ip} 的local_retrieval记录")
                return None, None, None
            
            # 从local_retrieval记录之后查找community_retrieval和reasoning_path_retrieval
            # Find community_retrieval and reasoning_path_retrieval records after local_retrieval
            for i in range(local_index + 1, len(history_data)):
                record = history_data[i]
                
                # 查找community_retrieval记录
                # Find community_retrieval record
                if record.get("type") == "community_retrieval" and community_result is None:
                    community_result = record.get("result")
                    print(f"找到community_retrieval记录，索引: {i}")
                
                # 查找reasoning_path_retrieval记录
                # Find reasoning_path_retrieval record
                if record.get("type") == "reasoning_path_retrieval" and reasoning_result is None:
                    reasoning_result = record.get("result")
                    print(f"找到reasoning_path_retrieval记录，索引: {i}")
                
                # 如果两个都找到了，可以提前退出
                # Exit early if both are found
                if community_result is not None and reasoning_result is not None:
                    break
            
            if community_result is None:
                print(f"警告: 未找到IP {ip} 对应的community_retrieval记录")
            
            if reasoning_result is None:
                print(f"警告: 未找到IP {ip} 对应的reasoning_path_retrieval记录")
            
            return local_result, community_result, reasoning_result
            
        except Exception as e:
            print(f"加载检索结果时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def run_retrieval_algorithm(self, test_fingerprint, top_k=5, 
                                whether_local=True, 
                                whether_community=True, 
                                whether_reasoning=True,
                                local_result=None, community_result=None,
                                llm_type="deepseek", device_name=None):
        """
        运行检索算法，返回局部检索、社区检索和推理路径检索的结果
        Run retrieval algorithm and return local retrieval, community retrieval and reasoning path retrieval results
        
        Args:
            test_fingerprint: 查询指纹 / Query fingerprint
            top_k: 返回top_k个结果 / Return top_k results
            whether_local: 是否执行局部检索 / Whether to run local retrieval
            whether_community: 是否执行社区检索 / Whether to run community retrieval
            whether_reasoning: 是否执行推理路径检索 / Whether to run reasoning path retrieval
            local_result: 已有的局部检索结果(可选) / Existing local result (optional)
            community_result: 已有的社区检索结果(可选) / Existing community result (optional)
            llm_type: LLM类型 / LLM type
            device_name: 设备类型名称, 用于分类保存 / Device type name for categorized saving
        """
        reasoning_result = None

        if whether_local:        
            print("=== 局部检索(Local Entity Retrieval)开始 ===")
            logging.info("局部检索(Local Entity Retrieval)开始")

            local_result = self.local_retrieval(test_fingerprint, top_k=top_k)
            
            num_similar = len(local_result.get('similar_devices', []))
            print(f"局部检索结果: 找到 {num_similar} 个相似设备\n")
            logging.info(f"局部检索结果: 找到 {num_similar} 个相似设备")

            print("=== 局部检索(Local Entity Retrieval)结束 ===")
            logging.info("局部检索(Local Entity Retrieval)结束")

        if whether_community:
            print("=== 社区检索(Community Retrieval)开始 ===")
            logging.info("社区检索(Community Retrieval)开始")

            community_result = None

            if local_result and local_result.get('similar_devices'):
                if llm_type == "deepseek":
                    community_result = self.community_retrieval(
                        test_fingerprint,
                        local_result['similar_devices'],
                        langchain_version="0.3.27"
                    )
                else:
                    community_result = self.community_retrieval(
                        test_fingerprint,
                        local_result['similar_devices']
                    )
            
            if community_result:
                num_clusters = len(community_result.get('matched_clusters', []))
                print(f"社区检索结果: 匹配 {num_clusters} 个cluster\n")
                logging.info(f"社区检索结果: 匹配 {num_clusters} 个cluster")

            print("=== 社区检索(Community Retrieval)结束 ===")
            logging.info("社区检索(Community Retrieval)结束")

        if whether_reasoning:
            print("=== 推理路径检索(Reasoning Path Retrieval)开始 ===")
            logging.info("推理路径检索(Reasoning Path Retrieval)开始")

            if community_result and community_result.get('matched_clusters'):
                reasoning_result = self.reasoning_path_retrieval(
                    local_result,
                    community_result
                )
        
            if reasoning_result and 'error' not in reasoning_result:
                print(f"\n推理路径检索结果:")
                print(f"  分析了 {reasoning_result['summary']['total_clusters_analyzed']} 个cluster")
                
                if reasoning_result['summary']['top_cluster']:
                    top = reasoning_result['summary']['top_cluster']
                    print(f"  最佳匹配cluster: {top['cluster_key']}")
                    print(f"  路径匹配分数: {top['path_matching_score']:.4f}")
                    print(f"  重要特征数量: {len(top['important_features'])}")
                    
                    if top['important_features']:
                        print(f"\n  重要特征列表:")
                        for feat in top['important_features'][:5]:
                            print(f"    - {feat['feature_name']}: 重要性={feat['importance_score']:.4f}")
            elif reasoning_result:
                print(f"推理路径检索错误: {reasoning_result.get('error')}")
                logging.error(f"推理路径检索错误: {reasoning_result.get('error')}")

            print("=== 推理路径检索(Reasoning Path Retrieval)结束 ===")
            logging.info("推理路径检索(Reasoning Path Retrieval)结束")

        # 分类保存检索结果
        # Save retrieval results by type
        if device_name:
            self.save_history_by_type(
                device_name=device_name,
                local_result=local_result if whether_local else None,
                community_result=community_result if whether_community else None,
                reasoning_result=reasoning_result
            )
            print(f"\n检索结果已按类型保存到 query_db/local|community|reasoning/{device_name}_*.json")
            logging.info(f"检索结果已按类型保存 (device_name={device_name})")
        else:
            self.save_history()
            print("\n检索历史已保存")
            logging.info("检索历史已保存到 retrieval_history.json")

        return local_result, community_result, reasoning_result

    
if __name__ == "__main__":
    pass