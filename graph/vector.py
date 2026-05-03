"""
vector.py - 将 platform_data/csv/local/1 下各 perspective 的 embedding CSV 
            通过 Milvus Lite 向量数据库进行保存

Save embedding CSVs from platform_data/csv/local/1/embedding_{perspective}/
into a local Milvus vector database stored at platform_data/csv/local/1/vectorDB/

Usage:
    conda run -n probe python agent/vector.py
    conda run -n probe python agent/vector.py --perspectives as certificate
    conda run -n probe python agent/vector.py --devices CAMERA PRINTER
    conda run -n probe python agent/vector.py --batch_size 5000
    conda run -n probe python agent/vector.py --drop  # 重建所有collection
"""

import os
import re
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)

# ─── 路径配置 / Path config ───────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PATH = os.path.join(BASE_PATH, "platform_data", "csv", "local", "1")
VECTOR_DB_DIR = os.path.join(LOCAL_PATH, "vectorDB")
MILVUS_DB_FILE = os.path.join(VECTOR_DB_DIR, "milvus.db")
RAG_DOMAIN_FILE = os.path.join(BASE_PATH, "rag_domain.json")

EMBEDDING_DIM = 1024
EMBEDDING_OVERALL_DIM = 1024*11
MAX_IP_LENGTH = 64
MAX_DEVICE_LENGTH = 64
BATCH_SIZE = 5000  # 每批插入条数 / rows per insert batch

perspective_info_config = load_perspective_info()


# ─── 日志 / Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_device_names(rag_domain_path: str = RAG_DOMAIN_FILE) -> List[str]:
    """
    从 rag_domain.json 读取所有设备名称
    Load all device names from rag_domain.json
    """
    with open(rag_domain_path, "r") as f:
        domain = json.load(f)
    devices = []
    for category_devices in domain.values():
        devices.extend(category_devices)
    return sorted(set(devices))


def get_device_csv_files(local_path: str, perspective: str, device_names: List[str]) -> List[dict]:
    """
    根据 rag_domain.json 中的设备名称, 获取某个 perspective 下各设备的 embedding CSV 文件(排除 PCA 版本)
    Get embedding CSV files for each device under a perspective directory, excluding PCA versions.
    Device names are sourced from rag_domain.json.
    Returns list of {device, filepath}
    """
    emb_dir = os.path.join(local_path, f"embedding_{perspective}")
    if not os.path.isdir(emb_dir):
        return []

    results = []
    for device in sorted(device_names):
        fname = f"ipraw_{device}_embedding_{perspective}.csv"
        fpath = os.path.join(emb_dir, fname)
        if os.path.isfile(fpath):
            results.append({
                "device": device,
                "filepath": fpath,
            })
    return results


def create_collection(client: MilvusClient, perspective: str, drop: bool = False, col_name: str = None, embedding_dim: int = 1024):
    """
    为某个 perspective 创建 Milvus collection
    Schema: id(int64 pk auto), ip(varchar), device_type(varchar), embedding(float_vector 1024)
    """
    if col_name is None:
        col_name = f"embedding_{perspective}"

    if drop and client.has_collection(col_name):
        logging.info(f"Dropping existing collection: {col_name}")
        client.drop_collection(col_name)

    if client.has_collection(col_name):
        logging.info(f"Collection already exists: {col_name}")
        return col_name

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("ip", DataType.VARCHAR, max_length=MAX_IP_LENGTH)
    schema.add_field("device_type", DataType.VARCHAR, max_length=MAX_DEVICE_LENGTH)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim)

    # 创建 collection
    client.create_collection(
        collection_name=col_name,
        schema=schema,
    )

    # 创建向量索引 (IVF_FLAT for local use)
    index_params = client.prepare_index_params()
    if perspective == "overall":
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 32, "efConstruction": 64},
        )
    else:
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
    client.create_index(
        collection_name=col_name,
        index_params=index_params,
    )

    logging.info(f"Created collection: {col_name} (dim={EMBEDDING_DIM})")
    return col_name


def insert_csv_to_collection(
    client: MilvusClient,
    col_name: str,
    device: str,
    filepath: str,
    batch_size: int = BATCH_SIZE,
):
    """
    读取 CSV 文件并批量插入到 Milvus collection
    """
    logging.info(f"Loading CSV: {os.path.basename(filepath)} (device={device})")

    # 分块读取大文件 / read large files in chunks
    total_inserted = 0
    for chunk_df in pd.read_csv(filepath, chunksize=batch_size):
        ip_list = chunk_df["ip"].astype(str).tolist()
        embedding_cols = [c for c in chunk_df.columns if c.startswith("embedding")]
        embeddings = chunk_df[embedding_cols].values.tolist()

        data = [
            {
                "ip": ip_list[i],
                "device_type": device,
                "embedding": embeddings[i],
            }
            for i in range(len(ip_list))
        ]

        client.insert(collection_name=col_name, data=data)
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


def check_existing_count(client: MilvusClient, col_name: str, device: str) -> int:
    """检查 collection 中某个 device 已有的记录数"""
    if not client.has_collection(col_name):
        return 0
    try:
        results = client.query(
            collection_name=col_name,
            filter=f'device_type == "{device}"',
            output_fields=["ip"],
            limit=1,
        )
        # 如果能查到至少一条，说明已经有数据
        return len(results)
    except Exception:
        return 0


def parse_log_for_resume(log_path: str) -> Optional[dict]:
    """
    解析 store_vector.log, 找到最后一个未完成的 (perspective, device)
    Parse store_vector.log to find the last incomplete (perspective, device).
    
    逻辑: 从日志尾部向前搜索, 找到最后一条 "Loading CSV" 行,
    如果之后没有对应的 "Finished" 行, 说明该文件处理中断了。
    
    Returns:
        dict with keys {perspective, device, collection, inserted_so_far} or None if all complete.
    """
    if not os.path.exists(log_path):
        logging.warning(f"Log file not found: {log_path}")
        return None

    loading_pattern = re.compile(
        r"Loading CSV: ipraw_(.+)_embedding_(.+)\.csv \(device=(.+)\)"
    )
    finished_pattern = re.compile(
        r"Finished ipraw_(.+)_embedding_(.+)\.csv: (\d+) rows inserted into (.+)"
    )
    inserted_pattern = re.compile(
        r"Inserted batch: \d+ rows \(total so far: (\d+)\)"
    )

    last_loading = None
    last_loading_line_num = -1  # log中记录的最后加载csv文件的行号
    finished_after_last_loading = False  # 加载后是否完成
    last_inserted_total = 0  # 最后插入的总行数

    with open(log_path, "r") as f:
        for line_num, line in enumerate(f):
            # 1. 匹配所有加载过的device和perspective对应的csv文件
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
            
            # 2. 匹配所有完成的device和perspective对应的csv文件
            m_fin = finished_pattern.search(line)
            if m_fin and last_loading and line_num > last_loading_line_num:
                if (m_fin.group(1) == last_loading["device"]
                        and m_fin.group(2) == last_loading["perspective"]):
                    finished_after_last_loading = True

            # 3. 匹配已经插入的行数
            m_ins = inserted_pattern.search(line)
            if m_ins and last_loading and line_num > last_loading_line_num:
                last_inserted_total = int(m_ins.group(1))

    if last_loading and not finished_after_last_loading:
        last_loading["inserted_so_far"] = last_inserted_total
        return last_loading

    return None


def delete_device_from_collection(client: MilvusClient, col_name: str, device: str) -> int:
    """
    删除 collection 中某个 device 的所有记录
    Delete all records for a specific device from a collection.
    
    Returns: 删除的记录数 (approximate)
    """
    if not client.has_collection(col_name):
        logging.warning(f"Collection {col_name} does not exist, nothing to delete")
        return 0

    logging.info(f"Deleting all records for device '{device}' from {col_name}...")

    # Milvus Lite delete by filter
    try:
        result = client.delete(
            collection_name=col_name,
            filter=f'device_type == "{device}"',
        )
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logging.info(f"Deleted records for device '{device}' from {col_name} (result: {result})")
        return deleted_count
    except Exception as e:
        logging.error(f"Error deleting device '{device}' from {col_name}: {e}")
        return 0


def main():
    """
    # 全部导入 / Import all
    python vector.py
    
    # 指定 perspective / Specific perspectives
    python vector.py --perspectives as certificate
    
    # 指定设备 / Specific devices
    python vector.py --devices SCADA PRINTER
    
    # 重建 collection / Rebuild (drop existing)
    python vector.py --drop
    
    # 调整批量大小 / Adjust batch size
    python vector.py --batch_size 2000

    # Resume from where it stopped, fixing the partial stored data
    python vector.py --resume
    """
    
    parser = argparse.ArgumentParser(
        description="Save embedding CSVs to Milvus vector DB"
    )
    parser.add_argument(
        "--perspectives", nargs="*", default=None,
        help="Perspective names to process (default: all)"
    )
    parser.add_argument(
        "--devices", nargs="*", default=None,
        help="Device types to process (default: all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Batch size for insert (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop existing collections and rebuild"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip device if already has data in collection (default: True)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Auto-detect interrupted task from store_vector.log, delete partial data and re-insert, then continue"
    )
    args = parser.parse_args()

    log_filename = f"store_vector.log"
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # 确保 vectorDB 目录存在 / ensure vectorDB directory exists
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    logging.info(f"Milvus DB path: {MILVUS_DB_FILE}")

    # 连接 Milvus Lite (本地文件模式)
    client = MilvusClient(uri=MILVUS_DB_FILE)
    logging.info("Connected to Milvus Lite")

    # 从 rag_domain.json 获取设备名称 / Load device names from rag_domain.json
    all_devices = load_device_names()

    # 获取所有 perspective
    with open(os.path.join(BASE_PATH, "perspective_name.json"), "r") as f:
        all_perspectives = list(json.load(f).keys())
    
    perspectives = args.perspectives if args.perspectives else all_perspectives
    logging.info(f"Perspectives to process: {perspectives}")

    # ─── --resume: 解析日志, 找到中断的任务, 删除不完整数据并重新插入 ───
    # --resume: parse log, find interrupted task, delete partial data and re-insert
    resume_perspective = None
    resume_device = None
    if args.resume:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store_vector.log")
        interrupted = parse_log_for_resume(log_path)
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
            delete_device_from_collection(client, col_name, resume_device)
            logging.info(
                f"[Resume] Partial data for {resume_device} in {col_name} has been deleted. "
                f"Will re-insert from scratch."
            )
        else:
            logging.info("[Resume] No interrupted task found in log. All tasks completed.")

    total_rows = 0
    for perspective in perspectives:
        if perspective not in all_perspectives:
            logging.warning(f"Perspective '{perspective}' not found, skipping")
            continue

        csv_files = get_device_csv_files(LOCAL_PATH, perspective, all_devices)
        if not csv_files:
            logging.warning(f"No CSV files found for perspective '{perspective}'")
            continue

        # 按 device 过滤 / filter by device
        if args.devices:
            csv_files = [f for f in csv_files if f["device"] in args.devices]

        if not csv_files:
            logging.info(f"No matching device CSVs for perspective '{perspective}'")
            continue

        # 创建 collection
        col_name = create_collection(client, perspective, drop=args.drop, embedding_dim=EMBEDDING_DIM)

        # 逐文件插入 / insert each CSV
        for csv_info in csv_files:
            device = csv_info["device"]
            filepath = csv_info["filepath"]

            # --resume 模式: 被中断的 (perspective, device) 已删除旧数据, 强制重新插入
            # --resume mode: interrupted (perspective, device) had partial data deleted, force re-insert
            is_resume_target = (
                args.resume
                and resume_perspective == perspective
                and resume_device == device
            )

            # 跳过已存在的数据 / skip if already loaded
            if args.skip_existing and not args.drop and not is_resume_target:
                existing = check_existing_count(client, col_name, device)
                if existing > 0:
                    logging.info(
                        f"Skipping {device} in {col_name} (already has data)"
                    )
                    continue

            if is_resume_target:
                logging.info(f"[Resume] Re-inserting {device} into {col_name}")

            rows = insert_csv_to_collection(
                client, col_name, device, filepath, args.batch_size
            )
            total_rows += rows

    # ─── 处理 embedding_overall 文件 / Process embedding_overall files ───
    logging.info("Processing embedding_overall files...")
    overall_dir = os.path.join(LOCAL_PATH, "embedding_overall")
    
    if os.path.isdir(overall_dir):
        # 创建 overall collection
        overall_col_name = create_collection(client, "overall", drop=args.drop, col_name="overall", embedding_dim=EMBEDDING_OVERALL_DIM)
        
        # 遍历所有设备的 embedding_overall_raw.csv 文件
        for device in all_devices:
            fname = f"ipraw_{device}_embedding_overall_raw.csv"
            fpath = os.path.join(overall_dir, fname)
            if os.path.isfile(fpath):
                # 检查是否跳过已存在的数据
                if args.skip_existing and not args.drop:
                    existing = check_existing_count(client, overall_col_name, device)
                    if existing > 0:
                        logging.info(
                            f"Skipping {device} in {overall_col_name} (already has data)"
                        )
                        continue
                
                rows = insert_csv_to_collection(
                    client, overall_col_name, device, fpath, args.batch_size
                )
                total_rows += rows
            else:
                logging.warning(f"File not found: {fname}")
    else:
        logging.warning(f"Directory not found: {overall_dir}")

    logging.info(f"All done. Total rows inserted: {total_rows}")
    logging.info(f"Vector DB saved to: {VECTOR_DB_DIR}")
    client.close()


if __name__ == "__main__":
    main()
