"""
agent.py - IdentificationAgent: 集成问题分解和检索功能的设备识别Agent
IdentificationAgent: Device identification agent integrating query decomposition and retrieval

Usage:
    # 仅问题分解 / Decomposition only
    python agent/agent.py --decompose

    # 仅局部检索 / Local retrieval only
    python agent/agent.py --local

    # 局部 + 社区检索 / Local + community retrieval
    python agent/agent.py --local --community

    # 全部流程 / Full pipeline
    python agent/agent.py --decompose --local --community --reasoning

    # 指定设备和LLM / Specify devices and LLM
    python agent/agent.py --local --devices CAMERA PRINTER --llm deepseek --top_k 10
"""

import os
import sys
import json
import re
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import MultiLevelRetrieval
from util import *

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='a',
    filename="agent.log"
)


class IdentificationAgent:
    """
    设备识别Agent: 集成问题分解与多层次检索
    Device Identification Agent: integrates query decomposition and multi-level retrieval
    
    流程 / Pipeline:
        1. (可选) 问题分解 / (Optional) Query decomposition
        2. 从 evaluation/validation/test_{dev}_1.csv 加载待测IP与指纹
           Load test IPs and fingerprints from evaluation/validation/test_{dev}_1.csv
        3. 对每个IP执行检索 (local / community / reasoning)
           Run retrieval for each IP (local / community / reasoning)
        4. 按设备类型和检索类型分别保存结果到 agent/query_db/{local,community,reasoning}/
           Save results by device type and retrieval type
    """

    def __init__(self, llm: str = "CLAUDE", gpu: int = -1):
        """
        初始化IdentificationAgent
        Initialize IdentificationAgent
        
        Args:
            llm: LLM类型 / LLM type ("gemini", "deepseek", "openai")
        """
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.validation_path = os.path.join(self.base_path, "evaluation", "validation")
        self.llm_type = llm

        # 初始化MultiLevelRetrieval
        # Initialize MultiLevelRetrieval
        print(f"=== 初始化IdentificationAgent (LLM={llm}) ===")
        logging.info(f"初始化IdentificationAgent (LLM={llm})")
        self.retrieval_agent = MultiLevelRetrieval(llm=llm, gpu=gpu)
        print("=== IdentificationAgent初始化完成 ===\n")
        logging.info("IdentificationAgent初始化完成")

    def load_test_fingerprints(self, device_name: str) -> List[Dict[str, Any]]:
        """
        从 evaluation/validation/test_{device_name}_1.csv 加载所有IP及其query fingerprint
        Load all IPs and their query fingerprints from evaluation/validation/test_{device_name}_1.csv
        
        每一行的所有列(包括ip)组成一个 {column_name: value} 字典作为 query_fingerprint
        Each row's columns (including ip) form a {column_name: value} dict as query_fingerprint
        
        Args:
            device_name: 设备类型名称 / Device type name (e.g. "CAMERA")
        
        Returns:
            指纹列表, 每个元素为一个字典 / List of fingerprint dicts
        """
        csv_path = os.path.join(self.validation_path, f"test_{device_name}_1.csv")
        if not os.path.exists(csv_path):
            print(f"警告: 测试文件不存在 {csv_path}")
            return []

        print(f"加载测试数据: {csv_path}")
        logging.info(f"[TESTING] 开始加载测试数据: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        
        fingerprints = []
        for _, row in df.iterrows():
            fp = {}
            for col in df.columns:
                val = row[col]
                # 将NaN转为None方便JSON序列化
                # Convert NaN to None for JSON serialization
                if pd.isna(val):
                    fp[col] = None
                else:
                    fp[col] = val
            fingerprints.append(fp)

        print(f"加载了 {len(fingerprints)} 条测试指纹 (设备: {device_name})")
        logging.info(f"[TESTING] 加载了 {len(fingerprints)} 条测试指纹 (设备: {device_name})")

        return fingerprints

    def check_ip_already_retrieved(self, ip: str, device_name: str,
                                    whether_local: bool, whether_community: bool,
                                    whether_reasoning: bool) -> bool:
        """
        检查某个IP在分类保存的结果中是否已经完成了所需的检索
        Check if an IP has already completed required retrievals in categorized result files
        
        Args:
            ip: IP地址
            device_name: 设备类型名称
            whether_local / whether_community / whether_reasoning: 需要的检索类型
        
        Returns:
            True 如果所有需要的检索类型都已有结果
        """
        local_result, community_result, reasoning_result = \
            self.retrieval_agent.load_retrieval_result_by_type(ip, device_name)
        
        if whether_local and local_result is None:
            return False
        if whether_community and community_result is None:
            return False
        if whether_reasoning and reasoning_result is None:
            return False
        return True

    def run_vector_store(self, whether_resume: bool = False, whether_drop: bool = False, whether_skip: bool = False):
        self.retrieval_agent.vector_store_embedding(whether_resume=whether_resume, whether_drop=whether_drop, whether_skip=whether_skip)
    
    def _quick_load_done_ips(self, device_name: str,
                              whether_local: bool,
                              whether_community: bool,
                              whether_reasoning: bool) -> set:
        """
        一次性从 query_db/{local,community,reasoning}/{device_name}_*.json 加载已完成的IP集合
        Batch-load the set of IPs that have completed all required retrieval types
        
        Args:
            device_name: 设备类型名称 / Device type name
            whether_local / whether_community / whether_reasoning: 需要的检索类型
        
        Returns:
            已完成所有所需检索类型的IP集合 / Set of IPs that completed all required retrieval types
        """
        query_db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "query_db"
        )

        type_flags = []
        if whether_local:
            type_flags.append("local")
        if whether_community:
            type_flags.append("community")
        if whether_reasoning:
            type_flags.append("reasoning")

        if not type_flags:
            return set()

        # 为每种检索类型加载已完成IP
        # Load completed IPs for each retrieval type
        ip_sets = {}
        for rtype in type_flags:
            filepath = os.path.join(query_db_path, rtype, f"{device_name}_{rtype}.json")
            done_ips = set()
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    if isinstance(records, list):
                        for rec in records:
                            fp = rec.get("query_fingerprint", {})
                            if isinstance(fp, dict) and fp.get("ip"):
                                done_ips.add(fp["ip"])
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"quick_resume: JSON损坏 {filepath}: {e}, 视为空")
                    print(f"警告: quick_resume读取失败 {filepath}: {e}")
            ip_sets[rtype] = done_ips
            logging.info(f"quick_resume: {rtype}/{device_name} 已有 {len(done_ips)} 条记录")

        # 取交集: 所有所需类型都已完成的IP
        # Intersect: IPs that have completed ALL required types
        result = set.intersection(*ip_sets.values()) if ip_sets else set()
        print(f"quick_resume: 设备 {device_name} 已完成 {len(result)} 个IP (需要类型: {type_flags})")
        logging.info(f"quick_resume: 设备 {device_name} 已完成 {len(result)} 个IP")
        return result

    def run_retrieval(self, whether_decompose: bool = False,
            whether_local: bool = True,
            whether_community: bool = False,
            whether_reasoning: bool = False,
            devices: List[str] = None,
            top_k: int = 5,
            test_query: str = "Identify the device type and vendor.",
            quick_resume: bool = False):
        """
        主工作函数: 执行问题分解和/或检索流程
        Main workflow function: run decomposition and/or retrieval pipeline
        
        Args:
            whether_decompose: 是否执行问题分解 / Whether to run query decomposition
            whether_local: 是否执行局部检索 / Whether to run local retrieval
            whether_community: 是否执行社区检索 / Whether to run community retrieval
            whether_reasoning: 是否执行推理路径检索 / Whether to run reasoning path retrieval
            devices: 指定设备列表, None则自动扫描所有 / Device list, None for all available
            top_k: 局部检索返回数量 / Number of local retrieval results
            test_query: 用于问题分解的查询 / Query for decomposition
            quick_resume: 快速恢复模式, 一次性加载已完成IP集合避免逐条读取JSON
                          Quick resume mode, batch-load done IPs to avoid per-IP JSON reads
        """

        # ── Step 1: 问题分解 (可选) / Query decomposition (optional) ──
        problems = ["DEVICE_TYPE", "DEVICE_VENDOR"]
        if whether_decompose:
            try:
                from decomposition import main as decomposition_main
                print("=== 开始问题分解 ===")
                logging.info("开始问题分解")
                decomposition_result = decomposition_main(test_query)
                print("=== 问题分解完成 ===\n")
                logging.info("问题分解完成")
                problems = decomposition_result.get("identified_problems", problems)
                print(f"识别到的问题类型: {problems}")
                logging.info(f"识别到的问题类型: {problems}")
            except Exception as e:
                print(f"问题分解失败, 使用默认问题类型: {e}")
                logging.error(f"问题分解失败, 使用默认问题类型: {e}")
        else:
            print("跳过问题分解, 使用默认问题类型: DEVICE_TYPE, DEVICE_VENDOR\n")
            logging.info("跳过问题分解, 使用默认问题类型: DEVICE_TYPE, DEVICE_VENDOR")

        if "DEVICE_TYPE" not in problems:
            print(f"未检测到DEVICE_TYPE问题, 识别的问题类型: {problems}")
            logging.warning(f"未检测到DEVICE_TYPE问题, 识别的问题类型: {problems}")
            print("检索流程不执行")
            return

        # ── Step 2: 确定待处理设备列表 / Determine device list ──
        if devices is None:
            devices = load_all_dev_labels()
        
        if not devices:
            print("未找到任何待测试设备")
            logging.warning("未找到任何待测试设备")
            return
        
        print(f"待处理设备列表: {devices}\n")
        logging.info(f"待处理设备列表: {devices}")

        # 是否需要检索
        need_retrieval = whether_local or whether_community or whether_reasoning
        if not need_retrieval:
            print("未指定任何检索类型 (--local / --community / --reasoning), 仅完成问题分解")
            logging.info("未指定任何检索类型, 仅完成问题分解")
            return

        # ── Step 3: 逐设备逐IP检索 / Retrieve per device per IP ──
        for device_name in devices:
            print(f"\n{'='*60}")
            print(f"处理设备: {device_name}")
            print(f"{'='*60}")
            logging.info(f"开始处理设备: {device_name}")
            
            fingerprints = self.load_test_fingerprints(device_name)
            if not fingerprints:
                print(f"设备 {device_name} 无测试数据, 跳过")
                logging.warning(f"设备 {device_name} 无测试数据, 跳过")
                continue

            processed_count = 0
            skipped_count = 0

            # quick_resume: 一次性加载已完成IP集合
            # quick_resume: batch-load completed IP set
            if quick_resume:
                done_ips = self._quick_load_done_ips(
                    device_name, whether_local, whether_community, whether_reasoning
                )
            else:
                done_ips = None

            for i, query_fingerprint in enumerate(fingerprints):
                # if i>10:
                #     break
                ip = query_fingerprint.get("ip", f"unknown_{i}")
                # if ip != "212.50.39.34":
                #     continue
                print(f"\n--- [{device_name}] IP {i+1}/{len(fingerprints)}: {ip} ---")
                logging.info(f"[{device_name}] 处理IP {i+1}/{len(fingerprints)}: {ip}")

                # 检查是否已有结果 (跳过已完成的)
                # Check if already retrieved (skip completed ones)
                if quick_resume and done_ips is not None:
                    if ip in done_ips:
                        print(f"IP {ip} 已有完整检索结果 (quick_resume), 跳过")
                        logging.info(f"IP {ip} 已有完整检索结果 (quick_resume), 跳过")
                        skipped_count += 1
                        continue
                elif self.check_ip_already_retrieved(
                    ip, device_name, whether_local, whether_community, whether_reasoning
                ):
                    print(f"IP {ip} 已有完整检索结果, 跳过")
                    logging.info(f"IP {ip} 已有完整检索结果, 跳过")
                    skipped_count += 1
                    continue

                # 加载已有的部分结果用于级联检索
                # Load existing partial results for cascading retrieval
                existing_local, existing_community, _ = \
                    self.retrieval_agent.load_retrieval_result_by_type(ip, device_name)

                # 调用 run_retrieval_algorithm 执行检索
                # Call run_retrieval_algorithm to perform retrieval
                local_result, community_result, reasoning_result = \
                    self.retrieval_agent.run_retrieval_algorithm(
                        test_fingerprint=query_fingerprint,
                        top_k=top_k,
                        whether_local=(whether_local and existing_local is None),
                        whether_community=(whether_community and existing_community is None),
                        whether_reasoning=whether_reasoning,
                        local_result=existing_local,
                        community_result=existing_community,
                        llm_type=self.llm_type,
                        device_name=device_name
                    )
                
                # 清空内存中的历史记录, 避免累积
                # Clear in-memory history to avoid accumulation
                self.retrieval_agent.clear_history()
                
                processed_count += 1
                print(f"结果已保存 (设备: {device_name}, IP: {ip})")
                logging.info(f"结果已保存 (设备: {device_name}, IP: {ip})")

            print(f"\n设备 {device_name} 处理完成: 新处理 {processed_count}, 跳过 {skipped_count}, 总计 {len(fingerprints)}")
            logging.info(f"设备 {device_name} 处理完成: 新处理 {processed_count}, 跳过 {skipped_count}, 总计 {len(fingerprints)}")

        print(f"\n{'='*60}")
        print("=== 所有设备检索完成 ===")
        print(f"{'='*60}")
        logging.info("所有设备检索完成")


def main():
    # python agent.py --local --community --reasoning --device ROUTER --llm DEEPSEEK
    # python agent.py --local --community --reasoning --device NVR --llm DEEPSEEK
    # python agent.py --local --community --reasoning --device POWER_METER --llm DEEPSEEK
    parser = argparse.ArgumentParser(
        description="IdentificationAgent: 设备识别Agent, 集成问题分解与多层次检索"
    )

    parser.add_argument(
        "--decompose", action="store_true", default=False,
        help="是否执行问题分解 / Whether to run query decomposition"
    )
    
    parser.add_argument(
        "--vector", action="store_true", default=False,
        help="是否执行向量存储 / Whether to run vector store"
    )

    parser.add_argument(
        "--vector_resume", action="store_true", default=False,
        help="是否执行向量存储恢复 / Whether to run vector store resume"
    )

    parser.add_argument(
        "--vector_drop", action="store_true", default=False,
        help="是否执行向量存储恢复 / Whether to drop existing stored vectors"
    )

    parser.add_argument(
        "--vector_skip", action="store_true", default=True,
        help="是否跳过执行已存储的向量 / Whether to skip existing stored vectors"
    )

    parser.add_argument(
        "--local", action="store_true", default=False,
        help="是否执行局部检索 / Whether to run local (entity) retrieval"
    )
    parser.add_argument(
        "--community", action="store_true", default=False,
        help="是否执行社区检索 / Whether to run community retrieval"
    )
    parser.add_argument(
        "--reasoning", action="store_true", default=False,
        help="是否执行推理路径检索 / Whether to run reasoning path retrieval"
    )
    parser.add_argument('--device', type=str, nargs='+', default=None,
                        help='指定设备类型 (如 CAMERA NAS)，默认处理全部')
    parser.add_argument(
        "--llm", type=str, default="CLAUDE", choices=["GEMINI", "DEEPSEEK", "OPENAI"],
        help="LLM类型 / LLM type (default: DEEPSEEK)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="局部检索返回数量 / Number of local retrieval results (default: 5)"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="是否使用GPU? 如果是，指定GPU号 (default: -1)"
    )

    parser.add_argument(
        "--query", type=str, default="Identify the device type and vendor.",
        help="问题分解查询文本 / Query text for decomposition"
    )
    parser.add_argument(
        "--quick_resume", action="store_true", default=False,
        help="快速恢复: 一次性加载已完成IP集合, 跳过已检索IP / Quick resume: batch-load completed IPs and skip them"
    )
    args = parser.parse_args()

    # 如果没有指定任何操作, 打印帮助
    # If no action specified, print help
    if not (args.vector or args.decompose or args.local or args.community or args.reasoning):
        parser.print_help()
        print("\n请至少指定一个操作: --vector --decompose, --local, --community, --reasoning")
        return
    
    if args.device:
        device_types = []
        for dev in args.device:
            device_types.extend([d.strip() for d in dev.split(',') if d.strip()])
    else:
        device_types = load_all_dev_labels()

    agent = IdentificationAgent(llm=args.llm, gpu=args.gpu)

    # python agent.py --vector
    if args.vector:
        log_filename = f"store_vector.log"
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        agent.run_vector_store(whether_resume=args.vector_resume, whether_drop=args.vector_drop, whether_skip=args.vector_skip)

    agent.run_retrieval(
        whether_decompose=args.decompose,
        whether_local=args.local,
        whether_community=args.community,
        whether_reasoning=args.reasoning,
        devices=device_types,
        top_k=args.top_k,
        test_query=args.query,
        quick_resume=args.quick_resume,
    )


if __name__ == "__main__":
    main()
