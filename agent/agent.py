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

    # 仅决策 (需要先完成检索) / Decision only (retrieval must be done first)
    python agent/agent.py --decision

    # 全部流程 / Full pipeline (decomposition + retrieval + decision)
    python agent/agent.py --decompose --local --community --reasoning --decision

    # 指定设备和LLM / Specify devices and LLM
    python agent/agent.py --local --decision --device CAMERA PRINTER --llm deepseek --top_k 10
"""

import os
import sys
import json
import re
import time
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
        5. (可选) 决策分类: 基于检索结果, Gemini+Claude联合投票判定设备类型与厂商
           (Optional) Decision: Gemini+Claude joint voting for device type & vendor
           Results saved to evaluation/predict/result/
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

    # ── Decision 相关方法 ──────────────────────────────────────────────────

    def _quick_load_decision_done_ips(self, device_name: str) -> set:
        """
        一次性从 evaluation/predict/{device_name}_type_prediction.json 加载已有决策结果的IP集合
        Batch-load IPs that already have decision results from the prediction JSON file
        """
        result_path = os.path.join(self.base_path, "evaluation", "predict")
        filepath = os.path.join(result_path, f"{device_name}_type_prediction.json")
        done_ips = set()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    ip = rec.get("ip")
                    if ip:
                        done_ips.add(str(ip))
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"decision quick_resume: JSON损坏 {filepath}: {e}")
                print(f"警告: decision quick_resume读取失败 {filepath}: {e}")
        print(f"decision quick_resume: 设备 {device_name} 已有 {len(done_ips)} 条决策记录")
        logging.info(f"decision quick_resume: {device_name} 已有 {len(done_ips)} 条决策记录")
        return done_ips

    def _save_decision_merge(self, device_name: str, new_results: List[Dict]):
        """
        将新决策结果与已有结果合并保存到三个位置:
        1. evaluation/predict/{device_name}_{suffix}.json — 完整JSON结果
        2. evaluation/validation/type/predict/IoTProber/type_{device_name}.csv — 设备类型预测CSV
        3. evaluation/validation/vendor/predict/IoTProber/vendor_{device_name}.csv — 设备厂商预测CSV

        Merge new decision results with existing ones and save to three locations.
        """
        predict_path = os.path.join(self.base_path, "evaluation", "predict")
        type_csv_path = os.path.join(self.base_path, "evaluation", "validation", "type", "predict", "IoTProber")
        vendor_csv_path = os.path.join(self.base_path, "evaluation", "validation", "vendor", "predict", "IoTProber")
        os.makedirs(predict_path, exist_ok=True)
        os.makedirs(type_csv_path, exist_ok=True)
        os.makedirs(vendor_csv_path, exist_ok=True)

        new_type_records = [
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
                "first_stage":           r.get("first_stage"),
                "elapsed_sec":           r.get("elapsed_sec"),
            }
            for r in new_results
        ]

        new_vendor_records = [
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
            for r in new_results
        ]

        # Save full JSON results to evaluation/predict/
        for suffix, new_records in [
            ("type_prediction", new_type_records),
            ("vendor_prediction", new_vendor_records),
        ]:
            filepath = os.path.join(predict_path, f"{device_name}_{suffix}.json")
            existing = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    existing = []

            existing_ips = {str(r.get("ip")) for r in existing}
            for rec in new_records:
                if str(rec.get("ip")) not in existing_ips:
                    existing.append(rec)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            logging.info(f"[Decision] 保存 {len(existing)} 条记录 → {filepath}")
            print(f"  ✓  保存 → {filepath}")

        # Save type prediction CSV to evaluation/validation/type/predict/IoTProber/
        type_csv_file = os.path.join(type_csv_path, f"type_{device_name}.csv")
        type_df = pd.DataFrame(new_type_records)
        if os.path.exists(type_csv_file):
            existing_type_df = pd.read_csv(type_csv_file)
            existing_type_ips = set(existing_type_df["ip"].astype(str))
            new_type_rows = type_df[~type_df["ip"].astype(str).isin(existing_type_ips)]
            combined_type_df = pd.concat([existing_type_df, new_type_rows], ignore_index=True)
        else:
            combined_type_df = type_df
        combined_type_df.to_csv(type_csv_file, index=False)
        logging.info(f"[Decision] 保存 {len(combined_type_df)} 条类型预测 → {type_csv_file}")
        print(f"  ✓  保存 → {type_csv_file}")

        # Save vendor prediction CSV to evaluation/validation/vendor/predict/IoTProber/
        vendor_csv_file = os.path.join(vendor_csv_path, f"vendor_{device_name}.csv")
        vendor_df = pd.DataFrame(new_vendor_records)
        if os.path.exists(vendor_csv_file):
            existing_vendor_df = pd.read_csv(vendor_csv_file)
            existing_vendor_ips = set(existing_vendor_df["ip"].astype(str))
            new_vendor_rows = vendor_df[~vendor_df["ip"].astype(str).isin(existing_vendor_ips)]
            combined_vendor_df = pd.concat([existing_vendor_df, new_vendor_rows], ignore_index=True)
        else:
            combined_vendor_df = vendor_df
        combined_vendor_df.to_csv(vendor_csv_file, index=False)
        logging.info(f"[Decision] 保存 {len(combined_vendor_df)} 条厂商预测 → {vendor_csv_file}")
        print(f"  ✓  保存 → {vendor_csv_file}")

    def run_decision(self, devices: List[str] = None, quick_resume: bool = False,
                     whether_local: bool = True,
                     whether_community: bool = True,
                     whether_reasoning: bool = True,
                     top_k: int = 5,
                     enable_first_stage: bool = True,
                     unseen_adapter_path: str = None,
                     unseen_load_in_4bit: bool = False,
                     drift_model_dir: str = None,
                     gpu: int = -1):
        """
        决策流程 (LangGraph): unseen(LLaMA) + Tavily ReAct → drift 条件 → Gemini/Claude
        并行 decision ReAct 分支 → joint voting finish.
        Decision pipeline implemented as a LangGraph workflow that replaces the
        legacy create_openai_tools_agent ReAct implementation.

        Args:
            devices: 指定设备列表, None则自动扫描所有 / Device list, None for all available
            quick_resume: 快速恢复模式, 跳过已有决策结果的IP
                          Quick resume mode, skip IPs with existing decision results
            whether_local/whether_community/whether_reasoning: 检索工具运行哪些检索算法
                          Which retrieval algorithms the retrieval tool should run
            top_k: 局部检索返回数量 / Number of local retrieval results
            enable_first_stage: 是否启用 unseen + drift 第一阶段
                          Whether to enable the unseen + drift first stage
            unseen_adapter_path / unseen_load_in_4bit / drift_model_dir / gpu:
                          unseen/drift 检测器配置 / detector configuration
        """
        print(f"\n{'='*60}")
        print("=== 开始决策流程 (Decision / LangGraph) ===")
        print(f"{'='*60}")
        logging.info("开始决策流程 (Decision / LangGraph)")

        decision_graph = IoTDecisionGraph(
            retrieval_agent=self.retrieval_agent,
            whether_local=whether_local,
            whether_community=whether_community,
            whether_reasoning=whether_reasoning,
            top_k=top_k,
            llm_type=self.llm_type,
            enable_first_stage=enable_first_stage,
            unseen_adapter_path=unseen_adapter_path,
            unseen_load_in_4bit=unseen_load_in_4bit,
            drift_model_dir=drift_model_dir,
            gpu=gpu,
        )

        if devices is None:
            devices = load_all_dev_labels()

        if not devices:
            print("未找到任何待处理设备")
            logging.warning("[Decision] 未找到任何待处理设备")
            return

        print(f"待处理设备列表: {devices}\n")
        logging.info(f"[Decision] 待处理设备列表: {devices}")

        for device_name in devices:
            print(f"\n{'='*60}")
            print(f"[Decision] 处理设备: {device_name}")
            print(f"{'='*60}")
            logging.info(f"[Decision] 开始处理设备: {device_name}")

            fingerprints = self.load_test_fingerprints(device_name)
            if not fingerprints:
                print(f"设备 {device_name} 无测试数据, 跳过")
                logging.warning(f"[Decision] 设备 {device_name} 无测试数据, 跳过")
                continue

            # quick_resume: 加载已有决策结果的IP集合
            # quick_resume: batch-load IPs with existing decision results
            if quick_resume:
                done_ips = self._quick_load_decision_done_ips(device_name)
            else:
                done_ips = set()

            processed_count = 0
            skipped_count = 0
            new_results = []

            for i, fp in enumerate(fingerprints):
                ip = str(fp.get("ip", f"unknown_{i}"))
                print(f"\n--- [Decision][{device_name}] IP {i+1}/{len(fingerprints)}: {ip} ---")
                logging.info(f"[Decision][{device_name}] 处理IP {i+1}/{len(fingerprints)}: {ip}")

                if quick_resume and ip in done_ips:
                    print(f"IP {ip} 已有决策结果 (quick_resume), 跳过")
                    logging.info(f"[Decision] IP {ip} 已有决策结果, 跳过")
                    skipped_count += 1
                    continue

                result = decision_graph.classify(ip, device_name, fp)
                new_results.append(result)
                processed_count += 1

                print(f"[Decision] 结果: {result.get('predicted_device_type', 'N/A')} "
                      f"(conf={result.get('final_confidence', 0):.3f}, "
                      f"winner={result.get('winning_llm', 'N/A')})")
                logging.info(f"[Decision] IP={ip} → {result.get('predicted_device_type')} "
                            f"conf={result.get('final_confidence')}")

            # 保存结果 (与已有结果合并)
            # Save results (merge with existing)
            if new_results:
                self._save_decision_merge(device_name, new_results)

            print(f"\n[Decision] 设备 {device_name} 处理完成: 新处理 {processed_count}, "
                  f"跳过 {skipped_count}, 总计 {len(fingerprints)}")
            logging.info(f"[Decision] 设备 {device_name} 完成: 新处理 {processed_count}, "
                        f"跳过 {skipped_count}, 总计 {len(fingerprints)}")

        print(f"\n{'='*60}")
        print("=== 所有设备决策完成 ===")
        print(f"{'='*60}")
        logging.info("所有设备决策完成")

        # ── 图更新: 根据识别结果更新层次图 / Graph update based on decision results ──
        if new_results:
            self._run_graph_update(devices)

    def _run_graph_update(self, devices: List[str]):
        """
        根据决策识别结果, 调用图增量更新算法更新层次图.
        Trigger graph incremental update based on decision results.

        读取 evaluation/predict/ 下的决策结果, 对每个识别出的设备:
        1. 从测试CSV加载指纹特征
        2. 使用retrieval_agent._embed_fingerprint计算各perspective的embedding
        3. 调用GraphUpdater.update_devices进行图更新
        """
        try:
            sys.path.insert(0, os.path.join(self.base_path, "graph"))
            from update_graph import GraphUpdater
        except ImportError as e:
            logging.warning(f"[GraphUpdate] 无法导入GraphUpdater, 跳过图更新: {e}")
            print(f"[GraphUpdate] 无法导入GraphUpdater, 跳过图更新: {e}")
            return

        logging.info("[GraphUpdate] 开始图增量更新")
        print(f"\n{'='*60}")
        print("=== 开始图增量更新 (Graph Update) ===")
        print(f"{'='*60}")

        updater = GraphUpdater(llm=self.retrieval_agent.llm)

        predict_path = os.path.join(self.base_path, "evaluation", "predict")
        update_devices = []

        for device_name in devices:
            type_json = os.path.join(predict_path, f"{device_name}_type_prediction.json")
            if not os.path.exists(type_json):
                continue

            with open(type_json, "r", encoding="utf-8") as f:
                type_results = json.load(f)

            if not type_results:
                continue

            # 加载测试指纹用于提取特征和计算embedding
            fingerprints = self.load_test_fingerprints(device_name)
            fp_map = {str(fp.get("ip")): fp for fp in fingerprints}

            for rec in type_results:
                ip = str(rec.get("ip", ""))
                if not ip or ip not in fp_map:
                    continue

                fp = fp_map[ip]
                predicted_type = rec.get("predicted_device_type", device_name)

                # 使用retrieval_agent计算各perspective的embedding
                try:
                    embeddings_2d = self.retrieval_agent._embed_fingerprint(fp)
                except Exception as e:
                    logging.warning(f"[GraphUpdate] IP {ip} embedding失败, 跳过: {e}")
                    continue

                # 将1024维embedding转为各perspective的PCA embedding (256维)
                # embeddings_2d shape: (n_perspectives, 1024)
                perspective_names = self.retrieval_agent.retrieval_perspective_names
                embeddings = {}
                for idx, p_name in enumerate(perspective_names):
                    if idx < len(embeddings_2d):
                        emb_1024 = embeddings_2d[idx]
                        # PCA降维: 使用retrieval_agent的pca (需要先fit)
                        # 这里简化处理: 直接截取前256维作为近似PCA
                        # 实际应用中应使用已fit的PCA模型
                        embeddings[p_name] = emb_1024[:256].tolist()

                # 构建comprehensive embedding (所有perspective拼接)
                comprehensive_emb = []
                for idx in range(len(embeddings_2d)):
                    comprehensive_emb.extend(embeddings_2d[idx][:256])
                embeddings["comprehensive"] = comprehensive_emb

                # 提取Layer-1特征 (local_used_feature.txt中定义的特征)
                features = {}
                for feat_name in self.retrieval_agent.perspective_info_config.get("overall", {}).get("cols", []):
                    if feat_name in fp and fp[feat_name] is not None:
                        features[feat_name] = str(fp[feat_name])

                update_devices.append({
                    "ip": ip,
                    "device_type": predicted_type,
                    "features": features,
                    "embeddings": embeddings,
                })

        if update_devices:
            logging.info(f"[GraphUpdate] 更新 {len(update_devices)} 个设备到图中")
            print(f"[GraphUpdate] 更新 {len(update_devices)} 个设备到图中")
            updater.update_devices(update_devices)
        else:
            logging.info("[GraphUpdate] 无需更新的设备")
            print("[GraphUpdate] 无需更新的设备")

        print(f"\n{'='*60}")
        print("=== 图增量更新完成 ===")
        print(f"{'='*60}")
        logging.info("图增量更新完成")


# ═════════════════════════════════════════════════════════════════════════════
# LangGraph Decision Graph
# ═════════════════════════════════════════════════════════════════════════════
#
# 图分支 / Graph branches:
#   START → unseen ──(needs web search)──▶ unseen_tools(Tavily) ──▶ unseen
#                 └──(sufficient)──▶ gate ──(both probs < 0.5)──▶ drift ──▶ prepare
#                                          └──(otherwise)──────────────▶ prepare
#   prepare ─┬─▶ gemini_agent ⇄ gemini_tools(retrieval) ─▶ gemini_finalize ─┐
#            └─▶ claude_agent ⇄ claude_tools(retrieval) ─▶ claude_finalize ─┤
#                                                                            ▼
#                                                        finish (joint voting) → END
#
# - unseen 节点使用训练好的 LLaMA-3.1-8B (unseen.py detect_unseen), 非 LoRA 训练逻辑.
# - unseen 的 tool node 是 Tavily web search, 当信息不足/不确定时补充字段查询 (ReAct 循环).
# - condition 节点依据 unseen 概率决定是否进入 drift 节点 (两概率均 < 0.5 → drift).
# - 两个 decision LLM 节点并行, 每个节点后接一个 retrieval tool node 构成 ReAct 循环,
#   首个动作必须调用整合后的 multi-level retrieval 工具.
# - finish 节点按 decision.py 的 joint voting 策略, 依据置信度选出最终结果.

class IoTDecisionGraph:
    """
    使用 LangGraph 实现的设备识别决策图 (替换原 create_openai_tools_agent ReAct 实现).
    LangGraph implementation of the device-identification decision workflow.
    """

    _MAX_WEB_ITERS = 2       # unseen Tavily ReAct 最大轮数
    _MAX_TOOL_CALLS = 3      # 每个 decision 分支的最大工具调用次数

    def __init__(
        self,
        retrieval_agent: Optional[MultiLevelRetrieval] = None,
        whether_local: bool = True,
        whether_community: bool = True,
        whether_reasoning: bool = True,
        top_k: int = 5,
        llm_type: str = "CLAUDE",
        enable_first_stage: bool = True,
        unseen_model_path: Optional[str] = None,
        unseen_adapter_path: Optional[str] = None,
        unseen_load_in_4bit: bool = False,
        drift_model_dir: Optional[str] = None,
        gpu: int = -1,
        recursion_limit: int = 60,
    ):
        # ── lazy heavy imports (only when the decision graph is actually built) ──
        from typing import Annotated, TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage, ToolMessage,
        )
        from tools.tavily_search import TAVILY_TOOLS

        from decision import (
            RetrievalToolRuntime,
            set_retrieval_runtime,
            configurable_multi_level_retrieval,
            extract_decision_json,
            normalize_decision,
            joint_vote,
            _AGENT_SYSTEM,
            _AGENT_HUMAN,
        )

        self._AIMessage = AIMessage
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage
        self._ToolMessage = ToolMessage
        self._extract_decision_json = extract_decision_json
        self._normalize_decision = normalize_decision
        self._joint_vote = joint_vote
        self._agent_system = _AGENT_SYSTEM
        self._agent_human = _AGENT_HUMAN

        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.enable_first_stage = enable_first_stage
        self.recursion_limit = recursion_limit

        # first-stage detector configuration (lazy, fail-safe)
        self._unseen_model_path = unseen_model_path
        self._unseen_adapter_path = unseen_adapter_path
        self._unseen_load_in_4bit = unseen_load_in_4bit
        self._drift_model_dir = drift_model_dir
        self._gpu = gpu
        self._unseen_detector = None
        self._unseen_detector_ready = False
        self._drift_detector = None
        self._drift_detector_ready = False

        # ── configurable retrieval tool runtime (shared by both decision branches) ──
        self.runtime = RetrievalToolRuntime(
            whether_local=whether_local,
            whether_community=whether_community,
            whether_reasoning=whether_reasoning,
            top_k=top_k,
            llm_type=llm_type,
            retrieval_agent=retrieval_agent,
        )
        set_retrieval_runtime(self.runtime)
        self._retrieval_tool = configurable_multi_level_retrieval

        # ── LLM clients (OpenAI-compatible endpoints) bound with the retrieval tool ──
        cfg_path = os.path.join(self.base_path, "llm_config.json")
        with open(cfg_path, "r") as fh:
            cfg = json.load(fh)
        self._gemini_llm = ChatOpenAI(
            api_key=cfg["GEMINI"]["API_KEY"],
            base_url=cfg["GEMINI"]["BASE_URL"],
            model=cfg["GEMINI"]["MODEL"],
            temperature=0.3,
            max_tokens=4096,
        ).bind_tools([self._retrieval_tool])
        self._claude_llm = ChatOpenAI(
            api_key=cfg["CLAUDE"]["API_KEY"],
            base_url=cfg["CLAUDE"]["BASE_URL"],
            model=cfg["CLAUDE"]["MODEL"],
            temperature=0.3,
            max_tokens=4096,
        ).bind_tools([self._retrieval_tool])

        # ── build state schema + graph ──
        GraphState = TypedDict(
            "GraphState",
            {
                "ip": str,
                "device_name": str,
                "fingerprint": dict,
                "reasoning_result": Optional[dict],
                "local_result": Optional[dict],
                "community_result": Optional[dict],
                "web_search_results": str,
                "unseen_iter": int,
                "unseen_messages": Annotated[list, add_messages],
                "unseen_result": dict,
                "first_stage": dict,
                "drift_result": Optional[dict],
                "gemini_messages": Annotated[list, add_messages],
                "claude_messages": Annotated[list, add_messages],
                "gemini_result": dict,
                "claude_result": dict,
                "result": dict,
            },
        )

        self._tavily_tools = TAVILY_TOOLS

        g = StateGraph(GraphState)
        g.add_node("unseen", self._unseen_node)
        g.add_node(
            "unseen_tools",
            ToolNode(self._tavily_tools, messages_key="unseen_messages"),
        )
        g.add_node("gate", self._gate_node)
        g.add_node("drift", self._drift_node)
        g.add_node("prepare", self._prepare_decision_node)
        g.add_node("gemini_agent", self._gemini_agent_node)
        g.add_node(
            "gemini_tools",
            ToolNode([self._retrieval_tool], messages_key="gemini_messages"),
        )
        g.add_node("gemini_finalize", self._gemini_finalize_node)
        g.add_node("claude_agent", self._claude_agent_node)
        g.add_node(
            "claude_tools",
            ToolNode([self._retrieval_tool], messages_key="claude_messages"),
        )
        g.add_node("claude_finalize", self._claude_finalize_node)
        g.add_node("finish", self._finish_node, defer=True)

        g.add_edge(START, "unseen")
        g.add_conditional_edges(
            "unseen", self._route_after_unseen,
            {"search": "unseen_tools", "gate": "gate"},
        )
        g.add_edge("unseen_tools", "unseen")
        g.add_conditional_edges(
            "gate", self._route_after_gate,
            {"drift": "drift", "decision": "prepare"},
        )
        g.add_edge("drift", "prepare")
        g.add_edge("prepare", "gemini_agent")
        g.add_edge("prepare", "claude_agent")
        g.add_conditional_edges(
            "gemini_agent", self._route_gemini,
            {"tools": "gemini_tools", "finalize": "gemini_finalize"},
        )
        g.add_edge("gemini_tools", "gemini_agent")
        g.add_edge("gemini_finalize", "finish")
        g.add_conditional_edges(
            "claude_agent", self._route_claude,
            {"tools": "claude_tools", "finalize": "claude_finalize"},
        )
        g.add_edge("claude_tools", "claude_agent")
        g.add_edge("claude_finalize", "finish")
        g.add_edge("finish", END)

        self.graph = g.compile()
        logging.info(
            "IoTDecisionGraph built (first_stage=%s, levels=%s).",
            enable_first_stage,
            self.runtime_levels(),
        )

    # ── detector loaders (lazy, fail-safe) ────────────────────────────────────

    def runtime_levels(self) -> List[str]:
        levels = []
        if self.runtime.whether_local:
            levels.append("local")
        if self.runtime.whether_community:
            levels.append("community")
        if self.runtime.whether_reasoning:
            levels.append("reasoning")
        return levels

    def _get_unseen_detector(self):
        if self._unseen_detector_ready:
            return self._unseen_detector
        self._unseen_detector_ready = True
        if not self.enable_first_stage:
            self._unseen_detector = None
            return None
        try:
            from unseen import UnseenDeviceDetector
            self._unseen_detector = UnseenDeviceDetector(
                model_path=self._unseen_model_path,
                adapter_path=self._unseen_adapter_path,
                gpu=self._gpu,
                load_in_4bit=self._unseen_load_in_4bit,
            )
        except Exception as exc:
            logging.warning("Unseen detector unavailable, skipping: %s", exc)
            self._unseen_detector = None
        return self._unseen_detector

    def _get_drift_detector(self):
        if self._drift_detector_ready:
            return self._drift_detector
        self._drift_detector_ready = True
        try:
            from drift import DriftDetector, DRIFT_OUTPUT_DIR
            model_dir = self._drift_model_dir or DRIFT_OUTPUT_DIR
            self._drift_detector = DriftDetector(model_dir=model_dir)
        except Exception as exc:
            logging.warning("Drift detector unavailable, skipping: %s", exc)
            self._drift_detector = None
        return self._drift_detector

    # ── unseen node + Tavily ReAct loop ───────────────────────────────────────

    def _unseen_node(self, state: dict) -> dict:
        ip = state["ip"]
        device_name = state["device_name"]
        fingerprint = state["fingerprint"]

        # ensure retrieval evidence exists (runs enabled levels if missing)
        reasoning_result = state.get("reasoning_result")
        local_result = state.get("local_result")
        community_result = state.get("community_result")
        if reasoning_result is None and self.runtime.retrieval_agent is not None:
            try:
                self.runtime.register(ip, device_name, fingerprint)
                self.runtime.ensure(ip)
            except Exception as exc:
                logging.warning("Retrieval bootstrap for unseen failed (%s): %s", ip, exc)
        if reasoning_result is None and self.runtime.retrieval_agent is not None:
            try:
                local_result, community_result, reasoning_result = (
                    self.runtime.retrieval_agent.load_retrieval_result_by_type(ip, device_name)
                )
            except Exception as exc:
                logging.warning("Loading retrieval results for unseen failed (%s): %s", ip, exc)

        unavailable = {
            "new_type_probability": 0.0,
            "new_vendor_probability": 0.0,
            "is_unseen": False,
            "predicted_type": "none",
            "predicted_vendor": "none",
            "confidence": 0.0,
            "needs_web_search": False,
            "search_queries": [],
            "available": False,
        }

        detector = self._get_unseen_detector()
        if detector is None or reasoning_result is None:
            return {
                "unseen_result": unavailable,
                "reasoning_result": reasoning_result,
                "local_result": local_result,
                "community_result": community_result,
            }

        # Extract web search results from ToolMessages (returning from unseen_tools)
        unseen_messages = state.get("unseen_messages", [])
        tool_results = []
        for msg in unseen_messages:
            if isinstance(msg, self._ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    tool_results.append(content)
        web_results = (
            "\n\n".join(tool_results) if tool_results
            else (state.get("web_search_results") or None)
        )
        try:
            res = detector.detect_unseen(
                reasoning_result=reasoning_result,
                local_result=local_result,
                community_result=community_result,
                web_search_results=web_results,
                allow_web_search=True,
            )
            res["available"] = True
        except Exception as exc:
            logging.warning("Unseen detection failed for %s: %s", ip, exc)
            res = unavailable

        update = {
            "unseen_result": res,
            "reasoning_result": reasoning_result,
            "local_result": local_result,
            "community_result": community_result,
        }

        # If web search is needed, emit AIMessage with tool_calls for the ToolNode
        current_iter = state.get("unseen_iter", 0)
        if (
            res.get("needs_web_search")
            and res.get("search_queries")
            and current_iter < self._MAX_WEB_ITERS
        ):
            queries = res["search_queries"][:3]
            tool_calls = [
                {
                    "name": "tavily_web_search",
                    "args": {"query": q},
                    "id": f"unseen_search_{current_iter}_{i}",
                }
                for i, q in enumerate(queries)
            ]
            update["unseen_messages"] = [
                self._AIMessage(content="", tool_calls=tool_calls)
            ]
            update["unseen_iter"] = current_iter + 1

        return update

    def _route_after_unseen(self, state: dict) -> str:
        res = state.get("unseen_result", {})
        if (
            res.get("needs_web_search")
            and res.get("search_queries")
            and state.get("unseen_iter", 0) < self._MAX_WEB_ITERS
        ):
            return "search"
        return "gate"

    # ── condition (gate) + drift node ─────────────────────────────────────────

    def _gate_node(self, state: dict) -> dict:
        res = state.get("unseen_result", {})
        new_type = float(res.get("new_type_probability", 0.0))
        new_vendor = float(res.get("new_vendor_probability", 0.0))
        run_drift = self.enable_first_stage and (new_type < 0.5) and (new_vendor < 0.5)
        first_stage = {
            "unseen": {
                "new_type_probability": new_type,
                "new_vendor_probability": new_vendor,
                "is_unseen": bool(res.get("is_unseen", False)),
                "predicted_type": res.get("predicted_type", "none"),
                "predicted_vendor": res.get("predicted_vendor", "none"),
                "confidence": res.get("confidence", 0.0),
                "web_search_used": res.get("web_search_used", False),
                "available": res.get("available", False),
            },
            "drift_checked": run_drift,
        }
        return {"first_stage": first_stage}

    def _route_after_gate(self, state: dict) -> str:
        return "drift" if state.get("first_stage", {}).get("drift_checked") else "decision"

    def _drift_node(self, state: dict) -> dict:
        first_stage = dict(state.get("first_stage", {}))
        drift_det = self._get_drift_detector()
        drift_res: Dict[str, Any]
        if drift_det is None:
            drift_res = {"available": False}
        else:
            try:
                drift_res = drift_det.detect_query_device(state["fingerprint"])
            except Exception as exc:
                logging.warning("Drift detection failed for %s: %s", state.get("ip"), exc)
                drift_res = {"error": str(exc)}
        first_stage["drift"] = drift_res
        return {"first_stage": first_stage, "drift_result": drift_res}

    # ── decision preparation + parallel LLM ReAct branches ────────────────────

    @staticmethod
    def _fmt_fp(fp: dict) -> str:
        display = {
            k: v
            for k, v in fp.items()
            if k != "ip" and v is not None and str(v) not in ("nan", "None", "")
        }
        return json.dumps(display, indent=2, ensure_ascii=False)

    def _prepare_decision_node(self, state: dict) -> dict:
        ip = state["ip"]
        # register per-IP context so the retrieval tool knows device + fingerprint
        self.runtime.register(ip, state["device_name"], state["fingerprint"])
        human = self._agent_human.format(ip=ip, fingerprint=self._fmt_fp(state["fingerprint"]))
        base = [self._SystemMessage(self._agent_system), self._HumanMessage(human)]
        return {"gemini_messages": list(base), "claude_messages": list(base)}

    def _gemini_agent_node(self, state: dict) -> dict:
        resp = self._gemini_llm.invoke(state["gemini_messages"])
        return {"gemini_messages": [resp]}

    def _claude_agent_node(self, state: dict) -> dict:
        resp = self._claude_llm.invoke(state["claude_messages"])
        return {"claude_messages": [resp]}

    def _route_branch(self, messages: list) -> str:
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None)
        n_tool_msgs = sum(1 for m in messages if isinstance(m, self._ToolMessage))
        if tool_calls and n_tool_msgs < self._MAX_TOOL_CALLS:
            return "tools"
        return "finalize"

    def _route_gemini(self, state: dict) -> str:
        return self._route_branch(state.get("gemini_messages", []))

    def _route_claude(self, state: dict) -> str:
        return self._route_branch(state.get("claude_messages", []))

    def _last_ai_text(self, messages: list) -> str:
        for msg in reversed(messages):
            if isinstance(msg, self._AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if content and str(content).strip():
                    return str(content)
        return ""

    def _finalize_branch(self, messages: list, llm_name: str) -> dict:
        raw = self._last_ai_text(messages)
        try:
            parsed = self._extract_decision_json(raw)
        except Exception:
            parsed = {}
        return self._normalize_decision(parsed, llm_name, raw)

    def _gemini_finalize_node(self, state: dict) -> dict:
        return {"gemini_result": self._finalize_branch(state.get("gemini_messages", []), "GEMINI")}

    def _claude_finalize_node(self, state: dict) -> dict:
        return {"claude_result": self._finalize_branch(state.get("claude_messages", []), "CLAUDE")}

    # ── finish (joint voting) ─────────────────────────────────────────────────

    def _finish_node(self, state: dict) -> dict:
        gemini = state.get("gemini_result") or self._normalize_decision({}, "GEMINI", "")
        claude = state.get("claude_result") or self._normalize_decision({}, "CLAUDE", "")
        voting = self._joint_vote(gemini, claude)
        result = {
            "ip": state["ip"],
            "true_device_type": state["device_name"],
            "predicted_device_type": voting["final_device_type"],
            "device_type_reason": voting["final_device_type_reason"],
            "predicted_vendor": voting["final_vendor"],
            "vendor_reason": voting["final_vendor_reason"],
            "final_confidence": voting["final_confidence"],
            "winning_llm": voting["winning_llm"],
            "llm_agreement": voting["llm_agreement"],
            "first_stage": state.get("first_stage"),
            "gemini": voting["gemini"],
            "claude": voting["claude"],
        }
        return {"result": result}

    # ── public entry point ────────────────────────────────────────────────────

    def classify(self, ip: str, device_name: str, fingerprint: dict) -> dict:
        """Run the full decision graph for a single device IP."""
        self.runtime.register(ip, device_name, fingerprint)
        init_state = {
            "ip": ip,
            "device_name": device_name,
            "fingerprint": fingerprint,
            "reasoning_result": None,
            "local_result": None,
            "community_result": None,
            "web_search_results": "",
            "unseen_iter": 0,
            "unseen_messages": [],
            "unseen_result": {},
            "first_stage": {},
            "drift_result": None,
            "gemini_messages": [],
            "claude_messages": [],
            "gemini_result": {},
            "claude_result": {},
            "result": {},
        }
        t0 = time.time()
        try:
            final = self.graph.invoke(
                init_state, config={"recursion_limit": self.recursion_limit}
            )
        except Exception as exc:
            logging.error("Decision graph failed for %s: %s", ip, exc, exc_info=True)
            return {"error": str(exc), "ip": ip, "true_device_type": device_name}
        result = final.get("result") or {}
        result.setdefault("ip", ip)
        result.setdefault("true_device_type", device_name)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result


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
    parser.add_argument(
        "--decision", action="store_true", default=False,
        help="是否执行决策分类 (Gemini+Claude联合投票) / Whether to run decision classification"
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
    parser.add_argument(
        "--no_first_stage", action="store_true", default=False,
        help="禁用 unseen + drift 第一阶段 / Disable unseen + drift first stage"
    )
    parser.add_argument(
        "--unseen_adapter", type=str, default=None,
        help="unseen LLaMA LoRA adapter 路径 (可选) / Path to unseen LoRA adapter"
    )
    parser.add_argument(
        "--unseen_load_in_4bit", action="store_true", default=False,
        help="unseen LLaMA 使用 4-bit 量化 / Load unseen LLaMA in 4-bit"
    )
    parser.add_argument(
        "--drift_dir", type=str, default=None,
        help="训练好的 PACA drift 模型目录 / Directory with trained PACA drift model"
    )
    args = parser.parse_args()

    # 如果没有指定任何操作, 打印帮助
    # If no action specified, print help
    if not (args.vector or args.decompose or args.local or args.community or args.reasoning or args.decision):
        parser.print_help()
        print("\n请至少指定一个操作: --vector, --decompose, --local, --community, --reasoning, --decision")
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

    # ── 检索流程 / Retrieval pipeline ──
    need_retrieval = args.local or args.community or args.reasoning or args.decompose
    if need_retrieval:
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

    # ── 决策流程 / Decision pipeline (LangGraph) ──
    if args.decision:
        agent.run_decision(
            devices=device_types,
            quick_resume=args.quick_resume,
            whether_local=args.local,
            whether_community=args.community,
            whether_reasoning=args.reasoning,
            top_k=args.top_k,
            enable_first_stage=not args.no_first_stage,
            unseen_adapter_path=args.unseen_adapter,
            unseen_load_in_4bit=args.unseen_load_in_4bit,
            drift_model_dir=args.drift_dir,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()

