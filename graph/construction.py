"""
graph/construction.py - RAG层次图构建全流程编排
RAG Hierarchical Graph Construction Pipeline Orchestrator

Pipeline:
    Step 1 (--hgt):     HGT - 异构图Transformer设备嵌入学习
                        HGT device embedding generation on Device-Feature bipartite graph
    Step 2 (--cluster): Cluster - 多视角HDBSCAN聚类 + 综合视角聚类 + 聚类报告
                        Multi-perspective clustering + overall clustering + report generation
    Step 3 (--build):   Build - 构建三层层次化图到Neo4j
                        Build 3-layer hierarchical graph to Neo4j
    Step 4 (--vector):  Vector - 嵌入向量存储到Milvus向量数据库
                        Store embeddings into Milvus vector DB

Usage:
    # 全流程 / Full pipeline
    python graph/construction.py --hgt --cluster --build --vector --gpu 0

    # 仅HGT / HGT only
    python graph/construction.py --hgt --gpu 0 --epochs 200

    # 仅聚类 / Cluster only
    python graph/construction.py --cluster --gpu 0

    # 仅构建图 / Build only
    python graph/construction.py --build

    # 仅向量存储 / Vector store only
    python graph/construction.py --vector

    # 跳过聚类报告 / Skip report generation
    python graph/construction.py --cluster --no_report --gpu 0

    # 聚类报告错误恢复 / Report error recovery
    python graph/construction.py --cluster --recovery --gpu 0
"""

import os
import sys
import subprocess
import argparse
import logging
import time

# ─── 路径配置 / Path config ───────────────────────────────────────────
GRAPH_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(GRAPH_PATH)

# ─── 日志 / Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


class GraphConstruction:
    """
    RAG层次图构建全流程编排
    RAG Hierarchical Graph Construction Pipeline Orchestrator

    每个步骤以独立子进程运行, 避免GPU显存泄漏和模块冲突
    Each step runs in an isolated subprocess to prevent GPU memory leaks and module conflicts

    流程 / Pipeline:
        1. HGT:     异构图Transformer → 学习设备综合嵌入 (1024维)
        2. Cluster:  多视角HDBSCAN聚类 + 综合视角聚类 + LLM聚类报告
        3. Build:    三层层次化图构建到Neo4j (Entity → Community → ComCluster)
        4. Vector:   嵌入向量存入Milvus向量数据库
    """

    def __init__(self, gpu: int = 0):
        self.gpu = gpu
        self.graph_path = GRAPH_PATH
        self.base_path = BASE_PATH

    def _run_step(self, cmd: list, step_name: str):
        """
        执行子进程命令, 将stdout/stderr直接输出到当前终端
        Run a subprocess command, piping stdout/stderr to the current terminal
        """
        cmd_str = ' '.join(str(c) for c in cmd)
        print(f"\n{'='*60}")
        print(f"=== {step_name} ===")
        print(f"命令 / Command: {cmd_str}")
        print(f"{'='*60}")
        logging.info(f"{step_name} 开始, 命令: {cmd_str}")

        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.graph_path,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            msg = f"{step_name} 失败 (exit code: {result.returncode}), 耗时 {elapsed:.1f}s"
            print(f"错误: {msg}")
            logging.error(msg)
            raise RuntimeError(msg)

        print(f"\n{step_name} 完成, 耗时 {elapsed:.1f}s")
        logging.info(f"{step_name} 完成, 耗时 {elapsed:.1f}s")

    # ── Step 1: HGT ──────────────────────────────────────────────────

    def run_hgt(self, epochs: int = 100):
        """
        Step 1: 使用HGT在Device-Feature二部图上学习设备嵌入
        Run HGT to learn device embeddings on Device-Feature bipartite graph

        等价于 / Equivalent to:
            python HGT.py --gpu {gpu} --epochs {epochs}
        """
        self._run_step(
            [sys.executable, "HGT.py", "--gpu", str(self.gpu), "--epochs", str(epochs)],
            "Step 1: HGT 设备嵌入生成",
        )

    # ── Step 2: Cluster ──────────────────────────────────────────────

    def run_cluster(self, target: str = "all", overall: bool = True,
                    report: bool = True, recovery: bool = False):
        """
        Step 2: 多视角聚类 + 综合视角聚类 + 聚类报告生成
        Multi-perspective clustering + overall clustering + report generation

        等价于 / Equivalent to:
            python cluster.py --target all --gpu {gpu} --overall --report
        """
        cmd = [sys.executable, "cluster.py",
               "--target", target,
               "--gpu", str(self.gpu)]
        if overall:
            cmd.append("--overall")
        if report:
            cmd.append("--report")
        if recovery:
            cmd.append("--recovery")
        self._run_step(cmd, "Step 2: 多视角聚类")

    # ── Step 3: Build ────────────────────────────────────────────────

    def run_build(self):
        """
        Step 3: 构建三层层次化图到Neo4j
        Build 3-layer hierarchical graph into Neo4j

        Layer 1: Device Entity Graph       (Device ←Has_*→ Feature)
        Layer 2: Single-perspective Community (Device →IN_COMMUNITY→ SingleCommunity)
        Layer 3: Comprehensive-view Community (Device →IN_COMMUNITY→ ComCluster)

        等价于 / Equivalent to:
            python build.py
        """
        self._run_step(
            [sys.executable, "build.py"],
            "Step 3: 构建层次化图到Neo4j",
        )

    # ── Step 4: Vector ───────────────────────────────────────────────

    def run_vector(self, drop: bool = False, batch_size: int = 5000,
                   resume: bool = False):
        """
        Step 4: 将嵌入向量存储到Milvus向量数据库
        Store embeddings into Milvus vector database

        等价于 / Equivalent to:
            python vector.py [--drop] [--resume] --batch_size {batch_size}
        """
        cmd = [sys.executable, "vector.py",
               "--batch_size", str(batch_size)]
        if drop:
            cmd.append("--drop")
        if resume:
            cmd.append("--resume")
        self._run_step(cmd, "Step 4: 向量存储到Milvus")


def main():
    parser = argparse.ArgumentParser(
        description="RAG层次图构建全流程编排 / RAG Hierarchical Graph Construction Pipeline"
    )

    # ── 步骤开关 / Step switches ──
    parser.add_argument(
        "--hgt", action="store_true", default=False,
        help="是否执行HGT设备嵌入 / Whether to run HGT device embedding"
    )
    parser.add_argument(
        "--cluster", action="store_true", default=False,
        help="是否执行多视角聚类 / Whether to run multi-perspective clustering"
    )
    parser.add_argument(
        "--build", action="store_true", default=False,
        help="是否构建层次化图到Neo4j / Whether to build hierarchical graph to Neo4j"
    )
    parser.add_argument(
        "--vector", action="store_true", default=False,
        help="是否存储向量到Milvus / Whether to store vectors to Milvus"
    )

    # ── 通用参数 / Common parameters ──
    parser.add_argument(
        "--gpu", type=int, default=0, choices=[-1, 0, 1],
        help="GPU编号, -1为CPU / GPU device number, -1 for CPU (default: 0)"
    )

    # ── HGT 参数 / HGT parameters ──
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="HGT训练轮次 / HGT training epochs (default: 100)"
    )

    # ── Cluster 参数 / Cluster parameters ──
    parser.add_argument(
        "--cluster_target", type=str, default="all",
        help="聚类目标视角, 'all'为全部 / Cluster target perspective, 'all' for all (default: all)"
    )
    parser.add_argument(
        "--no_overall", action="store_true", default=False,
        help="跳过综合视角聚类 / Skip overall clustering"
    )
    parser.add_argument(
        "--no_report", action="store_true", default=False,
        help="跳过聚类报告生成 / Skip cluster report generation"
    )
    parser.add_argument(
        "--recovery", action="store_true", default=False,
        help="聚类报告错误恢复模式 / Cluster report error recovery mode"
    )

    # ── Vector 参数 / Vector parameters ──
    parser.add_argument(
        "--vector_drop", action="store_true", default=False,
        help="重建向量库 (删除已有collection) / Drop and rebuild vector collections"
    )
    parser.add_argument(
        "--vector_resume", action="store_true", default=False,
        help="向量存储恢复模式 / Vector store resume mode"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5000,
        help="向量插入批量大小 / Vector insert batch size (default: 5000)"
    )

    args = parser.parse_args()

    # 如果没有指定任何步骤, 打印帮助
    # If no step specified, print help
    if not (args.hgt or args.cluster or args.build or args.vector):
        parser.print_help()
        print("\n请至少指定一个步骤: --hgt, --cluster, --build, --vector")
        return

    # 添加文件日志处理器 / Add file log handler
    log_filename = os.path.join(GRAPH_PATH, "construction.log")
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    pipeline = GraphConstruction(gpu=args.gpu)

    t_total = time.time()
    steps_done = []

    try:
        if args.hgt:
            pipeline.run_hgt(epochs=args.epochs)
            steps_done.append("HGT")

        if args.cluster:
            pipeline.run_cluster(
                target=args.cluster_target,
                overall=not args.no_overall,
                report=not args.no_report,
                recovery=args.recovery,
            )
            steps_done.append("Cluster")

        if args.build:
            pipeline.run_build()
            steps_done.append("Build")

        if args.vector:
            pipeline.run_vector(
                drop=args.vector_drop,
                batch_size=args.batch_size,
                resume=args.vector_resume,
            )
            steps_done.append("Vector")

    except RuntimeError as e:
        total_elapsed = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"流程中断: {e}")
        print(f"已完成步骤: {' → '.join(steps_done) if steps_done else '无'}")
        print(f"总耗时: {total_elapsed:.1f}s")
        print(f"{'='*60}")
        logging.error(f"流程中断: {e}, 已完成步骤: {steps_done}, 总耗时: {total_elapsed:.1f}s")
        sys.exit(1)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"=== 全流程完成 (步骤: {' → '.join(steps_done)}) ===")
    print(f"=== 总耗时: {total_elapsed:.1f}s ===")
    print(f"{'='*60}")
    logging.info(f"全流程完成 (步骤: {' → '.join(steps_done)}), 总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
