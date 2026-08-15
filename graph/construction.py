"""
graph/construction.py - RAG层次图构建全流程编排
RAG Hierarchical Graph Construction Pipeline Orchestrator

Pipeline (executed in TRUE data-dependency order):
    Step 1 (--cluster): Per-perspective Qwen3 embedding + HDBSCAN clustering +
                        embedding_local concatenation (+ optional community report)
    Step 2 (--build):   Build Layer-1 Device–Feature entity graph to Neo4j, then
                        EXPORT entity_graph/rag_only/{node,relation}.csv (HGT inputs)
                        via export_entity_graph.py --mode rag_only
    Step 3 (--hgt):     HGT - learn comprehensive device embeddings on the
                        Device–Feature graph (needs entity_graph + embedding_local)
    Step 4 (--hgt):     Comprehensive-view clustering on HGT embeddings
                        (cluster.py --hgt → community/embedding_overall)
    Step 5 (--build):   Build Layer-2 / Layer-3 community graphs to Neo4j
    Step 6 (--vector):  Store embeddings into Milvus vector DB + local npz

Why this order? HGT.py reads entity_graph/{node,relation}.csv (produced by build
Layer-1 export) AND embedding_local/*.csv (produced by cluster). So cluster+build
MUST precede HGT; and Layer-2/3 community graphs depend on the HGT comprehensive
clustering. The single-command full pipeline below runs everything in this order.

Usage:
    # 全流程 (推荐) / Full pipeline in correct order (recommended)
    python graph/construction.py --all --gpu 0

    # 等价于分步开关 (会被自动重排为正确顺序) / Equivalent step switches (auto-ordered)
    python graph/construction.py --cluster --build --hgt --vector --gpu 0

    # 仅聚类 / Cluster only
    python graph/construction.py --cluster --gpu 0

    # 仅构建图 (Layer1 + 导出 + Layer2/3) / Build graph only
    python graph/construction.py --build

    # 仅HGT (综合嵌入 + 综合聚类) / HGT only (embedding + comprehensive clustering)
    python graph/construction.py --hgt --gpu 0 --epochs 200

    # 仅向量存储 / Vector store only
    python graph/construction.py --vector

    # 跳过聚类报告 / Skip report generation
    python graph/construction.py --cluster --no_report --gpu 0
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

    def __init__(self, gpu: int = 0, devices: list = None):
        self.gpu = gpu
        self.devices = devices
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

    # ── Step: HGT (embedding + comprehensive-view clustering) ────────────

    def run_hgt(self, epochs: int = 100):
        """
        HGT: learn comprehensive device embeddings on the Device–Feature graph,
        then run comprehensive-view clustering on those embeddings.

        Requires (produced by earlier steps):
          - entity_graph/{node,relation}.csv   (build --layer1 --export)
          - embedding_local/*.csv              (cluster)

        等价于 / Equivalent to:
            python HGT.py --gpu {gpu} --epochs {epochs}
            python cluster.py --hgt --gpu {gpu}
        """
        cmd = [sys.executable, "HGT.py", "--gpu", str(self.gpu), "--epochs", str(epochs)]
        if self.devices:
            cmd.extend(["--devices"] + self.devices)
        self._run_step(
            cmd,
            "Step: HGT 设备综合嵌入生成",
        )
        # Comprehensive-View Clustering on HGT embeddings (Fig.1 "Comprehensive View Clustering")
        cmd2 = [sys.executable, "cluster.py", "--hgt", "--gpu", str(self.gpu)]
        if self.devices:
            cmd2.extend(["--devices"] + self.devices)
        self._run_step(
            cmd2,
            "Step: HGT 综合视角聚类",
        )

    # ── Step: Cluster ────────────────────────────────────────────────

    def run_cluster(self, target: str = "all", overall: bool = False,
                    report: bool = True, recovery: bool = False):
        """
        Per-perspective Qwen3 embedding + HDBSCAN clustering + embedding_local
        concatenation (+ optional community report). This is the FIRST data step;
        it produces the per-perspective embeddings that both HGT and the entity
        graph build consume.

        等价于 / Equivalent to:
            python cluster.py --target all --gpu {gpu} [--report]
        """
        cmd = [sys.executable, "cluster.py",
               "--target", target,
               "--gpu", str(self.gpu)]
        if self.devices:
            cmd.extend(["--devices"] + self.devices)
        if overall:
            cmd.append("--overall")
        if report:
            cmd.append("--report")
        if recovery:
            cmd.append("--recovery")
        self._run_step(cmd, "Step: 多视角聚类 (Qwen3 + HDBSCAN)")

    # ── Step: Build (Layer1 + export, and Layer2/3) ──────────────────

    def run_build_layer1_export(self):
        """
        Build Layer-1 Device–Feature entity graph to Neo4j, then export
        entity_graph/rag_only/{node,relation}.csv as HGT inputs.

        等价于 / Equivalent to:
            python build.py --layer1
            python export_entity_graph.py --mode rag_only
        """
        cmd = [sys.executable, "build.py", "--layer1"]
        if self.devices:
            cmd.extend(["--devices"] + self.devices)
        self._run_step(
            cmd,
            "Step: 构建 Layer1 实体图到Neo4j",
        )
        self._run_step(
            [sys.executable, "export_entity_graph.py", "--mode", "rag_only"],
            "Step: 导出 entity_graph CSV (Device+Feature, for HGT)",
        )

    def run_build_layer23(self):
        """
        Build Layer-2 (single-perspective community) and Layer-3
        (comprehensive-view community) graphs to Neo4j. Depends on the HGT
        comprehensive clustering output.

        等价于 / Equivalent to:
            python build.py --layer23
        """
        cmd = [sys.executable, "build.py", "--layer23"]
        if self.devices:
            cmd.extend(["--devices"] + self.devices)
        self._run_step(
            cmd,
            "Step: 构建 Layer2/3 社区图到Neo4j",
        )

    # ── Step 4: Vector ───────────────────────────────────────────────

    def run_vector(self, drop: bool = False, batch_size: int = 5000,
                   resume: bool = False):
        """
        Step 4: 向量存储流水线: Milvus (单 perspective) + local npz (多 perspective 拼接)
        Vector storage pipeline: Milvus (per-perspective) + local npz (multi-perspective concat)

        等价于 / Equivalent to:
            python vector.py [--drop] [--resume] --batch_size {batch_size}
        """
        cmd = [sys.executable, "vector.py",
               "--batch_size", str(batch_size)]
        if self.devices:
            cmd.extend(["--devices"] + self.devices)
        if drop:
            cmd.append("--drop")
        if resume:
            cmd.append("--resume")
        self._run_step(cmd, "Step 4: 向量存储 (Milvus + local npz)")


def main():
    parser = argparse.ArgumentParser(
        description="RAG层次图构建全流程编排 / RAG Hierarchical Graph Construction Pipeline"
    )

    # ── 步骤开关 / Step switches ──
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="执行完整流程 (按正确顺序: cluster→build→hgt→vector) / Run full pipeline in correct order"
    )
    parser.add_argument(
        "--hgt", action="store_true", default=False,
        help="是否执行HGT设备嵌入 + 综合视角聚类 / Whether to run HGT embedding + comprehensive clustering"
    )
    parser.add_argument(
        "--cluster", action="store_true", default=False,
        help="是否执行多视角聚类 / Whether to run multi-perspective clustering"
    )
    parser.add_argument(
        "--build", action="store_true", default=False,
        help="是否构建层次化图到Neo4j (Layer1+导出+Layer2/3) / Whether to build hierarchical graph to Neo4j"
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
        "--overall", action="store_true", default=False,
        help="额外运行旧版拼接式综合聚类 (cluster.py --overall); 默认综合聚类走 HGT 版 / "
             "Also run legacy concat-based overall clustering; by default comprehensive clustering uses the HGT path"
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
    parser.add_argument(
        "--devices", nargs="*", default=None,
        help="设备类型范围 / Device types to process (default: all from rag_devices.json)"
    )

    args = parser.parse_args()

    # 如果没有指定任何步骤, 打印帮助
    # If no step specified, print help
    if not (args.all or args.hgt or args.cluster or args.build or args.vector):
        parser.print_help()
        print("\n请至少指定一个步骤: --all, --cluster, --build, --hgt, --vector")
        return

    # --all 展开为全部步骤 / --all expands to every step
    do_cluster = args.all or args.cluster
    do_build   = args.all or args.build
    do_hgt     = args.all or args.hgt
    do_vector  = args.all or args.vector

    # 添加文件日志处理器 / Add file log handler
    log_filename = os.path.join(GRAPH_PATH, "construction.log")
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    pipeline = GraphConstruction(gpu=args.gpu, devices=args.devices)

    t_total = time.time()
    steps_done = []

    # 无论开关书写顺序如何, 始终按正确的数据依赖顺序执行:
    #   cluster → build(Layer1+export) → HGT(+综合聚类) → build(Layer2/3) → vector
    # Always run in TRUE data-dependency order regardless of flag order:
    #   cluster → build(Layer1+export) → HGT(+comprehensive clustering) → build(Layer2/3) → vector
    try:
        # Step 1: 逐视角聚类 (产出 embedding_local, 供 build/HGT 使用)
        if do_cluster:
            pipeline.run_cluster(
                target=args.cluster_target,
                overall=args.overall,          # 默认关闭; 综合聚类走 HGT 版
                report=not args.no_report,
                recovery=args.recovery,
            )
            steps_done.append("Cluster")

        # Step 2: 构建 Layer1 实体图 + 导出 entity_graph CSV (HGT 输入)
        if do_build:
            pipeline.run_build_layer1_export()
            steps_done.append("Build-Layer1+Export")

        # Step 3: HGT 综合嵌入 + 综合视角聚类 (依赖 entity_graph + embedding_local)
        if do_hgt:
            pipeline.run_hgt(epochs=args.epochs)
            steps_done.append("HGT+ComprehensiveCluster")

        # Step 4: 构建 Layer2/3 社区图 (依赖综合聚类结果)
        if do_build:
            pipeline.run_build_layer23()
            steps_done.append("Build-Layer2/3")

        # Step 5: 向量入库 Milvus
        if do_vector:
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
