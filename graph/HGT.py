"""
HGT.py - 使用 Heterogeneous Graph Transformer (HGT) 在Device-Feature二部图上
         学习每个设备IP的最终向量表示（综合视角嵌入）
         Device节点初始嵌入为11个视角嵌入的均值（1024维）
         Feature节点初始嵌入为Qwen3对"feature_name: value"字符串的编码（1024维）
         并对高度数Feature节点施加1/log(degree)惩罚因子
         最终向量表示的维度为1024维

Data is loaded from entity_graph/node.csv and entity_graph/relation.csv.
Results are saved to platform_data/csv/local/1/hgt_embeddings/

Usage:
    python graph/HGT.py
    python graph/HGT.py --gpu 0
    python graph/HGT.py --gpu -1  # CPU only
    python graph/HGT.py --epochs 200
"""

import os
import sys
import gc
import logging
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")

# ─── 路径配置 / Path config ───────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PATH = os.path.join(BASE_PATH, "platform_data", "csv", "local", "1")
ENTITY_GRAPH_PATH = os.path.join(BASE_PATH, "entity_graph")
EMBEDDING_MODEL_PATH = os.path.join(BASE_PATH, "qwen3_embedding_06b")

# 11个视角名称（与embedding_local CSV列前缀一致，排除hpart/http）
PERSPECTIVE_NAMES = ['as', 'whois', 'os', 'sw', 'hw', 'sd', 'body', 'htags', 'hfavicons', 'certificate', 'dns']

# ─── 日志 / Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8"


class HGTModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        # Device和Feature节点均为1024维初始嵌入，共享相同的输入维度
        self.device_lin = Linear(in_channels, hidden_channels)
        self.feature_lin = Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=0.1)

        node_types = metadata[0]
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)
            # 每层对每种节点类型独立的LayerNorm
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels) for nt in node_types
            }))

        self.device_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 输入投影
        x_dict['device'] = self.device_lin(x_dict['device']).relu()
        x_dict['feature'] = self.feature_lin(x_dict['feature']).relu()

        # HGT层：residual + Dropout + LayerNorm（公式3）
        # h_v^(l+1) = LayerNorm(h_v^(l) + Dropout(h̃_v^(l+1)))
        for conv, norm_dict in zip(self.convs, self.norms):
            h_dict = conv(x_dict, edge_index_dict)
            new_x_dict = {}
            for nt in x_dict:
                if nt in h_dict and nt in norm_dict:
                    new_x_dict[nt] = norm_dict[nt](x_dict[nt] + self.dropout(h_dict[nt]))
                else:
                    new_x_dict[nt] = x_dict[nt]
            x_dict = new_x_dict

        # 输出投影（仅对device节点）
        device_emb = self.device_out(x_dict['device'])
        return device_emb


def build_embedding_model(gpu: int):
    device_str = str(gpu) if gpu != -1 else "cpu"
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device_str},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
        query_encode_kwargs={"normalize_embeddings": True, "batch_size": 1},
    )
    if gpu != -1:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return model


def load_device_embeddings(device_nodes: pd.DataFrame) -> torch.Tensor:
    """
    从预计算的embedding_local CSV文件中加载各Device节点的11个视角嵌入，
    取均值得到1024维初始嵌入向量。
    """
    embedding_local_path = os.path.join(LOCAL_PATH, "embedding_local")
    ip_to_emb = {}

    if 'device_type' in device_nodes.columns:
        for dev_type, group in device_nodes.groupby('device_type'):
            csv_path = os.path.join(embedding_local_path, f"ipraw_{dev_type}_embedding.csv")
            if not os.path.exists(csv_path):
                logging.warning(f"[HGT] Embedding CSV not found: {csv_path}")
                for ip in group['ip'].values:
                    ip_to_emb[str(ip)] = [0.0] * 1024
                continue

            logging.info(f"[HGT] Loading embedding CSV for {dev_type}...")
            emb_df = pd.read_csv(csv_path)
            emb_df['ip'] = emb_df['ip'].astype(str)

            # 按视角名称确定对应的列组（每视角1024列）
            perspective_col_groups = []
            for p in PERSPECTIVE_NAMES:
                cols = [c for c in [f"{p}{i+1}" for i in range(1024)] if c in emb_df.columns]
                perspective_col_groups.append(cols)

            target_ips = {str(ip) for ip in group['ip'].values}
            matched = emb_df[emb_df['ip'].isin(target_ips)]

            for _, row in matched.iterrows():
                ip_str = row['ip']
                persp_embs = []
                for cols in perspective_col_groups:
                    if cols:
                        persp_embs.append(row[cols].values.astype(float))
                avg = sum(persp_embs) / len(persp_embs) if persp_embs else [0.0] * 1024
                ip_to_emb[ip_str] = avg.tolist() if hasattr(avg, 'tolist') else list(avg)

            for ip in group['ip'].values:
                if str(ip) not in ip_to_emb:
                    ip_to_emb[str(ip)] = [0.0] * 1024
    else:
        logging.warning("[HGT] device_type column not found; using zero embeddings for devices.")
        for ip in device_nodes['ip'].values:
            ip_to_emb[str(ip)] = [0.0] * 1024

    features = [ip_to_emb.get(str(ip), [0.0] * 1024) for ip in device_nodes['ip'].values]
    return torch.tensor(features, dtype=torch.float)


def run_hgt(gpu: int = 1, num_epochs: int = 100):
    """
    使用Heterogeneous Graph Transformer (HGT)在Device-Feature二部图上
    学习每个设备IP的最终向量表示（综合视角嵌入），最终向量维度为1024维。
    """

    print("[HGT] Starting HGT device embedding generation...")
    logging.info("[HGT] Starting HGT device embedding generation...")

    # Step 1: 加载节点和关系数据
    print("[HGT] Loading graph data...")
    logging.info("[HGT] Loading graph data...")

    node_df = pd.read_csv(os.path.join(ENTITY_GRAPH_PATH, "node.csv"))
    relation_df = pd.read_csv(os.path.join(ENTITY_GRAPH_PATH, "relation.csv"))

    # Step 2: 分离Device和Feature节点
    device_nodes = node_df[node_df['_labels'] == ':Device'].copy().reset_index(drop=True)
    feature_nodes = node_df[node_df['_labels'] == ':Feature'].copy().reset_index(drop=True)

    print(f"[HGT] Device nodes: {len(device_nodes)}, Feature nodes: {len(feature_nodes)}")
    logging.info(f"[HGT] Device nodes: {len(device_nodes)}, Feature nodes: {len(feature_nodes)}")

    # Step 3: 创建节点ID映射
    device_id_map = {int(row['_id']): idx for idx, row in device_nodes.iterrows()}
    feature_id_map = {int(row['_id']): idx for idx, row in feature_nodes.iterrows()}

    # Step 4: 构建边索引（支持多种Has_*边类型）
    print("[HGT] Building edge indices...")
    edge_dict = {}  # rel_type -> list of [device_idx, feature_idx]

    for _, row in relation_df.iterrows():
        if pd.isna(row['_start']) or pd.isna(row['_end']) or pd.isna(row['_type']):
            continue
        rel_type = str(row['_type'])
        if not rel_type.startswith('Has_'):
            continue
        start_id = int(row['_start'])
        end_id = int(row['_end'])
        if start_id in device_id_map and end_id in feature_id_map:
            if rel_type not in edge_dict:
                edge_dict[rel_type] = []
            edge_dict[rel_type].append([device_id_map[start_id], feature_id_map[end_id]])

    print(f"[HGT] Edge types found: {list(edge_dict.keys())}")
    logging.info(f"[HGT] Edge types found: {list(edge_dict.keys())}")

    # Step 5: 计算Feature节点的degree（连接的Device数量），用于惩罚因子1/log(degree)
    feature_degree = torch.zeros(len(feature_nodes), dtype=torch.float)
    for edges in edge_dict.values():
        for _, feature_idx in edges:
            feature_degree[feature_idx] += 1

    # Step 6: 构建Device节点初始嵌入（从预计算的11个视角嵌入取均值，1024维）
    print("[HGT] Building device node features from perspective embeddings...")
    logging.info("[HGT] Building device node features from perspective embeddings...")
    device_features = load_device_embeddings(device_nodes)
    print(f"[HGT] Device feature shape: {device_features.shape}")

    # Step 7: 构建Feature节点初始嵌入（"feature_name: value" → Qwen3嵌入 + degree惩罚）
    print("[HGT] Building feature node features...")
    logging.info("[HGT] Building feature node features...")
    embedding_model = build_embedding_model(gpu)

    feature_features_list = []
    for _, row in feature_nodes.iterrows():
        feat_name = str(row['feature_name']) if pd.notna(row.get('feature_name')) else ""
        feat_val = str(row['value']) if pd.notna(row.get('value')) else ""
        feat_str = f"{feat_name}: {feat_val}"
        emb = embedding_model.embed_query(feat_str)
        feature_features_list.append(emb)

    feature_features = torch.tensor(feature_features_list, dtype=torch.float)

    # 应用degree惩罚因子：1/log(degree)，degree<=1时惩罚因子取1.0
    penalty = torch.where(
        feature_degree > 1,
        1.0 / torch.log(feature_degree),
        torch.ones_like(feature_degree)
    )
    feature_features = feature_features * penalty.unsqueeze(1)
    print(f"[HGT] Feature feature shape: {feature_features.shape}")

    # Step 8: 创建异构图数据
    print("[HGT] Creating heterogeneous graph...")
    data = HeteroData()
    data['device'].x = device_features
    data['feature'].x = feature_features

    for rel_type, edges in edge_dict.items():
        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data['device', rel_type, 'feature'].edge_index = edge_tensor
        # 添加反向边（Feature→Device），支持第一层Feature聚合Device信息
        data['feature', f'rev_{rel_type}', 'device'].edge_index = edge_tensor.flip([0])

    print(f"[HGT] Total edge types (incl. reverse): {len(data.edge_index_dict)}")

    # Step 9: 初始化模型并移到GPU
    print("[HGT] Initializing HGT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HGT] Using device: {device}")
    logging.info(f"[HGT] Using device: {device}")

    model = HGTModel(
        in_channels=1024,
        hidden_channels=512,
        out_channels=1024,
        num_heads=8,
        num_layers=2,
        metadata=data.metadata(),
    ).to(device)
    data = data.to(device)

    # Step 10: 训练模型（自监督学习）
    print("[HGT] Training HGT model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        device_emb = model(data.x_dict, data.edge_index_dict)

        # 对比学习损失：从所有边类型中采样正样本对（Device-Feature对应相似）
        all_edge_tensors = [
            data['device', rel_type, 'feature'].edge_index
            for rel_type in edge_dict
        ]
        if all_edge_tensors:
            all_edges = torch.cat(all_edge_tensors, dim=1)
            num_samples = min(1000, all_edges.shape[1])
            sample_idx = torch.randperm(all_edges.shape[1])[:num_samples]
            sampled_edges = all_edges[:, sample_idx]

            device_idx = sampled_edges[0]
            feature_idx = sampled_edges[1]

            device_emb_sample = device_emb[device_idx]
            feature_emb_sample = data.x_dict['feature'][feature_idx]
            feature_proj = model.device_out(model.feature_lin(feature_emb_sample))

            cos_sim = F.cosine_similarity(device_emb_sample, feature_proj, dim=1)
            loss = 1 - cos_sim.mean()
            loss += 0.001 * (device_emb.norm(2) / device_emb.shape[0])
        else:
            loss = 0.001 * (device_emb.norm(2) / device_emb.shape[0])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[HGT] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            logging.info(f"[HGT] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Step 11: 生成最终的device embeddings
    print("[HGT] Generating final device embeddings...")
    model.eval()
    with torch.no_grad():
        final_device_emb = model(data.x_dict, data.edge_index_dict)
        final_device_emb = final_device_emb.cpu().numpy()

    # Step 12: 保存结果
    print("[HGT] Saving device embeddings...")
    hgt_save_path = os.path.join(LOCAL_PATH, "hgt_embeddings")
    os.makedirs(hgt_save_path, exist_ok=True)

    embedding_cols = [f"hgt_emb_{i}" for i in range(1024)]
    result_df = pd.DataFrame(final_device_emb, columns=embedding_cols)
    result_df.insert(0, 'ip', device_nodes['ip'].values)

    output_path = os.path.join(hgt_save_path, "device_hgt_embeddings.csv")
    result_df.to_csv(output_path, index=False)

    print(f"[HGT] Device embeddings saved to: {output_path}")
    print(f"[HGT] Total devices: {len(result_df)}, Embedding dimension: 1024")
    logging.info(f"[HGT] Device embeddings saved to: {output_path}")
    logging.info(f"[HGT] Total devices: {len(result_df)}, Embedding dimension: 1024")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="HGT Device Embedding Generation")
    parser.add_argument(
        "--gpu", type=int, default=1, choices=[-1, 0, 1],
        help="GPU device number to use (0 or 1), -1 for CPU only"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    args = parser.parse_args()

    log_filename = "HGT.log"
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    run_hgt(gpu=args.gpu, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
