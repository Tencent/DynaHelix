# Cluster token based on contact_prob
import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from skimage import measure
import itertools


def enhence_iface_by_target(z, contact_prob, input_feature_dict, iface_indices, k=3, ikeep=0, T=0.9):
    o_device = z.device
    o_dtype = z.dtype

    z = z.to(device="cpu", dtype=torch.float32)
    contact_prob = contact_prob.to("cpu")

    asym_id = input_feature_dict["asym_id"]
    N_token = len(asym_id)
    target_len = sum(asym_id == 0)

    # - distance_prob_map: [n, n] tensor，表示 token 之间距离 < 8Å 的概率
    # z_pair=[N, L]
    # p = contact_prob[:-ligand_len, :-ligand_len].numpy()  # [N, N]
    p = contact_prob[:target_len, :target_len].numpy()  # [N, N]

    np.fill_diagonal(p, 1.0)

    # 2. 转换为距离矩阵（概率越高，距离越小）
    D = 1.0 - p + 1e-6
    np.fill_diagonal(D, 0)

    # 3. 转换为 condensed 格式（scipy 需要）
    condensed_dist = squareform(D)

    # 4. 层次聚类（使用 'average' 链接方法）
    Z = linkage(condensed_dist, method="average")

    # 5. 按阈值 8Å 划分聚类（调整 threshold 到你的需求）
    clusters = fcluster(Z, t=T, criterion="distance")  # t 需要调整

    # 2. 计算每个样本的重要性得分（例如取每行的最大值）
    # z_pair = torch.max(torch.abs(pair), dim=-1).values[:-ligand_len, -ligand_len:]
    z_pair = torch.max(torch.abs(z), dim=-1).values[:target_len, iface_indices]

    # sample_scores = np.max(z_pair, axis=1)  # [N,]
    sample_scores = torch.max(z_pair, dim=1).values.numpy()

    # 3. 计算每个类的平均得分
    unique_clusters = np.unique(clusters)
    cluster_scores = []
    for c in unique_clusters:
        mask = clusters == c
        cluster_score = np.mean(sample_scores[mask])  # 类的平均得分
        cluster_scores.append(cluster_score)

    # 4. 选择得分最高的 3 个类
    top_clusters = unique_clusters[np.argsort(cluster_scores)[-k:]]  # 取 top-3

    # 5. 生成新的聚类标签（top-3 类保留，其余标记为 -1）
    new_clusters = np.where(np.isin(clusters, top_clusters), clusters, -1)

    new_z = z.clone()

    # mask = []
    # for i in range(k):
    #     if i != ikeep:           # interface to be keep
    #         iface_i = np.where(new_clusters == top_clusters[i])[0].tolist()         # interface to be mask
    #         mask.extend(iface_i)

    # rows, cols = torch.meshgrid(torch.tensor(mask), torch.tensor(iface_indices), indexing='ij')
    # new_z[rows, cols, :] *= 0.01
    # new_z[cols, rows, :] *= 0.01

    # new_z[rows, cols, :] = 0.0
    # new_z[cols, rows, :] = 0.0

    # mask = []
    # for i in range(k):
    #     if i == ikeep:           # interface to be keep
    #         iface_i = np.where(new_clusters == top_clusters[i])[0].tolist()         # interface to be mask
    #         mask.extend(iface_i)

    # rows, cols = torch.meshgrid(torch.tensor(mask), torch.tensor(iface_indices), indexing='ij')

    # new_z[rows, cols, :] *= 3
    # new_z[cols, rows, :] *= 3

    # print("\n\n")
    # print(mask)
    # print(iface_indices)
    # print("\n\n")

    mask = []
    for i in range(k):
        iface_i = np.where(new_clusters == top_clusters[i])[0].tolist()  # interface to be mask
        mask.extend(iface_i)

    rows, cols = torch.meshgrid(torch.tensor(mask), torch.tensor(iface_indices), indexing="ij")

    new_z[rows, cols, :] *= 3
    new_z[cols, rows, :] *= 3

    return rows, cols, new_z.to(device=o_device, dtype=o_dtype)
