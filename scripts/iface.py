# Copyright 2026 Tencent and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scripts.dyna_utils import DynaHelixDataset, contact_logits2probs, tokens_to_eid_cid
import numpy as np
from typing import List, Dict, Set, Any
import torch
import random


class ReceptorHotspotClustering:
    """
    蛋白贴片驱动的共接触聚类算法，增加同一iface内DNA token距离限制，并对iface进行二次聚类
    """

    def __init__(
        self,
        tau: float = 0.6,
        M: int = 8,
        m: int = 2,
        min_iface_prob: float = 0.3,
        iface_cluster_threshold: float = 0.3,
    ):
        """
        初始化聚类参数

        参数:
            tau: 相似度阈值
            M: 最大iface数
            m: 每个iface选取的token数
            min_iface_prob: 最小界面概率
            iface_cluster_threshold: iface间聚类阈值
        """
        self.tau = tau
        self.M = M
        self.m = m
        self.min_iface_prob = min_iface_prob
        self.iface_cluster_threshold = iface_cluster_threshold

    def get_receptor_hotspots(
        self,
        score_feature: np.ndarray,
        contact_prob: np.ndarray,
        contact_prob_close: np.ndarray,
        binder_indices: List[int],
        receptor_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        执行完整的受体热点聚类流程

        参数:
            score_feature: 特征矩阵
            contact_prob: [N, N] 接触概率矩阵
            contact_prob_close: [N, N] 近距离接触概率矩阵(用于iface间聚类)
            binder_indices: 蛋白残基索引
            receptor_indices: Receptor残基索引

        返回:
            聚类结果列表，每个元素包含iface信息
        """
        if len(score_feature.shape) == 2:
            # contact probs
            C = score_feature[binder_indices][:, receptor_indices].T  # for contact_probs
            s = np.max(C, axis=1)  # [receptor_len]
        else:
            # z feature
            C = score_feature[binder_indices][:, receptor_indices].transpose(1, 0, 2)  # [N_p, N_d, d]
            s = np.max(np.abs(C), axis=-1).max(-1)  # max[d] and max[N_p]->[receptor_len]

        # 3. 计算DNA-DNA共贴片相似度矩阵
        # 添加小常数避免除以零
        C = C.reshape(C.shape[0], -1)
        norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
        V = C / norms  # L2归一化
        S = V @ V.T  # cosine相似度矩阵 [receptor_len, receptor_len]

        # 4. 提取DNA-DNA距离信息 (从contact_prob_20)
        # 使用20Å接触概率作为距离代理 (概率越高表示距离越近)
        receptor_len = len(receptor_indices)
        receptor_contact = contact_prob[receptor_indices][:, receptor_indices]

        # 5. 贪心聚类 (增加距离约束)
        ifaces = self._first_level_clustering(s, S, receptor_contact, receptor_len)

        # 6. 为每个iface打分并选择token
        results = self._score_and_select_tokens(ifaces, s, receptor_indices)

        # 7. 对iface进行二次聚类（基于contact_prob_close）
        iface_clusters = self._cluster_ifaces(results, contact_prob_close)

        # print("ifaces: ", ifaces)
        # print("results: ", results)
        # print("iface_clusters: ", iface_clusters)

        # 8. 从每个类中选择最高分的iface
        final_results = self._select_best_ifaces_per_cluster(iface_clusters)

        return final_results

    def _first_level_clustering(
        self,
        s: np.ndarray,
        S: np.ndarray,
        receptor_contact: np.ndarray,
        receptor_len: int,
    ) -> List[Set[int]]:
        """
        第一级聚类：基于相似度和距离约束

        参数:
            s: 热点强度分数
            S: 相似度矩阵
            receptor_contact: 受体接触矩阵
            receptor_len: 受体长度

        返回:
            iface列表，每个iface包含token索引
        """
        ifaces = []
        unassigned = set(range(receptor_len))

        # 按热点强度排序
        sorted_indices = np.argsort(s)[::-1]

        for idx in sorted_indices:
            if idx not in unassigned:
                continue

            # 创建新iface
            seed = idx
            iface = {seed}
            unassigned.remove(seed)

            # 查找相似且距离足够近的token
            for j in list(unassigned):
                # 检查相似度和距离约束
                if S[seed, j] >= self.tau:
                    # 检查iface内所有token与新token的距离
                    valid = True
                    # 只检查是否与hotspot在足够近的距离
                    if receptor_contact[j, seed] < self.min_iface_prob:
                        valid = False

                    if valid:
                        iface.add(j)
                        # do not put it back to the bag
                        unassigned.remove(j)

            ifaces.append(iface)

            # 达到最大iface数时停止
            if len(ifaces) >= self.M:
                break

        return ifaces

    def _score_and_select_tokens(
        self, ifaces: List[Set[int]], s: np.ndarray, receptor_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        为每个iface打分并选择token

        参数:
            ifaces: iface列表
            s: 热点强度分数
            receptor_indices: 受体索引

        返回:
            包含iface信息的字典列表
        """
        results = []
        for iface in ifaces:
            # 计算iface分数
            iface_score = max(s[j] for j in iface)
            iface_mean = np.mean([s[j] for j in iface])

            # 选择top m个token
            iface_tokens = sorted(iface, key=lambda j: s[j], reverse=True)[: self.m]
            token_ids = [receptor_indices[j] for j in iface_tokens]  # 转换为全局索引

            results.append(
                {
                    "score": iface_score,
                    "mean_score": iface_mean,
                    "token_ids": token_ids,
                    "iface_size": len(iface),
                    "center_token": token_ids[0],  # 第一个token作为中心
                }
            )

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _cluster_ifaces(
        self, ifaces: List[Dict[str, Any]], contact_prob_close: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """
        基于contact_prob_close对iface进行聚类

        参数:
            ifaces: iface列表
            contact_prob_close: 近距离接触概率矩阵

        返回:
            iface聚类列表
        """
        if not ifaces:
            return []

        n_ifaces = len(ifaces)
        # 计算iface间的连接强度矩阵
        connection_strength = np.zeros((n_ifaces, n_ifaces))

        for i in range(n_ifaces):
            for j in range(i + 1, n_ifaces):
                # 计算两个iface间的平均接触概率
                strength = self._calculate_iface_connection(ifaces[i], ifaces[j], contact_prob_close)
                connection_strength[i, j] = strength
                connection_strength[j, i] = strength

        # 贪心聚类算法
        clusters = []
        unassigned = set(range(n_ifaces))

        while unassigned:
            # 选择未分配的最高分iface作为种子
            seed = max(unassigned, key=lambda idx: ifaces[idx]["score"])
            cluster = {seed}
            unassigned.remove(seed)

            # 寻找与种子iface连接强度足够的iface
            for j in list(unassigned):
                if connection_strength[seed, j] >= self.iface_cluster_threshold:
                    cluster.add(j)
                    unassigned.remove(j)

            # 将cluster中的索引转换为实际的iface对象
            cluster_ifaces = [ifaces[i] for i in cluster]
            clusters.append(cluster_ifaces)

        return clusters

    def _calculate_iface_connection(
        self,
        iface1: Dict[str, Any],
        iface2: Dict[str, Any],
        contact_prob_close: np.ndarray,
    ) -> float:
        """
        计算两个iface间的连接强度

        参数:
            iface1: 第一个iface
            iface2: 第二个iface
            contact_prob_close: 近距离接触概率矩阵

        返回:
            两个iface间的平均接触概率
        """
        tokens1 = iface1["token_ids"]
        tokens2 = iface2["token_ids"]

        strengths = []
        for token1 in tokens1:
            for token2 in tokens2:
                strengths.append(contact_prob_close[token1, token2])

        return np.mean(strengths) if strengths else 0.0

    def _select_best_ifaces_per_cluster(self, iface_clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        从每个聚类中选择最高分的iface

        参数:
            iface_clusters: iface聚类列表

        返回:
            每个聚类中最高分的iface列表
        """
        final_results = []
        for cluster in iface_clusters:
            if cluster:  # 确保类不为空
                # 找到类中分数最高的iface
                best_iface = max(cluster, key=lambda x: x["score"])
                best_iface["cluster_size"] = len(cluster)  # 添加类大小信息
                final_results.append(best_iface)

        # 按分数重新排序
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results

    def set_parameters(
        self,
        tau: float = None,
        M: int = None,
        m: int = None,
        min_iface_prob: float = None,
        iface_cluster_threshold: float = None,
    ):
        """
        动态设置聚类参数

        参数:
            tau: 相似度阈值
            M: 最大iface数
            m: 每个iface选取的token数
            min_iface_prob: 最小界面概率
            iface_cluster_threshold: iface间聚类阈值
        """
        if tau is not None:
            self.tau = tau
        if M is not None:
            self.M = M
        if m is not None:
            self.m = m
        if min_iface_prob is not None:
            self.min_iface_prob = min_iface_prob
        if iface_cluster_threshold is not None:
            self.iface_cluster_threshold = iface_cluster_threshold


def get_indices_by_eid_cid(query_ids, entity_id, copy_id):
    mask_indices = np.zeros_like(entity_id)
    for identity in query_ids:
        query_e_id, query_c_id = identity
        mask = entity_id == query_e_id
        mask &= copy_id == query_c_id
        mask_indices[mask] = 1
    # get indices
    indices = np.where(mask_indices == 1)[0]
    return indices


def gen_constraint_json(chain_res, base_json, binder_ids, top=10, max_dist=6.5):
    final = []
    for i, tokens in enumerate(chain_res[:top]):
        c_id = i + 1
        currnt_json = base_json.copy()
        currnt_json["name"] = f"{currnt_json['name']}-cluster-{c_id}"
        # convert asym_id to str
        for token in tokens:
            token[0] = str(token[0])
        #
        constraint = {}
        constraint["pocket"] = {
            "binder_chain": [f"{binder_ids[0][0]}", int(binder_ids[0][1])],
            "contact_residues": tokens[:10],
            "max_distance": max_dist,
        }
        currnt_json["constraint"] = constraint
        final.append(currnt_json)
    # output
    return final


def get_constraints(base_json):
    receptor_ids = []
    binder_ids = []

    name = base_json["name"]
    entity_counter = 1
    for seq_info in base_json["sequences"]:
        assert len(seq_info) == 1
        seq_type = next(iter(seq_info))
        if seq_type in ["ligand"]:
            continue

        count = seq_info[seq_type]["count"]

        entity = entity_counter
        copy_ids = [i + 1 for i in range(count)]  # start from 1

        isbinder = seq_info[seq_type]["isbinder"]
        istarget = seq_info[seq_type]["istarget"]

        if isbinder == 1 and istarget == 1:
            print(f"error in {name}: a chain connot be both binder and target")
            return [], []

        if isbinder == 1 and istarget == 0:
            for copy_id in copy_ids[:1]:  # select the first copy only
                binder_ids.append((entity, copy_id))

        if isbinder == 0 and istarget == 1:  # select the first copy only
            for copy_id in copy_ids[:1]:
                receptor_ids.append((entity, copy_id))

        entity_counter = entity_counter + 1

    return receptor_ids, binder_ids, len(receptor_ids), len(receptor_ids)


def generate_iface_inputs(
    base_json,
    name,
    seed,
    data_root,
    surface_finder,
    binding_type="protein-dna",
    num_samples=1,
):
    if "binding_type" in base_json:
        binding_type = base_json["binding_type"]

    try:
        receptor_ids, binder_ids_, len_receptor_ids, len_binder_ids = get_constraints(base_json)
    except (KeyError, IndexError, TypeError, ValueError):
        if binding_type == "protein-dna":
            receptor_ids = [(2, 1), (3, 1)]
            binder_ids_ = [(1, 1)]

    print("binder_ids_: ", binder_ids_)
    binder_ids = [random.choice(binder_ids_)]  # in case there are many binder ids

    print("data_root: ", data_root)
    train_set = DynaHelixDataset(
        data_root=data_root  # dyna, constraint
    )

    # retrieve z
    result = train_set.get_by_name(f"{name}-{seed}", num_samples=num_samples)

    # desired query
    queries = {}  # set(["1-1-45", "1-1-34", "1-1-72", "1-1-83", "1-1-40"])  # 8hij five hbonds
    # constant
    first_entry = result[0]["data"]
    entity_id = (first_entry["input_feature_dict"]["entity_id"] + 1).numpy()
    copy_id = (first_entry["input_feature_dict"]["sym_id"] + 1).numpy()
    # get indices
    binder_indices = get_indices_by_eid_cid(binder_ids, entity_id, copy_id)
    receptor_indices = get_indices_by_eid_cid(receptor_ids, entity_id, copy_id)

    # z features
    all_z = [item["data"]["z"].unsqueeze(0) for item in result]
    all_contact_logits = [item["data"]["contact_logits"].unsqueeze(0) for item in result]
    #####
    z = torch.cat(all_z).mean(0).float()  # [N, N, 128]
    z = z.numpy()

    # contact
    contact_logits = torch.cat(all_contact_logits).mean(0).float()  # [N, N, 64]
    contact_prob_20 = contact_logits2probs(contact_logits, 20).numpy()  # [N, N]
    contact_prob_close = contact_logits2probs(contact_logits, surface_finder.contact_prob_dist_close).numpy()  # [N, N]

    # select score features
    if surface_finder.use_z:
        score_feature = z
    else:
        contact_prob_score = contact_logits2probs(contact_logits, surface_finder.contact_prob_dist).numpy()
        score_feature = contact_prob_score

    # 执行聚类算法
    clustering = ReceptorHotspotClustering(
        tau=surface_finder.tau,
        M=surface_finder.M,
        m=surface_finder.m,
        min_iface_prob=0.8,
    )
    ifaces = clustering.get_receptor_hotspots(
        score_feature=score_feature,
        contact_prob=contact_prob_20,
        contact_prob_close=contact_prob_close,
        binder_indices=binder_indices,
        receptor_indices=receptor_indices,
    )

    print("ifaces: ", ifaces)

    chain_res = [tokens_to_eid_cid(iface["token_ids"], entity_id, copy_id) for iface in ifaces]

    print("chain_res: ", chain_res)

    json_data = gen_constraint_json(
        chain_res,
        base_json,
        binder_ids,
        top=surface_finder.topk,
        max_dist=surface_finder.max_dist,
    )

    return json_data, ifaces, chain_res
