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

import json
import numpy as np
import torch
from typing import List, Dict, Any
import os

from scripts.dyna_utils import DynaHelixDataset, contact_logits2probs


def tokens_to_eid_cid(dna_tokens, entity_id, copy_id):
    receptor_info = set()
    res = []
    for token in dna_tokens:
        e_id = int(entity_id[token])
        c_id = int(copy_id[token])
        # get res_id
        position = 0
        for i in range(token + 1):
            if entity_id[i] == e_id and copy_id[i] == c_id:
                position += 1
        # record
        receptor_info.add((e_id, c_id))
        res.append([e_id, c_id, position])
    return {"tokens": res, "receptor_info": receptor_info}


def gen_constraint_json(base_json, surface_patches, top=5, max_dist=6.5):
    final = []
    for i, data in enumerate(surface_patches[:top]):
        c_id = i + 1
        currnt_json = base_json.copy()
        currnt_json["name"] = f"{currnt_json['name']}-cluster-{c_id}"
        # convert datatype
        binder_chain = [str(data["binder_info"][0]), data["binder_info"][1]]
        contact_residues = [[str(x[0]), x[1], x[2]] for x in data["receptor_tokens"]]
        #
        constraint = {}
        constraint["pocket"] = {
            "binder_chain": binder_chain,
            "contact_residues": contact_residues,
            "max_distance": max_dist,
        }
        currnt_json["constraint"] = constraint
        final.append(currnt_json)
    # output
    return final


class ReceptorHotspotClustering:
    """
    蛋白贴片驱动的共接触聚类算法，支持指定配体-受体对，或自动全图搜索。
    """

    def __init__(
        self,
        entity_id: np.ndarray,
        sym_id: np.ndarray,
        mol_type_info: dict,
        skip_pairs: list = None,
        tau: float = 0.6,
        M: int = 8,
        m: int = 2,
        min_iface_prob: float = 0.3,
        iface_cluster_threshold: float = 0.3,
        skip_ifaces_config: dict = None,
    ):
        self.tau = tau
        self.M = M
        self.m = m
        self.min_iface_prob = min_iface_prob
        self.iface_cluster_threshold = iface_cluster_threshold
        self.entity_id = entity_id
        self.sym_id = sym_id
        self.mol_type_info = mol_type_info
        self._get_valid_binder_receptor(skip_ifaces_config, skip_pairs)

    def _get_valid_binder_receptor(self, config, skip_pairs):
        # 遍历查找存在的 (e, s) 组合
        # 注意：这里假设输入的 entity_id/sym_id 已经是 1-based 或者 0-based 统一的
        # 为了稳健，我们通过遍历数组来收集
        combined_ids = np.stack([self.entity_id, self.sym_id], axis=1)
        self.unique_chains = list(np.unique(combined_ids, axis=0))  # array of [e_id, s_id]
        ##
        valid_binder_receptor_indices = {}
        for binder_id in self.unique_chains:
            valid_receptors = []
            for receptor_id in self.unique_chains:
                binder_pair_str = f"{binder_id[0]}_{binder_id[1]}"
                receptor_pair_str = f"{receptor_id[0]}_{receptor_id[1]}"
                # skip the same molecule
                if binder_pair_str == receptor_pair_str:
                    continue
                # 1. skip homology
                if config["skip_homo"]:
                    if binder_id[0] == receptor_id[0]:
                        continue
                # 2. skip any ligand
                if config["skip_ligand"]:
                    if (
                        self.mol_type_info[binder_pair_str] == "ligand"
                        or self.mol_type_info[receptor_pair_str] == "ligand"
                    ):
                        continue
                # 3. skip dsDNA pair
                if config["skip_dna_dna"]:
                    if self.mol_type_info[binder_pair_str] == "dna" and self.mol_type_info[receptor_pair_str] == "dna":
                        continue
                # 4. skip designed pair
                check = False
                for first_entry, second_entry in skip_pairs:
                    # 預計算配對雙方的 ID
                    first_id = f"{first_entry[0]}_{first_entry[1]}"
                    second_id = f"{second_entry[0]}_{second_entry[1]}"
                    # 檢查是否匹配任一方向
                    if (binder_pair_str == first_id and second_id == receptor_pair_str) or (
                        binder_pair_str == second_id and first_id in receptor_pair_str
                    ):
                        check = True
                if check:
                    continue
                # last: hit
                valid_receptors.append(receptor_id)
            # get receptor indices
            if valid_receptors:
                mask = np.zeros_like(self.entity_id)
                for receptor_id in valid_receptors:
                    mask |= (self.entity_id == receptor_id[0]) & (self.sym_id == receptor_id[1])
                r_indices = np.where(mask)[0]
                b_indiecs = np.where((self.entity_id == binder_id[0]) & (self.sym_id == binder_id[1]))[0]
                #
                valid_binder_receptor_indices[binder_pair_str] = {
                    "binder_id": binder_id,
                    "binder_indices": b_indiecs,
                    "r_indices": r_indices,
                    "valid_receptors": valid_receptors,
                }

        print(f"Valid binders, n:{len(valid_binder_receptor_indices)}")
        self.valid_binder_receptor_indices = valid_binder_receptor_indices
        pass

    def get_receptor_hotspots(
        self,
        score_feature: np.ndarray,
        contact_prob: np.ndarray,
        contact_prob_close: np.ndarray,
        binder_indices: List[int],
        receptor_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        原始的核心聚类逻辑 (保持不变，用于处理特定的一对索引)
        """
        # 1. 提取特征子矩阵
        if len(score_feature.shape) == 2:
            C = score_feature[binder_indices][:, receptor_indices].T
            s = np.max(C, axis=1)
        else:
            # z feature [N, N, d] -> Slice -> [N_binder, N_receptor, d]
            C = score_feature[binder_indices][:, receptor_indices].transpose(1, 0, 2)
            # s: [receptor_len], 衡量Receptor上每个残基与Binder的最强特征强度
            s = np.max(np.abs(C), axis=-1).max(-1)

        # 2. 计算Receptor残基间的相似度 (基于它们与Binder的交互模式)
        C_flat = C.reshape(C.shape[0], -1)
        norms = np.linalg.norm(C_flat, axis=1, keepdims=True) + 1e-8
        V = C_flat / norms
        S = V @ V.T  # Cosine Similarity [receptor_len, receptor_len]

        # 3. 提取受体内部距离信息
        receptor_len = len(receptor_indices)
        receptor_contact = contact_prob[receptor_indices][:, receptor_indices]

        # 4. 聚类流程
        ifaces = self._first_level_clustering(s, S, receptor_contact, receptor_len)
        results = self._score_and_select_tokens(ifaces, s, receptor_indices)
        iface_clusters = self._cluster_ifaces(results, contact_prob_close)
        final_results = self._select_best_ifaces_per_cluster(iface_clusters)

        return final_results

    def auto_detect_interfaces(
        self,
        z_feature: np.ndarray,
        contact_prob: np.ndarray,
        contact_prob_close: np.ndarray,
        pair_contact_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        自动遍历所有链对，基于Z feature提取高分界面。

        参数:
            z_feature: [N, N, 128] 原生特征矩阵
            contact_prob: [N, N] 用于距离约束的接触概率 (通常是 20A)
            contact_prob_close: [N, N] 用于二次聚类的近距离接触概率
            pair_contact_threshold: 链间平均接触概率阈值，低于此值认为无交互，跳过计算。

        返回:
            List[Dict]: 包含 binder, receptor, tokens, score 的列表
        """
        discovered_interfaces = []
        # 2. 全对全遍历 (All-vs-All)
        for binder_id_str, pair_info in self.valid_binder_receptor_indices.items():
            b_id = pair_info["binder_id"]
            b_indices = pair_info["binder_indices"]
            r_indices = pair_info["r_indices"]

            if len(b_indices) == 0 or len(r_indices) == 0:
                continue

            # 3. 快速预筛选：检查两链是否有物理接触
            # 提取两块区域的接触概率矩阵子块
            # sub_contact = contact_prob[b_indices][:, r_indices]
            # 这里可以使用 mean 或者 max top-k mean
            # 如果两链完全不挨着，平均接触概率会非常低（接近0）
            # interaction_strength = np.mean(sub_contact)

            # 如果交互太弱，直接跳过，节省计算资源
            # if interaction_strength < pair_contact_threshold:
            #     continue

            # 4. 执行核心聚类算法
            # 注意：这里 score_feature 传入 z_feature
            hotspots = self.get_receptor_hotspots(
                score_feature=z_feature,
                contact_prob=contact_prob,
                contact_prob_close=contact_prob_close,
                binder_indices=b_indices,
                receptor_indices=r_indices,
            )

            # 5. 格式化结果
            # 每个 hotspot 代表一个潜在的 binding site
            for iface in hotspots:
                # 构造输出格式
                res_obj = {
                    "binder": (int(b_id[0]), int(b_id[1])),  # (entity, copy)
                    "receptor": (
                        iface["token_ids"]  # 全局 token indices
                    ),
                    "score": float(iface["score"]),
                    "mean_score": float(iface["mean_score"]),
                    "size": int(iface["iface_size"]),
                }
                discovered_interfaces.append(res_obj)

        # 6. 全局排序：将所有链对中发现的界面按分数排序
        # 这样最显著的界面（无论在哪些链之间）会排在前面
        discovered_interfaces.sort(key=lambda x: x["score"], reverse=True)

        return discovered_interfaces

    # ... (保留原有的 _first_level_clustering, _score_and_select_tokens 等私有方法) ...
    def _first_level_clustering(self, s, S, receptor_contact, receptor_len):
        ifaces = []
        unassigned = set(range(receptor_len))
        sorted_indices = np.argsort(s)[::-1]
        for idx in sorted_indices:
            if idx not in unassigned:
                continue
            seed = idx
            iface = {seed}
            unassigned.remove(seed)
            for j in list(unassigned):
                if S[seed, j] >= self.tau:
                    valid = True
                    if receptor_contact[j, seed] < self.min_iface_prob:
                        valid = False
                    if valid:
                        iface.add(j)
                        unassigned.remove(j)
            ifaces.append(iface)
            if len(ifaces) >= self.M:
                break
        return ifaces

    def _score_and_select_tokens(self, ifaces, s, receptor_indices):
        results = []
        for iface in ifaces:
            iface_score = max(s[j] for j in iface)
            iface_mean = np.mean([s[j] for j in iface])
            iface_tokens = sorted(iface, key=lambda j: s[j], reverse=True)[: self.m]
            token_ids = [receptor_indices[j] for j in iface_tokens]
            results.append(
                {
                    "score": iface_score,
                    "mean_score": iface_mean,
                    "token_ids": token_ids,
                    "iface_size": len(iface),
                    "center_token": token_ids[0],
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _cluster_ifaces(self, ifaces, contact_prob_close):
        if not ifaces:
            return []
        n_ifaces = len(ifaces)
        connection_strength = np.zeros((n_ifaces, n_ifaces))
        for i in range(n_ifaces):
            for j in range(i + 1, n_ifaces):
                strength = self._calculate_iface_connection(ifaces[i], ifaces[j], contact_prob_close)
                connection_strength[i, j] = connection_strength[j, i] = strength
        clusters = []
        unassigned = set(range(n_ifaces))
        while unassigned:
            seed = max(unassigned, key=lambda idx: ifaces[idx]["score"])
            cluster = {seed}
            unassigned.remove(seed)
            for j in list(unassigned):
                if connection_strength[seed, j] >= self.iface_cluster_threshold:
                    cluster.add(j)
                    unassigned.remove(j)
            clusters.append([ifaces[i] for i in cluster])
        return clusters

    def _calculate_iface_connection(self, iface1, iface2, contact_prob_close):
        tokens1 = iface1["token_ids"]
        tokens2 = iface2["token_ids"]
        strengths = [contact_prob_close[t1, t2] for t1 in tokens1 for t2 in tokens2]
        return np.mean(strengths) if strengths else 0.0

    def _select_best_ifaces_per_cluster(self, iface_clusters):
        final_results = []
        for cluster in iface_clusters:
            if cluster:
                best_iface = max(cluster, key=lambda x: x["score"])
                final_results.append(best_iface)
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results


def check_skip(receptor_info, b_eid, b_cid, skip_pairs, mol_type_info, config):
    """
    mol_type_info: {'2_1': 'protein', '1_1': 'ligand'}
    """
    receptor_set = set([f"{x[0]}_{x[1]}" for x in receptor_info])
    binder_pair = f"{b_eid}_{b_cid}"
    # 1. skip homology
    if config["skip_homo"]:
        check = [1 for info in receptor_set if b_eid == int(info.split("_")[0])]
        if sum(check) != 0:
            return True
    # 2. skip any pair contains ligand
    if config["skip_ligand"]:
        if mol_type_info[binder_pair] == "ligand":
            return True
        check = [1 for receptor_pair in receptor_set if mol_type_info[receptor_pair] == "ligand"]
        # receptor are all ligand's atom
        if sum(check) == len(receptor_set):
            return True
    # 3. skip dsDNA pair
    if config["skip_dna_dna"]:
        if mol_type_info[binder_pair] == "dna":
            check = [1 for receptor_pair in receptor_set if mol_type_info[receptor_pair] == "dna"]
            # receptor are all another dna's residue
            if sum(check) == len(receptor_set):
                return True
    # 4. skip designed pair
    for first_entry, second_entry in skip_pairs:
        # 預計算配對雙方的 ID
        first_id = f"{first_entry[0]}_{first_entry[1]}"
        second_id = f"{second_entry[0]}_{second_entry[1]}"

        # 檢查是否匹配任一方向
        if (binder_pair == first_id and second_id in receptor_set) or (
            binder_pair == second_id and first_id in receptor_set
        ):
            return True

    return False


def get_mol_type_info(input_feature_dict):
    info = {}
    entity_id, sym_id = (
        input_feature_dict["entity_id"] + 1,
        input_feature_dict["sym_id"] + 1,
    )
    n_token = len(entity_id)
    for i in range(n_token):
        key = f"{entity_id[i]}_{sym_id[i]}"
        if key not in info:
            centre_atom_index = input_feature_dict["frame_atom_index"][i][0]
            if input_feature_dict["is_protein"][centre_atom_index]:
                info[key] = "protein"
            elif input_feature_dict["is_ligand"][centre_atom_index]:
                info[key] = "ligand"
            elif input_feature_dict["is_dna"][centre_atom_index]:
                info[key] = "dna"
            elif input_feature_dict["is_rna"][centre_atom_index]:
                info[key] = "rna"
    return info


def run(data, skip_pairs=None, top=5, config=None) -> list:
    if skip_pairs is None:
        skip_pairs = []
    if config is None:
        raise ValueError("config is required")
    # 1. 准备数据 (Tensor -> Numpy)
    first_entry = data[0]["data"]
    entity_id = (first_entry["input_feature_dict"]["entity_id"] + 1).numpy()
    copy_id = (first_entry["input_feature_dict"]["sym_id"] + 1).numpy()
    ###
    # features
    all_z = [item["data"]["z"].unsqueeze(0) for item in data]
    all_contact_logits = [item["data"]["contact_logits"].unsqueeze(0) for item in data]
    # z
    z = torch.cat(all_z).mean(0).float()  # [N, N, 128]
    z = z.numpy()
    # contact
    contact_logits = torch.cat(all_contact_logits).mean(0).float()  # [N, N, 64]
    contact_prob_20 = contact_logits2probs(contact_logits, 20).numpy()  # [N, N]
    contact_prob_close = contact_logits2probs(contact_logits, config.contact_prob_dist_close).numpy()  # [N, N]
    # mol_types
    mol_type_info = get_mol_type_info(first_entry["input_feature_dict"])

    # print("config.tau: ", config.tau)
    # print("config.M: ", config.M)
    # print("config.m: ", config.m)
    # print("config.skip_ifaces_config: ", config.skip_ifaces_config)

    # 2. 初始化聚类器
    # 注意：Z feature 通常比较稠密，可能需要调整 min_iface_prob 或 tau
    clustering = ReceptorHotspotClustering(
        tau=config.tau,  # 相似度阈值
        M=config.M,  # 最大初始簇数量
        m=config.m,  # 每个簇保留的token数
        min_iface_prob=0.8,  # 距离约束
        iface_cluster_threshold=0.3,
        entity_id=entity_id,
        sym_id=copy_id,
        mol_type_info=mol_type_info,
        skip_pairs=skip_pairs,
        skip_ifaces_config=config.skip_ifaces_config,
    )

    # print("clustering: ", clustering)

    # 3. 自动检测 (只输入 Z feature 作为 score source)
    if config.use_z:
        print(f"Running auto detection using Z features...")
        auto_results = clustering.auto_detect_interfaces(
            z_feature=z,  # 直接传入 Z
            contact_prob=contact_prob_20,  # 用于几何距离过滤
            contact_prob_close=contact_prob_close,  # 用于二次聚类
        )
    else:
        print(f"Running auto detection using contact_prob_close features...")
        auto_results = clustering.auto_detect_interfaces(
            z_feature=contact_prob_close,  # 直接传入 Z
            contact_prob=contact_prob_20,  # 用于几何距离过滤
            contact_prob_close=contact_prob_close,  # 用于二次聚类
        )

    # 4. 打印结果
    print(f"Total discovered interfaces: {len(auto_results)}")
    print(f"Top {top} Interfaces:")
    surface_patches = []
    for i, res in enumerate(auto_results):
        b_eid, b_cid = res["binder"]
        tokens = res["receptor"]

        # 将 token indices 转换为可读格式 (E-C-ResId)
        readable_info = tokens_to_eid_cid(tokens, entity_id, copy_id)
        readable_tokens, receptor_info = (
            readable_info["tokens"],
            readable_info["receptor_info"],
        )

        print(f"Rank {i + 1} | Score: {res['score']:.4f}")
        print(f"  Binder  : Entity {b_eid}, Copy {b_cid}")
        print(f"  Receptor Tokens: {readable_tokens}")
        print("-" * 50)
        surface_patches.append({"receptor_tokens": readable_tokens, "binder_info": res["binder"]})
        if len(surface_patches) >= top:
            break
    return surface_patches


def generate_iface_inputs_at(base_json, name, seed, data_root, surface_finder, binding_type=None, num_samples=1):
    print("data_root: ", data_root)
    train_set = DynaHelixDataset(
        data_root=data_root  # dyna, constraint
    )

    data = train_set.get_by_name_seed(name, seed=seed)

    surface_patches = run(
        data,
        skip_pairs=base_json.get("skip_pairs", []),
        top=surface_finder.topk,
        config=surface_finder,
    )

    # print("surface_patches: ", surface_patches)

    json_data = gen_constraint_json(
        base_json,
        surface_patches,
        top=surface_finder.topk,
        max_dist=surface_finder.max_dist,
    )

    return json_data, None, None
