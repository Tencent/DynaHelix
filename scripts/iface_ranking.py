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
from sklearn.cluster import DBSCAN
from itertools import combinations


class IfaceRanking:
    def __init__(self):
        pass

    @staticmethod
    def jaccard_distance_matrix(sets):
        """计算一组集合之间的Jaccard距离矩阵"""
        n = len(sets)
        dist = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            d = 1 - inter / union if union > 0 else 1.0
            dist[i, j] = dist[j, i] = d
        return dist

    @staticmethod
    def find_eps_for_target_clusters(dist, target_clusters, min_samples=1, tol=1):
        eps_values = np.linspace(0.1, 1.0, 30)
        best_eps = None
        best_diff = float("inf")
        best_labels = None

        for eps in eps_values:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(dist)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            diff = abs(n_clusters - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_eps = eps
                best_labels = clustering.labels_

            if n_clusters == target_clusters:
                break
        return best_eps, best_labels, n_clusters

    @staticmethod
    def cluster_contacts(all_inputs_info, eps=0.5, min_samples=1):
        """对相同pdb的条目，根据contact_residues聚类"""

        # 先按pdb分组
        pdb_groups = {}
        for full_name, info in all_inputs_info.items():
            pdb_groups.setdefault(info["pdb"], []).append(full_name)

        # 对每个pdb内聚类
        for pdb, full_names in pdb_groups.items():
            if len(full_names) == 1:
                all_inputs_info[full_names[0]]["cluster_id"] = 0
                continue

            # 转为集合形式
            sets = [set(all_inputs_info[name]["contact_residues"]) for name in full_names]

            # print(sets)
            # assert 1==2

            # 计算Jaccard距离矩阵
            dist = IfaceRanking.jaccard_distance_matrix(sets)

            eps, labels, n_clusters = IfaceRanking.find_eps_for_target_clusters(dist, target_clusters=10)
            if n_clusters <= 1 or np.all(labels == -1):
                # labels = np.arange(len(dist))
                labels = [-1 for i in range(len(dist))]
                # print(f"all disjoint of {pdb}")
                # print("labels: ", labels)

            # 写入结果
            for name, label in zip(full_names, labels):
                all_inputs_info[name]["cluster_id"] = int(label)

        return all_inputs_info

    @staticmethod
    def cluster_by_constraint(inputs_json_list):

        all_inputs = {}
        seed_list = []
        for seed, input_json_path in inputs_json_list:
            with open(input_json_path, "r") as f:
                inputs = json.load(f)
                all_inputs[seed] = inputs
                seed_list.append(seed)

        all_inputs_info = {}
        for seed in seed_list:
            inputs = all_inputs[seed]

            for data in inputs:
                name = data["name"]
                full_name = f"{seed}_{name}"
                pdb = name.split("-cluster-")[0]

                if full_name not in all_inputs_info:
                    all_inputs_info[full_name] = {}

                all_inputs_info[full_name]["seed"] = seed
                all_inputs_info[full_name]["name"] = name
                all_inputs_info[full_name]["pdb"] = pdb
                all_inputs_info[full_name]["contact_residues"] = [
                    tuple(contact) for contact in data["constraint"]["pocket"]["contact_residues"]
                ]

        IfaceRanking.cluster_contacts(all_inputs_info, eps=0.4)

        return all_inputs_info


if __name__ == "__main__":
    inputs_json_list = [
        (
            101,
            f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/foldbench/AA/fb_AA_base_constraint_101.json",
        ),
        (
            102,
            f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/foldbench/AA/fb_AA_base_constraint_102.json",
        ),
        (
            103,
            f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/foldbench/AA/fb_AA_base_constraint_103.json",
        ),
        (
            104,
            f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/foldbench/AA/fb_AA_base_constraint_104.json",
        ),
        (
            105,
            f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/foldbench/AA/fb_AA_base_constraint_105.json",
        ),
    ]
    all_inputs_info = IfaceRanking.cluster_by_constraint(inputs_json_list)

    print(all_inputs_info)
