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

import gzip
import json
import os
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial import KDTree
from protenix.data.utils import save_atoms_to_cif


class ProteinInterfaceAnalyzer:
    """分析蛋白質結構中不同鏈之間的界面殘基，支持界面過濾和參數調控"""

    def __init__(self, gz_file_path, config: Optional[Dict] = None):
        """
        初始化分析器

        Args:
            gz_file_path: gzip壓縮的pickle文件路徑
            config: 配置字典，包含過濾條件
        """
        self.gz_file_path = gz_file_path
        self.config = config or {
            "skip_homo": True,
            "skip_ligand": True,
            "skip_dna_dna": True,
        }
        self.load_data()

    def load_data(self):
        """從gz文件加載數據，只考慮resolved的原子"""
        with gzip.open(self.gz_file_path, "rb") as f:
            data = pickle.load(f)

        self.atom_array = data["atom_array"]
        self.token_array = data["token_array"]

        # 提取原子級信息
        self.chain_ids = self.atom_array.chain_id
        self.coords = self.atom_array.coord
        self.res_ids = self.atom_array.res_id
        self.is_resolved = self.atom_array.is_resolved
        self.entity_ids = self.atom_array.entity_id_int + 1
        self.sym_ids = self.atom_array.sym_id_int + 1
        self.mol_type = self.atom_array.mol_type

        # 只保留resolved的原子
        resolved_indices = np.where(self.is_resolved)[0]
        print(f"總原子數: {len(self.atom_array)}, resolved原子數: {len(resolved_indices)}")

        # 獲取每個原子的唯一標識（只包含resolved原子）
        self.atom_info = []
        for i in resolved_indices:
            self.atom_info.append(
                {
                    "coord": self.coords[i],
                    "chain": self.chain_ids[i],
                    "res_id": self.res_ids[i],
                    "entity_id": self.entity_ids[i],
                    "sym_id": self.sym_ids[i],
                    "mol_type": self.mol_type[i],
                }
            )

        # 創建分子類型映射
        self._create_mol_type_info()
        self._mol_type2entity_poly_type()

    def _create_mol_type_info(self):
        """創建(entity_id, sym_id)到分子類型的映射"""
        self.mol_type_info = {}
        for atom in self.atom_info:
            key = f"{atom['entity_id']}_{atom['sym_id']}"
            if key not in self.mol_type_info:
                self.mol_type_info[key] = atom["mol_type"]

    def _mol_type2entity_poly_type(self):
        entity_poly_type = {}
        for key, mol_type in self.mol_type_info.items():
            if mol_type == "protein" or mol_type == "dna":
                e_id = key.split("_")[0]
                entity_poly_type[e_id] = "1"
        self.entity_poly_type = entity_poly_type

    def _get_valid_binder_receptor(self, skip_pairs: List[Tuple] = None) -> List[Dict]:
        """
        獲取有效的binder-receptor配對，根據config過濾

        Returns:
            有效的binder-receptor配對列表
        """
        if skip_pairs is None:
            skip_pairs = []

        # 收集唯一的entity_id和sym_id組合
        entity_sym_pairs = [(atom["entity_id"], atom["sym_id"]) for atom in self.atom_info]
        unique_chains = list(set(entity_sym_pairs))

        valid_pairs = []

        for binder_id in unique_chains:
            for receptor_id in unique_chains:
                # 跳過同一分子
                if binder_id == receptor_id:
                    continue

                binder_str = f"{binder_id[0]}_{binder_id[1]}"
                receptor_str = f"{receptor_id[0]}_{receptor_id[1]}"

                # 1. 跳過同源
                if self.config.get("skip_homo", True):
                    if binder_id[0] == receptor_id[0]:
                        continue

                # 2. 跳過配體
                if self.config.get("skip_ligand", True):
                    if (
                        self.mol_type_info.get(binder_str) == "ligand"
                        or self.mol_type_info.get(receptor_str) == "ligand"
                    ):
                        continue

                # 3. 跳過DNA-DNA配對
                if self.config.get("skip_dna_dna", True):
                    if self.mol_type_info.get(binder_str) == "dna" and self.mol_type_info.get(receptor_str) == "dna":
                        continue

                # 4. 跳過指定配對
                skip = False
                for first_entry, second_entry in skip_pairs:
                    first_id = f"{first_entry[0]}_{first_entry[1]}"
                    second_id = f"{second_entry[0]}_{second_entry[1]}"

                    if (binder_str == first_id and receptor_str == second_id) or (
                        binder_str == second_id and receptor_str == first_id
                    ):
                        skip = True
                        break

                if skip:
                    continue

                # 通過所有檢查，添加到有效配對
                valid_pairs.append(
                    {
                        "binder_id": binder_id,
                        "receptor_id": receptor_id,
                        "binder_str": binder_str,
                        "receptor_str": receptor_str,
                    }
                )

        return valid_pairs

    def find_interchain_contacts(self, distance_threshold: float = 5.0, skip_pairs: List[Tuple] = None) -> List[Dict]:
        """
        找到不同鏈之間距離小於閾值的原子對，只考慮有效配對

        Args:
            distance_threshold: 距離閾值(Å)
            skip_pairs: 要跳過的配對列表

        Returns:
            包含所有跨鏈接觸的列表
        """
        # 獲取有效配對
        valid_pairs = self._get_valid_binder_receptor(skip_pairs)

        if not valid_pairs:
            return []

        # 按鏈分組原子索引
        chain_atom_indices = defaultdict(list)
        chain_coords = defaultdict(list)

        for i, atom in enumerate(self.atom_info):
            chain_key = (atom["entity_id"], atom["sym_id"])
            chain_atom_indices[chain_key].append(i)
            chain_coords[chain_key].append(atom["coord"])

        contacts = []

        for pair in valid_pairs:
            binder_key = pair["binder_id"]
            receptor_key = pair["receptor_id"]

            if binder_key not in chain_coords or receptor_key not in chain_coords:
                continue

            coords_a = np.array(chain_coords[binder_key])
            coords_b = np.array(chain_coords[receptor_key])
            indices_a = chain_atom_indices[binder_key]
            indices_b = chain_atom_indices[receptor_key]

            if len(coords_a) == 0 or len(coords_b) == 0:
                continue

            # 構建KD樹
            tree_a = KDTree(coords_a)
            tree_b = KDTree(coords_b)

            # 查找接觸對
            pairs_indices = tree_a.query_ball_tree(tree_b, r=distance_threshold)

            for i, neighbors in enumerate(pairs_indices):
                atom_idx_a = indices_a[i]
                for j in neighbors:
                    atom_idx_b = indices_b[j]

                    # 計算實際距離
                    dist = np.linalg.norm(self.atom_info[atom_idx_a]["coord"] - self.atom_info[atom_idx_b]["coord"])

                    if dist < distance_threshold:
                        contacts.append(
                            {
                                "binder_id": binder_key,
                                "receptor_id": receptor_key,
                                "binder_chain": self.atom_info[atom_idx_a]["chain"],
                                "receptor_chain": self.atom_info[atom_idx_b]["chain"],
                                "atom_idx_a": atom_idx_a,
                                "atom_idx_b": atom_idx_b,
                                "distance": dist,
                            }
                        )

        return contacts

    def get_topk_interfaces(self, contacts: List[Dict], num_of_iface: int = 5, num_of_residues: int = 10) -> List[Dict]:
        """
        從接觸對中提取topk界面信息

        Args:
            contacts: 接觸對列表
            num_of_iface: 返回的界面數量
            num_of_residues: 每個界面返回的殘基數量

        Returns:
            界面信息列表
        """
        # 按binder-receptor配對分組
        interface_pairs = defaultdict(list)

        for contact in contacts:
            pair_key = (contact["binder_id"], contact["receptor_id"])
            interface_pairs[pair_key].append(contact)

        interfaces = []

        for (binder_id, receptor_id), pair_contacts in interface_pairs.items():
            if not pair_contacts:
                continue

            # 獲取binder信息
            binder_atom_idx = pair_contacts[0]["atom_idx_a"]
            binder_info = (
                int(self.atom_info[binder_atom_idx]["entity_id"]),
                int(self.atom_info[binder_atom_idx]["sym_id"]),
            )

            # 收集受體殘基
            receptor_residues = defaultdict(list)

            for contact in pair_contacts:
                receptor_atom_idx = contact["atom_idx_b"]
                atom_info = self.atom_info[receptor_atom_idx]
                res_key = (
                    atom_info["entity_id"],
                    atom_info["sym_id"],
                    atom_info["res_id"],
                )

                # 記錄距離
                receptor_residues[res_key].append(contact["distance"])

            # 計算每個殘基的最小距離
            residue_distances = []
            for (entity_id, sym_id, res_id), distances in receptor_residues.items():
                min_dist = min(distances)
                residue_distances.append(
                    {
                        "entity_id": int(entity_id),
                        "sym_id": int(sym_id),
                        "res_id": res_id,
                        "min_distance": min_dist,
                    }
                )

            # 按距離排序，取topk殘基
            residue_distances.sort(key=lambda x: x["min_distance"])
            top_residues = residue_distances[:num_of_residues]

            # 轉換格式
            receptor_tokens = [[r["entity_id"], r["sym_id"], r["res_id"]] for r in top_residues]

            interfaces.append(
                {
                    "binder_info": binder_info,
                    "receptor_tokens": receptor_tokens,
                    "contact_count": len(pair_contacts),
                    "binder_id": binder_id,
                    "receptor_id": receptor_id,
                }
            )

        # 按接觸數量排序，返回topk界面
        interfaces.sort(key=lambda x: x["contact_count"], reverse=True)
        return interfaces[:num_of_iface]

    def generate_constraint_jsons(
        self,
        base_json: Dict,
        num_of_iface: int = 5,
        num_of_residues: int = 10,
        max_dist: float = 6.5,
        distance_threshold: float = 5.0,
        skip_pairs: List[Tuple] = None,
    ) -> List[Dict]:
        """
        生成約束JSON，支持參數調控和過濾

        Args:
            base_json: 基礎JSON模板
            num_of_iface: 返回的界面數量
            num_of_residues: 每個界面包含的殘基數量
            max_dist: 最大距離閾值
            distance_threshold: 接觸距離閾值
            skip_pairs: 要跳過的配對列表

        Returns:
            約束JSON列表
        """
        # 查找所有有效接觸
        contacts = self.find_interchain_contacts(distance_threshold=distance_threshold, skip_pairs=skip_pairs)

        if not contacts:
            print(f"警告: 未找到任何有效接觸")
            return []

        # 提取topk界面
        interfaces = self.get_topk_interfaces(contacts, num_of_iface=num_of_iface, num_of_residues=num_of_residues)

        if not interfaces:
            print(f"警告: 未提取到任何界面")
            return []

        # 生成約束JSON
        return self.gen_constraint_json(base_json, interfaces, top=num_of_iface, max_dist=max_dist)

    @staticmethod
    def gen_constraint_json(base_json: Dict, interfaces: List[Dict], top: int = 5, max_dist: float = 6.5) -> List[Dict]:
        """
        生成約束JSON

        Args:
            base_json: 基礎JSON模板
            interfaces: 界面信息列表
            top: 選擇的top界面數量
            max_dist: 最大距離

        Returns:
            約束JSON列表
        """
        final = []
        for i, data in enumerate(interfaces[:top]):
            c_id = i + 1
            current_json = base_json.copy()
            current_json["name"] = f"{current_json['name']}-iface-{c_id}"

            # 轉換數據類型
            binder_chains = [int(data["binder_info"][0]), int(data["binder_info"][1])]
            contact_residues = [[int(x[0]), int(x[1]), int(x[2])] for x in data["receptor_tokens"]]

            # 構建約束
            constraint = {
                "pocket": {
                    "binder_chain": binder_chains,
                    "contact_residues": contact_residues,
                    "max_distance": max_dist,
                }
            }
            current_json["constraint"] = constraint
            final.append(current_json)

        return final

    def analyze_interfaces(
        self,
        num_of_iface: int = 10,
        num_of_residues: int = 10,
        distance_threshold: float = 5.0,
        skip_pairs: List[Tuple] = None,
    ) -> Dict:
        """
        完整分析流程

        Args:
            num_of_iface: 返回的界面數量
            distance_threshold: 接觸距離閾值
            skip_pairs: 要跳過的配對列表

        Returns:
            分析結果字典
        """
        # 查找所有有效接觸
        contacts = self.find_interchain_contacts(distance_threshold=distance_threshold, skip_pairs=skip_pairs)

        if not contacts:
            return {
                "total_contacts": 0,
                "interfaces": [],
                "chain_pairs": [],
                "valid_pairs": self._get_valid_binder_receptor(skip_pairs),
            }

        # 提取界面
        interfaces = self.get_topk_interfaces(
            contacts,
            num_of_iface=num_of_iface,
            num_of_residues=num_of_residues,  # 使用默認值
        )

        # 收集唯一的鏈對
        chain_pairs = set()
        for contact in contacts:
            pair = (
                (contact["binder_id"], contact["binder_chain"]),
                (contact["receptor_id"], contact["receptor_chain"]),
            )
            chain_pairs.add(pair)

        return {
            "total_contacts": len(contacts),
            "interfaces": interfaces,
            "chain_pairs": list(chain_pairs),
            "valid_pairs": self._get_valid_binder_receptor(skip_pairs),
        }


def mol_type2entity_poly_type(mol_type_info: dict) -> dict:
    entity_poly_type = {}
    for key, mol_type in mol_type_info.items():
        if mol_type == "protein" or mol_type == "dna":
            e_id = key.split("_")[0]
            entity_poly_type[e_id] = "1"
    return entity_poly_type


# 主程序示例
if __name__ == "__main__":
    # [Preprocess] First
    # python ~/eva/DynaHelix/scripts/prepare_training_data.py -i ./ -b bio/ -o bio/summary.csv -n 60
    bioassembly_path = "/mnt/ai4g_share/neogaryzhou/datasets/PXMeter/evaluation/supported_data/mmcif/bio"
    output_folder = "/opt/home/revoli/eva/DynaHelix/benchmark/input/gt_constraint"
    # input_json = '/mnt/ai4g_share/neogaryzhou/datasets/benchmark/validation/dsDNA/dsDNA.json'
    # input_json = '/mnt/ai4g_share/neogaryzhou/datasets/benchmark/validation/AA/AA.json'
    input_json = "/mnt/ai4g_share/neogaryzhou/datasets/benchmark/validation/PP/PP.json"
    # input_json = '/mnt/ai4g_share/neogaryzhou/datasets/benchmark/ab/ab-description-corrected.json'
    distance_threshold = 5.0
    num_ifaces = 10
    num_residues = 100
    os.makedirs(output_folder, exist_ok=True)

    # 配置參數
    config = {
        "skip_homo": True,  # 跳過同源
        "skip_ligand": True,  # 跳過配體
        "skip_dna_dna": True,  # 跳過DNA-DNA配對
    }

    # 跳過的配對列表
    skip_pairs = []  # 這裡可以添加要跳過的特定配對

    # 分析每個PDB文件
    for base_json in json.load(open(input_json)):
        pdb = base_json["name"]
        gz_f = f"{bioassembly_path}/{pdb}.pkl.gz"
        output_json_f = f"{output_folder}/{pdb}-gt.json"

        if not os.path.exists(gz_f):
            print(f"文件不存在: {gz_f}")
            continue

        analyzer = ProteinInterfaceAnalyzer(gz_f, config=config)

        # 進行界面分析
        analysis_result = analyzer.analyze_interfaces(
            num_of_iface=num_ifaces,  # 提取5個界面
            num_of_residues=num_residues,
            distance_threshold=distance_threshold,
            skip_pairs=skip_pairs,
        )

        print(f"{pdb}: 找到 {analysis_result['total_contacts']} 個跨鏈接觸")
        print(f"{pdb}: 涉及 {len(analysis_result['chain_pairs'])} 個鏈對")
        print(f"{pdb}: 提取了 {len(analysis_result['interfaces'])} 個界面")

        # 生成約束JSON
        # generate at most 5 different chains of ifaces and the 4 corresponding closest tokens as receptor
        constraint_jsons = analyzer.generate_constraint_jsons(
            base_json=base_json,
            num_of_iface=num_ifaces,
            num_of_residues=num_residues,
            max_dist=6.5,
            distance_threshold=distance_threshold,
            skip_pairs=skip_pairs,
        )

        if constraint_jsons:
            # 保存結果
            with open(output_json_f, "w") as f:
                json.dump(constraint_jsons, f, indent=2)
            print(f"{pdb}: 保存 {len(constraint_jsons)} 個約束JSON到 {output_json_f}")
        else:
            print(f"{pdb}: 未生成任何約束JSON")

        # dump the structure
        save_atoms_to_cif(
            f"{output_folder}/{pdb}_assembly1.cif",
            analyzer.atom_array,
            entity_poly_type=analyzer.entity_poly_type,
            pdb_id=pdb,
        )
