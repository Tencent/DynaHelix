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
import itertools
import yaml
import os
import requests
import pandas as pd


def az_generator():
    chars = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    # 1字母
    for c in chars:
        yield c
    # 2字母
    for c1 in chars:
        for c2 in chars:
            yield c1 + c2


class Mytools:
    def __inite__(self):
        pass

    @staticmethod
    def get_chai_command(configs):
        name = configs.get("name")
        pdb = configs.get("pdb")
        seed = configs.get("seed")
        base_cmd = configs.get("base_cmd")
        input_path = configs.get("input_path")
        output_path = configs.get("output_path")
        msa_path = configs.get("msa_path")
        constraint_path = configs.get("constraint_path")

        if not os.path.exists(input_path):
            print(f"{input_path} not exists.")
            return None

        params = [
            f"--seed {seed}",
            f"--constraint-path {constraint_path}",
            "--num-diffn-samples 5",
            "--num-trunk-recycles 10",
        ]

        if msa_path is not None and os.path.exists(msa_path):
            params.append(f"--msa-directory {msa_path}")
        else:
            params.append("--use-msa-server")

        full_command = " ".join([base_cmd, input_path, output_path] + params)
        return full_command

    @staticmethod
    def generate_constraint_from_dyna(base_data, seed, save_dir):
        name = base_data["name"]
        sequences = base_data["sequences"]
        seq_dict = {}
        entity_no = len(sequences)
        chain_id_iter = az_generator()

        entity_id = 1
        for i in range(entity_no):
            item = sequences[i]
            item_type = list(item.keys())[0]
            sequence = item[item_type].get("sequence", "")
            seq_count = item[item_type]["count"]

            for c in range(seq_count):
                copy_id = c + 1
                k = f"{entity_id}-{copy_id}"
                chain_id = next(chain_id_iter)
                seq_dict[k] = {"chain_id": chain_id, "sequence": sequence}

            entity_id += 1

        binder_chain = base_data["constraint"]["pocket"]["binder_chain"]

        try:
            binder_chain_info = seq_dict[f"{binder_chain[0]}-{binder_chain[1]}"]
        except Exception as e:
            print(f"{name} failed; seed: {seed}")
            print("binder_chain: ", binder_chain)
            raise e

        contact_residues = base_data["constraint"]["pocket"]["contact_residues"]
        contact_number = len(contact_residues)

        failed_names = set()
        chai_style_constraint_list = []
        for i in range(contact_number):
            contact = contact_residues[i]
            entity_id = contact[0]
            copy = contact[1]
            idx = contact[2]
            seq_idx = idx - 1

            k = f"{entity_id}-{copy}"

            try:
                tmp = {
                    "chainA": binder_chain_info["chain_id"],
                    "res_idxA": "",
                    "chainB": seq_dict[k]["chain_id"],
                    "res_idxB": f"{seq_dict[k]['sequence'][seq_idx]}{idx}",
                    "connection_type": "pocket",
                    "confidence": 1.0,
                    "min_distance_angstrom": 0.0,
                    "max_distance_angstrom": 5.5,
                    "comment": "from-dyna",
                    "restraint_id": f"restraint_{i}",
                }
            except (KeyError, IndexError, TypeError, ValueError):
                print(f"{name} failed; seed: {seed}")
                failed_names.add(name)
                continue
                # raise e

            chai_style_constraint_list.append(tmp)

        df = pd.DataFrame(chai_style_constraint_list)

        save_path = os.path.join(save_dir, f"{name}_{seed}")
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"{save_path} has been saved.")

    @staticmethod
    def protenix2boltz(protenix_json, out_dir, msa_dir=None):
        with open(protenix_json, "r") as f:
            data_list = json.load(f)

        type_map = {
            "proteinChain": "protein",
            "dnaSequence": "dna",
            "ligand": "ligand",
            "rnaSequence": "rna",
        }

        for data in data_list:
            name = data["name"]
            sequences = data["sequences"]

            yaml_data = {"sequences": []}

            # it = iter(string.ascii_uppercase)
            it = az_generator()

            for seq_info in sequences:
                seq_type = list(seq_info.keys())[0]
                boltz_type = type_map[seq_type]

                if seq_type == "ligand":
                    seq_str = seq_info[seq_type]["ligand"]
                    if "CCD" in seq_str:
                        seq_str = seq_str.split("_")[1]

                    seq_number = seq_info[seq_type]["count"]
                    ids = [next(it) for i in range(seq_number)]
                    yaml_data["sequences"].append(
                        {
                            boltz_type: {
                                "ccd": seq_str,
                                "id": ids,
                            }
                        }
                    )
                else:
                    seq_str = seq_info[seq_type]["sequence"]
                    seq_number = seq_info[seq_type]["count"]
                    ids = [next(it) for i in range(seq_number)]
                    yaml_data["sequences"].append({boltz_type: {"id": ids, "sequence": seq_str}})

            if msa_dir is not None:
                # 检查yaml_data中protein的数量
                protein_count = 0
                for seq_info_dict in yaml_data["sequences"]:
                    seq_type = list(seq_info_dict.keys())[0]
                    if seq_type == "protein":
                        protein_count = protein_count + 1

                msa_path = os.path.join(msa_dir, name)
                csvs = [f for f in os.listdir(msa_path) if f.endswith(".csv")]

                assert len(csvs) == protein_count

                for i in range(len(yaml_data["sequences"])):
                    seq_info_dict = yaml_data["sequences"][i]
                    csv_idx = i

                    seq_type = list(seq_info_dict.keys())[0]
                    if seq_type == "protein":
                        msa_csv_path = os.path.join(msa_path, f"{name}_{csv_idx}.csv")
                        assert os.path.exists(msa_csv_path), f"{msa_csv_path} not found."

                        seq_info_dict[seq_type]["msa"] = msa_csv_path
            else:
                pass

            out_path = os.path.join(out_dir, f"{name}.yaml")
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

    @staticmethod
    def protenix2chai(protenix_json, out_dir):
        with open(protenix_json, "r") as f:
            data_list = json.load(f)

        type_map = {
            "proteinChain": "protein",
            "dnaSequence": "dna",
            "ligand": "ligand",
            "rnaSequence": "rna",
        }

        for data in data_list:
            name = data["name"]
            sequences = data["sequences"]

            # 输出 FASTA 文件路径
            out_path = os.path.join(out_dir, f"{name}.fasta")

            lines = []
            # it = iter(string.ascii_uppercase)
            it = az_generator()

            for seq_info in sequences:
                seq_type = list(seq_info.keys())[0]
                boltz_type = type_map[seq_type]

                if seq_type == "ligand":
                    seq_str = seq_info[seq_type]["ligand"]
                    if "CCD" in seq_str:
                        seq_str = seq_str.split("_")[1]
                else:
                    seq_str = seq_info[seq_type]["sequence"]

                seq_number = seq_info[seq_type]["count"]

                # 生成链 ID，例如 A, B, C ...
                try:
                    ids = [next(it) for _ in range(seq_number)]
                except StopIteration as e:
                    print(f"pdb: {name}; seq_number: {seq_number}; error: {e}")

                # 写入 FASTA：每条链一条记录
                for cid in ids:
                    header = f">{boltz_type}|chain_{cid}"
                    lines.append(header)
                    lines.append(seq_str)

            # 写入文件
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                print(f"{out_path} has been saved.")

    @staticmethod
    def show_token_len(json_path):
        with open(json_path, "r") as f:
            data_list = json.load(f)

        seq_len_info = {}
        for data in data_list:
            pdb_name = data["name"]
            sequences = data["sequences"]
            total_len = 0
            for seq_info_dict in sequences:
                seq_type = list(seq_info_dict.keys())[0]

                if seq_type in ["ligand"]:
                    continue

                seq = seq_info_dict[seq_type]["sequence"]
                seq_len = len(seq)
                count = seq_info_dict[seq_type]["count"]
                seq_len = seq_len * count
                total_len = total_len + seq_len

            seq_len_info[pdb_name] = total_len

        return seq_len_info

    @staticmethod
    def download_fasta(pdb_list, save_dir):
        for pdb in pdb_list:
            pdb_id = os.path.splitext(pdb)[0].upper()
            url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"

            fasta_path = os.path.join(save_dir, f"{pdb}.fasta")
            if os.path.exists(fasta_path):
                continue

            try:
                response = requests.get(url)
                response.raise_for_status()

                with open(fasta_path, "w") as f:
                    f.write(response.text)
                print(f"Downloaded {pdb_id}.fasta")
            except (requests.exceptions.RequestException, OSError, ValueError) as e:
                print(f"Error downloading {pdb_id}: {e}")

    @staticmethod
    def download_cif(pdb_list, save_dir):
        for pdb in pdb_list:
            pdb_id = os.path.splitext(pdb)[0].lower()
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"

            cif_path = os.path.join(save_dir, f"{pdb}.cif")
            if os.path.exists(cif_path):
                continue

            try:
                response = requests.get(url)
                response.raise_for_status()

                with open(cif_path, "w") as f:
                    f.write(response.text)
                print(f"Downloaded {pdb_id}.fasta")
            except (requests.exceptions.RequestException, OSError, ValueError) as e:
                print(f"Error downloading {pdb_id}: {e}")

    @staticmethod
    def generate_assembly1_script(pdb_list, save_dir):
        all_commands = []
        for pdb in pdb_list:
            pdb_id = os.path.splitext(pdb)[0].lower()
            url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
            cmd = f"wget {url}\n"
            cmd = cmd + f"gunzip {pdb_id}-assembly1.cif.gz\n"
            all_commands.append(cmd)

        save_file_path = os.path.join(save_dir, "download.sh")
        with open(save_file_path, "w") as f:
            f.writelines(all_commands)

        print(f"{save_file_path} has been generated.")

    @staticmethod
    def get_pdb_list_from_protenix_inputs(json_path):
        with open(json_path, "r") as f:
            data_list = json.load(f)

        pdb_list = [d["name"] for d in data_list]

        return pdb_list

    @staticmethod
    def read_fasta(filename):
        sequences = []
        with open(filename, "r") as f:
            seq = ""
            desc = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq:  # 如果有已经读取的序列，则保存
                        sequences.append((desc, seq))
                        seq = ""
                    desc = line[1:]  # 去掉">"
                else:
                    seq += line
            # 最后一条序列
            if seq:
                sequences.append((desc, seq))

        seq_to_desc = {}
        for desc, seq in sequences:
            seq_to_desc[seq] = desc

        return sequences, seq_to_desc
