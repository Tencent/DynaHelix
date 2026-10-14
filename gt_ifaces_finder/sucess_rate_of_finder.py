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
import os
import pickle
import pandas as pd


def remap_eid(binder_info, receptor_info, pdb):
    if pdb in remapping_ref_input:
        new_binder_info = [
            remapping_ref_input[pdb].index(binder_info[0]) + 1,
            binder_info[1],
        ]
        for token in receptor_info:
            token[0] = remapping_ref_input[pdb].index(token[0]) + 1

        return new_binder_info, receptor_info
    return binder_info, receptor_info


if __name__ == "__main__":
    # find out the successful rate of auto chain finder
    gt_folder = "/opt/home/revoli/eva/DynaHelix/benchmark/input/gt_constraint/"
    remapping_ref_input = {"7vou": [3, 1, 2], "7txc": [3, 1, 2]}
    dataset = {
        "AA": "manual_base_constraint",
        "dsDNA": "55199",
        "PP": "base_constraint",
    }
    #
    all_dfs = []
    mean_dfs = []
    for data_name, model_name in dataset.items():
        print(data_name)
        for tau in range(1, 10):
            data = []
            for seed in range(101, 106):
                predict_json_f = (
                    f"/mnt/ai4g_share/neogaryzhou/datasets/benchmark/validation/{data_name}/"
                    f"vali_{data_name}_{tau}_{model_name}_{seed}.json"
                )
                # Example: /mnt/ai4g_share/neogaryzhou/datasets/benchmark/validation/dsDNA/
                # vali_dsDNA_7_55199_{seed}.json
                ###
                json_data = json.load(open(predict_json_f))
                for base_json in json_data:
                    pdb = base_json["name"].split("-")[0]
                    pocket_ct = base_json["constraint"]["pocket"]
                    binder_info = [int(x) for x in pocket_ct["binder_chain"]]
                    receptor_info = pocket_ct["contact_residues"]
                    # search
                    hit = 0
                    for gt_data in json.load(open(f"{gt_folder}/{pdb}-gt.json")):
                        gt_pocket_ct = gt_data["constraint"]["pocket"]
                        gt_binder_info, gt_receptor_info = remap_eid(
                            gt_pocket_ct["binder_chain"],
                            gt_pocket_ct["contact_residues"],
                            pdb,
                        )
                        gt_receptor_info_str = dict([(f"{x[0]}_{x[1]}_{x[2]}", 1) for x in gt_receptor_info])
                        if binder_info == gt_binder_info:
                            for token in receptor_info:
                                if f"{token[0]}_{token[1]}_{token[2]}" in gt_receptor_info_str:
                                    hit += 1
                    # make record
                    cluster_id = int(base_json["name"].split("-")[-1])
                    hit_percent = round((hit / len(receptor_info)), 3)
                    seed = int(os.path.basename(predict_json_f)[:-5].split("_")[-1])
                    data.append(
                        {
                            "dataset": data_name,
                            "json_f": predict_json_f,
                            "seed": seed,
                            "tau": tau * 0.1,
                            "pdb": pdb,
                            "hit": hit,
                            "cluster_id": cluster_id,
                            "hit%_within_iface": hit_percent,
                            "num_iface_spots": len(receptor_info),
                            "is_hit": hit > 0,
                            "name": base_json["name"],
                        }
                    )

            df = pd.DataFrame(data).round(3)
            # df = df[df['cluster_id'] <= 5]
            count_df = df.groupby("pdb").count()  # each group's n=50
            mean_df = df.groupby("pdb").mean(numeric_only=True).round(3)
            mean_df = mean_df.sort_values("is_hit", ascending=False)
            mean_df["json_f"] = ",".join(df["json_f"].unique().tolist())
            all_dfs.append(df)
            mean_dfs.append(mean_df)
    ##
    # analysis
    # for df in mean_dfs:
    #     print(df['json_f'].unique().tolist())
    #     print(f'mean of is_hit:{df['is_hit'].mean():.3f}')
    #     print(f'mean of hit%_within_iface:{df['hit%_within_iface'].mean():.3f}')

    all_dfs = pd.concat(all_dfs)
    all_dfs.to_csv(
        "/mnt/ai4g_share/revoli/Datasets/paper/DynaHelix/figure3/all_dfs.csv",
        index=False,
    )
    pass
