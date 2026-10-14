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

import numpy as np
import pandas as pd

dataset = {"AA": "_manual", "dsDNA": "", "PP": ""}

data = []
for subset, method in dataset.items():
    summary_json = f"/mnt/ai4g_share/neogaryzhou/DynaHelix_constraint/zp_exps/summary/vali_{subset}.json"
    json_data = json.load(open(summary_json))
    pdbs = list(json_data.keys())
    for pdb in pdbs:
        for tau in range(1, 10):
            for cif_path, cif_data in json_data[pdb][f"vali_{subset}_{tau}{method}"].items():
                basename = os.path.basename(cif_path)
                clusterid = int(basename.split("_")[0].split("-")[-1])
                seed = int(basename.split("_")[2])
                idockq = np.mean(cif_data["idockq"]) if len(cif_data["idockq"]) > 0 else np.mean(cif_data["dockq"])
                ranking_score = cif_data["ranking_score"]
                data.append(
                    {
                        "dataset": subset,
                        "cif_path": cif_path,
                        "pdb": pdb,
                        "tau": round(tau * 0.1, 1),
                        "idockq": idockq,
                        "ranking_score": ranking_score,
                        "seed": seed,
                        "cluster_id": clusterid,
                    }
                )
                # Example prediction path:
                # /mnt/ai4g_share/neogaryzhou/outputs/base_constraint/vali_AA_1/pred/
                # 7w71-cluster-4/seed_102/predictions/7w71-cluster-4_seed_102_sample_0.cif
df = pd.DataFrame(data)
df.to_csv(
    "/mnt/ai4g_share/revoli/Datasets/paper/DynaHelix/figure3/tau_iface_idockq_results.csv",
    index=False,
)
pass
