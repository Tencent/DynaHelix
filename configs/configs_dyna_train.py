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

import os
from copy import deepcopy

from protenix.config.extend_types import ListValue, GlobalConfigValue

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.0,  # SpatialCropping
            1.0,  # SpatialInterfaceCropping
        ],
        "crop_size": -1,
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
    "constraint": {
        "enable": False,
        "fix_seed": False,  # True means use use the same contact in each evaluation.
    },
}

DATA_ROOT_DIR = os.environ.get("PROTENIX_DATA_ROOT_DIR", "/release_data/")

default_uniform_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": ListValue([0.2, 0.4, 0.4]),
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
}

dyna_train_configs = {
    "dnashape_train": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly/dnashape.train.tar"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/dna.train.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        **deepcopy(default_uniform_configs),
    },
    "dnashape_test": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly/dnashape.test.tar"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/dna.test.csv",
            ),
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        **deepcopy(default_test_configs),
    },
    "af3_prot_dsDNA": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "recentPDB_bioassembly"),
            "indices_fpath": os.path.join(DATA_ROOT_DIR, "indices/af3_prot_dsDNA.csv"),
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        **deepcopy(default_test_configs),
    },
    "jaspar_train": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "dnashape_bioassembly/jaspar.train.tar"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/jaspar.train.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        **deepcopy(default_uniform_configs),
    },
}
