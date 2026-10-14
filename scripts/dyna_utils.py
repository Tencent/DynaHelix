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
import torch
from torch.utils.data import Dataset


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    Calculate the centers of the bins for a given range and number of bins.

    Args:
        min_bin (float): The minimum value of the bin range.
        max_bin (float): The maximum value of the bin range.
        no_bins (int): The number of bins.

    Returns:
        torch.Tensor: The centers of the bins.
            Shape: [no_bins]
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    bin_centers = boundaries + 0.5 * bin_width
    return bin_centers


def contact_logits2probs(distogram_logits, thres: float):
    distogram_prob = torch.nn.functional.softmax(distogram_logits, dim=-1)  # [N_token, N_token, N_bins]
    min_bin, max_bin, no_bins = 2.3125, 21.6875, 64
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    return contact_prob


class DynaHelixDataset(Dataset):
    def __init__(self, train_size=0, data_root=""):
        super().__init__()
        # dict_keys(['input_feature_dict', 's_inputs', 's', 'z', 'contact_probs'])
        self.indices = [f"{data_root}/{x}" for x in list(os.walk(data_root))[0][2]]
        self.train_size = train_size
        self.runner = 0

    def get_by_name_seed(self, name, seed):
        found = [(i, x) for i, x in enumerate(self.indices) if f"{name}-{seed}" in x]
        assert len(found) == 1, f"not found for seed {name}-{seed}"
        data = [self.__getitem__(found[0][0])]
        return data

    def get_by_name(self, name, num_samples=5):
        found = [(i, x) for i, x in enumerate(self.indices) if name in x]
        # random.shuffle(found)
        data = []
        for i, item in enumerate(found[self.runner :]):
            # print(item, end=',')
            data.append(self.__getitem__(item[0]))
            self.runner += 1
            if len(data) == num_samples:
                break
        # print('')
        return data

    def __getitem__(self, idx):
        for _ in range(10):
            data = torch.load(self.indices[idx], weights_only=False)
            #
            cpu_input_feature = data["input_feature_dict"]
            for k, v in cpu_input_feature.items():
                if type(v) == torch.Tensor:
                    cpu_input_feature[k] = v.cpu()
            #
            num_mol = len(data["input_feature_dict"]["asym_id"].unique())
            output_dict = {"name": self.indices[idx], "num_mol": num_mol, "data": data}
            return output_dict

    def __len__(self):
        if self.train_size:
            return self.train_size
        return len(self.indices)


def tokens_to_eid_cid(dna_tokens, entity_id, copy_id):
    res = []
    for token in dna_tokens:
        e_id = entity_id[token]
        c_id = copy_id[token]
        # get res_id
        position = 0
        for i in range(token + 1):
            if entity_id[i] == e_id:
                position += 1
        # record
        res.append([int(e_id), int(c_id), position])
    return res
