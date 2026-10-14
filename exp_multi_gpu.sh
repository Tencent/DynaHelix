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

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=false

if [ $# -eq 0 ]; then
  echo "Usage: $0 exp_name model_name save/input/iface/1b1/default"
  exit 1
fi

exp=$1
task=$2
valid_pdbs_path=${3:-"./default_path"}  # optional
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 runner/iface_inference.py --exp ${exp} --task ${task} --valid_pdbs_path ${valid_pdbs_path}