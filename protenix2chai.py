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

import argparse
import json
import os
from itertools import cycle

from scripts.tools import Mytools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Protenix-style inputs/constraints to Chai-1 format and generate shell scripts."
    )
    parser.add_argument(
        "--protenix_json",
        type=str,
        default="./example/7r58.json",
        help="Path to the Protenix input JSON file.",
    )
    parser.add_argument(
        "--constraint_json",
        type=str,
        default="./example/7r58_base_constraint_101.json",
        help="Path to the Protenix constraint JSON file.",
    )
    parser.add_argument(
        "--fasta_dir",
        type=str,
        default="./example/chai-fasta",
        help="Output directory for Chai-1 FASTA files.",
    )
    parser.add_argument(
        "--constraint_dir",
        type=str,
        default="./example/chai-constraint",
        help="Output directory for Chai-1 constraint CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/chai-1",
        help="Chai-1 inference output directory.",
    )
    parser.add_argument(
        "--chai_downloads_dir",
        type=str,
        required=True,
        help="Path to the Chai-1 model weights directory.",
    )
    parser.add_argument("--seed", type=int, default=101, help="Random seed.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to distribute tasks across.",
    )
    parser.add_argument(
        "--shell_output",
        type=str,
        default="chai_example_constraint.sh",
        help="Output shell script path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Step 1: Convert Protenix inputs to Chai-1 FASTA ----
    os.makedirs(args.fasta_dir, exist_ok=True)
    Mytools.protenix2chai(args.protenix_json, args.fasta_dir)

    # ---- Step 2: Convert Protenix constraints to Chai-1 constraints ----
    with open(args.constraint_json, "r") as f:
        constraint_inputs = json.load(f)

    os.makedirs(args.constraint_dir, exist_ok=True)
    for base_data in constraint_inputs:
        Mytools.generate_constraint_from_dyna(base_data, seed=args.seed, save_dir=args.constraint_dir)

    # ---- Step 3: Generate Chai-1 shell script ----
    gpu_cycle = cycle(range(args.num_gpus))
    all_commands = [f"export CHAI_DOWNLOADS_DIR={args.chai_downloads_dir}\n"]

    for entry in constraint_inputs:
        name = entry["name"]
        pdb = name.split("-")[0]
        constraint_path = os.path.join(args.constraint_dir, f"{name}_{args.seed}")
        if not os.path.exists(constraint_path):
            continue

        cuda = next(gpu_cycle)
        fasta_path = os.path.join(args.fasta_dir, f"{pdb}.fasta")

        configs = {
            "name": name,
            "pdb": pdb,
            "seed": args.seed,
            "base_cmd": f"CUDA_VISIBLE_DEVICES={cuda} chai-lab fold",
            "input_path": fasta_path,
            "output_path": args.output_dir,
            "msa_path": None,
            "constraint_path": constraint_path,
        }
        command = Mytools.get_chai_command(configs)

        if command is not None:
            all_commands.append(command + " &\n")

        if cuda == args.num_gpus - 1:
            all_commands.append("wait\n\n")

    all_commands.append("wait\n\n")

    with open(args.shell_output, "w") as f:
        f.writelines(all_commands)

    print()
    print("*" * 50)
    print(f"{args.shell_output} has been saved.")
    print(f"run 'sh {args.shell_output}' to inference with Chai-1")
    print("make sure the Chai-1 cache dir has been correctly set.")
    print("*" * 50)


if __name__ == "__main__":
    main()
