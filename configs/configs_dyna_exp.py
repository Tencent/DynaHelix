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

# pylint: disable=C0114
import os
from copy import deepcopy
from pathlib import Path

import yaml

ROOT_OUTPUTS_DIR = "./outputs"
DEFAULT_CKPT_DIR = "./ckpt"

dyna_configs = {
    "task": "",
    "sz_dir": "",
    "pocket_embedder": True,
    "contact_embedder": True,
    "substructure_embedder": False,
    "contact_atom_embedder": True,
    "ckpt_name": "55199",
    "exp": "exp1",
    "valid_pdbs_path": "",
}

ckpt_configs = {
    "base_constraint": {
        "model_name": "protenix_base_constraint_v0.5.0",
        "load_checkpoint_path": os.path.join(DEFAULT_CKPT_DIR, "protenix_base_constraint_v0.5.0.pt"),
    },
    "dyna_55199": {
        "model_name": "protenix_base_constraint_v0.5.0",
        "load_checkpoint_path": os.path.join(DEFAULT_CKPT_DIR, "55199_ema_0.999.pt"),
    },
    "base": {
        "model_name": "protenix_base_default_v0.5.0",
        "load_checkpoint_path": os.path.join(DEFAULT_CKPT_DIR, "protenix_base_default_v0.5.0.pt"),
    },
}

default_configs = {
    "N_cycle": 10,
    "step_scale_eta": 1.0,
    "gamma0": 0.0,
    "N_sample": 5,
    "N_step": 10,
}

# Common skip_ifaces_config shared across most surface_finder configurations
_DEFAULT_SKIP_IFACES = {
    "skip_homo": True,
    "skip_ligand": True,
    "skip_dna_dna": True,
}

# Base surface_finder template — all experiments share these defaults,
# only overriding the fields that differ.
_BASE_SURFACE_FINDER = {
    "max_dist": 6.5,
    "topk": 5,
    "M": 30,
    "m": 10,
    "contact_prob_dist": 10,
    "contact_prob_dist_close": 8,
    "tau": 0.6,
    "min_iface_prob": 0.8,
    "iface_cluster_threshold": 0.3,
    "skip_ifaces_config": _DEFAULT_SKIP_IFACES,
    "use_z": True,
}


def _make_surface_finder(**overrides):
    """Create a surface_finder config by overriding defaults."""
    sf = deepcopy(_BASE_SURFACE_FINDER)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(sf.get(key), dict):
            sf[key] = {**sf[key], **val}
        else:
            sf[key] = val
    return sf


default_surface_finder = {
    "protein-dna": _make_surface_finder(
        topk=5,
        tau=0.5,
        skip_homo=True,
        exempted_types=["ligand"],
    ),
    "protein-protein": _make_surface_finder(
        topk=5,
        tau=0.6,
        skip_homo=True,
        exempted_types=["ligand"],
    ),
    "abag": _make_surface_finder(
        topk=5,
        tau=0.80,
        skip_homo=True,
        exempted_types=["ligand"],
    ),
}

_EXP_CONFIG_PATH = Path(__file__).resolve().with_name("exp_base_configs.yaml")
_ALLOWED_BINDING_TYPES = {"abag", "protein-protein", "protein-dna"}
_ALLOWED_IFACE_VERSIONS = {"manual", "auto"}
_REQUIRED_EXP_FIELDS = {"input_json_f", "binding_type", "iface_version"}
_ALLOWED_EXP_FIELDS = _REQUIRED_EXP_FIELDS | {
    "seeds",
    "seed_range",
    "surface_finder",
    *default_configs,
}


def _normalize_exp_base_configs(configs):
    """Validate YAML experiment configs and normalize seed ranges."""
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f"Experiment config must be a non-empty mapping: {_EXP_CONFIG_PATH}")

    normalized = {}
    for exp_name, raw_exp_cfg in configs.items():
        if not isinstance(exp_name, str) or not exp_name:
            raise ValueError("Experiment names must be non-empty strings")
        if not isinstance(raw_exp_cfg, dict):
            raise TypeError(f"Experiment '{exp_name}' must be a mapping")

        unknown = set(raw_exp_cfg) - _ALLOWED_EXP_FIELDS
        if unknown:
            raise ValueError(f"Experiment '{exp_name}' has unknown fields: {sorted(unknown)}")
        missing = _REQUIRED_EXP_FIELDS - set(raw_exp_cfg)
        if missing:
            raise ValueError(f"Experiment '{exp_name}' is missing fields: {sorted(missing)}")

        exp_cfg = deepcopy(raw_exp_cfg)
        input_json_f = exp_cfg["input_json_f"]
        if not isinstance(input_json_f, str) or not input_json_f:
            raise ValueError(f"Experiment '{exp_name}'.input_json_f must be a non-empty string")
        if exp_cfg["binding_type"] not in _ALLOWED_BINDING_TYPES:
            raise ValueError(f"Experiment '{exp_name}' has unsupported binding_type: {exp_cfg['binding_type']!r}")
        if exp_cfg["iface_version"] not in _ALLOWED_IFACE_VERSIONS:
            raise ValueError(f"Experiment '{exp_name}' has unsupported iface_version: {exp_cfg['iface_version']!r}")
        if "surface_finder" in exp_cfg and not isinstance(exp_cfg["surface_finder"], dict):
            raise TypeError(f"Experiment '{exp_name}'.surface_finder must be a mapping")

        seed_fields = {"seeds", "seed_range"} & set(exp_cfg)
        if len(seed_fields) != 1:
            raise ValueError(f"Experiment '{exp_name}' must define exactly one of 'seeds' or 'seed_range'")
        if "seed_range" in exp_cfg:
            seed_range = exp_cfg.pop("seed_range")
            if not isinstance(seed_range, dict):
                raise TypeError(f"Experiment '{exp_name}'.seed_range must be a mapping")
            unknown_range_fields = set(seed_range) - {"start", "stop", "step"}
            if unknown_range_fields or not {"start", "stop"} <= set(seed_range):
                raise ValueError(f"Experiment '{exp_name}'.seed_range requires start/stop and optional step")
            start, stop, step = (
                seed_range["start"],
                seed_range["stop"],
                seed_range.get("step", 1),
            )
            if any(type(value) is not int for value in (start, stop, step)) or step == 0:
                raise ValueError(f"Experiment '{exp_name}'.seed_range values must be integers and step non-zero")
            exp_cfg["seeds"] = list(range(start, stop, step))

        seeds = exp_cfg["seeds"]
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"Experiment '{exp_name}'.seeds must be a non-empty integer list")

        for key in ("N_cycle", "N_sample", "N_step"):
            if key in exp_cfg and (type(exp_cfg[key]) is not int or exp_cfg[key] <= 0):
                raise ValueError(f"Experiment '{exp_name}'.{key} must be a positive integer")
        for key in ("step_scale_eta", "gamma0"):
            if key in exp_cfg and (isinstance(exp_cfg[key], bool) or not isinstance(exp_cfg[key], (int, float))):
                raise ValueError(f"Experiment '{exp_name}'.{key} must be a number")

        normalized[exp_name] = exp_cfg
    return normalized


def _load_exp_base_configs():
    """Load experiment definitions from the YAML file next to this module."""
    with _EXP_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        configs = yaml.safe_load(config_file)
    return _normalize_exp_base_configs(configs)


exp_base_configs = _load_exp_base_configs()


def _deep_merge(base, overrides):
    """Recursively merge experiment-level overrides into a copied base config."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def generate_exp_configs(exp):
    """Generate a complete experiment config by merging base, default, and ckpt configs.

    Args:
        exp: Experiment name key in ``exp_base_configs``.

    Returns:
        A dict containing the merged experiment configuration.
    """
    exp_cfg = exp_base_configs[exp]
    binding_type = exp_cfg["binding_type"]
    ckpt_name = {
        "abag": "base_constraint",
        "protein-protein": "base_constraint",
        "protein-dna": "dyna_55199",
    }[binding_type]
    ckpt_cfg = ckpt_configs[ckpt_name]

    # Resolve surface_finder from binding-type defaults plus experiment overrides
    surface_finder = _deep_merge(default_surface_finder[binding_type], exp_cfg.get("surface_finder", {}))

    # Merge default_configs with experiment-level overrides
    resolved = {key: exp_cfg.get(key, default_val) for key, default_val in default_configs.items()}

    # Reuse output directories across the save/input/iface stages.
    exp_base_dir = os.path.join(ROOT_OUTPUTS_DIR, ckpt_name, exp)
    sz_dir = os.path.join(ROOT_OUTPUTS_DIR, ckpt_name, "sz")
    dump_dir = os.path.join(exp_base_dir, "pred")

    os.makedirs(exp_base_dir, exist_ok=True)
    os.makedirs(sz_dir, exist_ok=True)

    return {
        "input_json_path": exp_cfg["input_json_f"],
        "ckpt_name": ckpt_name,
        "model_name": ckpt_cfg["model_name"],
        "load_checkpoint_path": ckpt_cfg["load_checkpoint_path"],
        "sz_dir": sz_dir,
        "dump_dir": dump_dir,
        "binding_type": binding_type,
        "iface_version": exp_cfg["iface_version"],
        "seeds": list(exp_cfg["seeds"]),
        "model": {
            "N_cycle": resolved["N_cycle"],
        },
        "sample_diffusion": {
            "step_scale_eta": resolved["step_scale_eta"],
            "gamma0": resolved["gamma0"],
            "N_sample": resolved["N_sample"],
            "N_step": resolved["N_step"],
        },
        "surface_finder": surface_finder,
    }
