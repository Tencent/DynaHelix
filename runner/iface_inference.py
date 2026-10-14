# Copyright 2026 Tencent and/or its affiliates.
# Copyright 2024 ByteDance and/or its affiliates.
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
import logging
import os
import pickle
import traceback
import urllib.request
from contextlib import nullcontext
from multiprocessing import Pool
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_dyna_exp import dyna_configs, generate_exp_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.protenix import Protenix
from protenix.model.protenix_1b1 import Protenix_1b1
from protenix.model.protenix_save import Protenix_save
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from protenix.web_service.dependency_url import URL
from runner.dumper import DataDumper
from scripts.iface import generate_iface_inputs
from scripts.iface_at import generate_iface_inputs_at

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precision mapping (shared by predict / save_sz)
# ---------------------------------------------------------------------------
_PRECISION_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# ---------------------------------------------------------------------------
# Model class registry for init_model dispatch
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "1b1": Protenix_1b1,
    "save": Protenix_save,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_cifs_from_path(cif_dir_path):
    """Recursively collect all .cif files under *cif_dir_path*."""
    return [os.path.join(root, f) for root, _, files in os.walk(cif_dir_path) for f in files if f.endswith(".cif")]


def cif_path_is_not_empty(cif_dir_path):
    return bool(get_cifs_from_path(cif_dir_path))


def _safe_empty_cache():
    """Empty the CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_sample_info(rank, sample_index, num_data, sample_name, data):
    logger.info(
        "[Rank %s (%s/%s)] %s: N_asym %s, N_token %s, N_atom %s, N_msa %s",
        rank,
        sample_index + 1,
        num_data,
        sample_name,
        data["N_asym"].item(),
        data["N_token"].item(),
        data["N_atom"].item(),
        data["N_msa"].item(),
    )


def _write_error(error_dir, filename, message):
    with open(opjoin(error_dir, filename), "a") as f:
        f.write(message)


def _load_dataloader(configs, error_dir):
    """Attempt to create the inference dataloader; return None on failure."""
    try:
        return get_inference_dataloader(configs=configs)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.error(error_message)
        _write_error(error_dir, "error.txt", error_message)
        return None


def _load_valid_pdbs(path):
    """Load a list of valid PDB names from a JSON file, or return None."""
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["valid_pdbs"]
    return None


# ---------------------------------------------------------------------------
# InferenceRunner
# ---------------------------------------------------------------------------
class InferenceRunner:
    """Manages model loading, inference, and result dumping."""

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self._init_env()
        self._init_basics()
        self._init_model()
        self._load_checkpoint()
        self._init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    # -- Environment ----------------------------------------------------------
    def _init_env(self) -> None:
        self._log(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                "LOCAL_RANK: %s - CUDA_VISIBLE_DEVICES: [%s]",
                DIST_WRAPPER.local_rank,
                devices,
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self._log(f"CUTLASS_PATH: {env}")
            assert env is not None, (
                "If using ds4sci, set `CUTLASS_PATH` environment variable according to "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
            logging.info("The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time.")

        if os.getenv("LAYERNORM_TYPE") == "fast_layernorm":
            logging.info("The kernels will be compiled when fast_layernorm is called for the first time.")

        logging.info("Finished init ENV.")

    # -- Basics ---------------------------------------------------------------
    def _init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    # -- Model ----------------------------------------------------------------
    def _init_model(self) -> None:
        model_cls = _MODEL_REGISTRY.get(self.configs.task, Protenix)
        self.model = model_cls(self.configs).to(self.device)

    def _load_checkpoint(self) -> None:
        checkpoint_path = (
            self.configs.load_checkpoint_path or f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: [{checkpoint_path}]")

        self._log(f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}")
        checkpoint = torch.load(checkpoint_path, self.device, weights_only=False)

        state_dict = checkpoint["model"]
        sample_key = next(iter(state_dict))
        self._log(f"Sampled key: {sample_key}")

        # Strip DDP "module." prefix if present
        if sample_key.startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict=state_dict, strict=self.configs.load_strict)
        self.model.eval()
        self._log("Finished loading checkpoint.")

    # -- Dumper ---------------------------------------------------------------
    def _init_dumper(self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # -- Inference ------------------------------------------------------------
    def _get_amp_context(self):
        """Return an AMP autocast context (or nullcontext on CPU)."""
        precision = _PRECISION_MAP[self.configs.dtype]
        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=precision)
        return nullcontext()

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """Run forward inference and return predictions."""
        data["input_feature_dict"]["iface"] = data["iface"]
        data = to_device(data, self.device)
        with self._get_amp_context():
            data["input_feature_dict"]["name"] = f"{data['sample_name']}-{data['seed']}"
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )
        return prediction

    @torch.no_grad()
    def save_sz(self, data: Mapping[str, Mapping[str, Any]]) -> Any:
        """Run forward pass in 'save' mode and return status."""
        data["input_feature_dict"]["iface"] = data["iface"]
        data = to_device(data, self.device)
        with self._get_amp_context():
            data["input_feature_dict"]["name"] = f"{data['sample_name']}-{data['seed']}"
            status = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )
        return status

    def turn_on_dropout(self):
        """Switch specific modules to train mode for MC-dropout style inference."""
        module_list = ["msa_module", "pairformer_stack"]
        logger.info("Turn on dropout of %s", module_list)
        for name, _param in self.model.named_parameters():
            module_name = name.split(".")[0] if DIST_WRAPPER.world_size == 1 else name.split(".")[1]
            if module_name in module_list:
                self.model.get_submodule(module_name).train()
        self.model.training = True

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs

    def _log(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


# ---------------------------------------------------------------------------
# Inference configuration helpers
# ---------------------------------------------------------------------------
def update_inference_configs(configs: Any, n_token: int):
    """Dynamically adjust AMP settings based on token count to avoid OOM."""
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


# ---------------------------------------------------------------------------
# Unified inference loop
# ---------------------------------------------------------------------------
def _run_inference_loop(
    runner: InferenceRunner,
    configs: Any,
    dataloader,
    seed: int,
    *,
    skip_existing: bool = False,
    dump_results: bool = True,
    valid_pdbs: list | None = None,
    sz_dir: str | None = None,
):
    """Core batch-processing loop shared by all inference modes.

    Args:
        runner: The InferenceRunner instance.
        configs: Global config object.
        dataloader: PyTorch DataLoader yielding batches.
        seed: Random seed for this run.
        skip_existing: If True, skip samples whose output directory already contains .cif files.
        dump_results: If True, call runner.dumper.dump() after prediction.
        valid_pdbs: Optional whitelist of PDB names.
        sz_dir: If set (save-sz mode), skip when .pt already exists, and use runner.predict instead of dump.
    """
    num_data = len(dataloader.dataset)
    seed_everything(seed=seed, deterministic=configs.deterministic)

    for batch in dataloader:
        data = atom_array = sample_name = None
        try:
            data, atom_array, data_error_message = batch[0]
            data["seed"] = seed
            sample_name = data["sample_name"]

            # --- Filter by valid_pdbs whitelist ---
            if valid_pdbs is not None and sample_name not in valid_pdbs:
                continue

            # --- Skip if sz output already exists ---
            if sz_dir is not None:
                sz_path = os.path.join(sz_dir, f"{sample_name}-{seed}.pt")
                if os.path.exists(sz_path):
                    logger.info("Skip existing sz: %s", sz_path)
                    continue

            # --- Skip if prediction CIF already exists ---
            if skip_existing:
                sample_dump_dir = os.path.join(configs.dump_dir, sample_name, f"seed_{seed}")
                if os.path.exists(sample_dump_dir) and cif_path_is_not_empty(sample_dump_dir):
                    logger.info("Skip existing output: %s", sample_dump_dir)
                    continue

            # --- Handle data errors ---
            if data_error_message:
                logger.info(data_error_message)
                _write_error(runner.error_dir, f"{sample_name}.txt", data_error_message)
                continue

            # --- Log sample info ---
            _log_sample_info(DIST_WRAPPER.rank, data["sample_index"], num_data, sample_name, data)

            # --- Update configs & run prediction ---
            new_configs = update_inference_configs(configs, data["N_token"].item())
            runner.update_model_configs(new_configs)
            prediction = runner.predict(data)

            # --- Dump results ---
            if dump_results:
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

            logger.info(
                "[Rank %s] %s succeeded. Results saved to %s",
                DIST_WRAPPER.rank,
                sample_name,
                configs.dump_dir,
            )
            _safe_empty_cache()

        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as e:
            name = sample_name or (data["sample_name"] if data else "unknown")
            error_message = f"[Rank {DIST_WRAPPER.rank}]{name} {e}:\n{traceback.format_exc()}"
            logger.error(error_message)
            _write_error(runner.error_dir, f"{name}.txt", error_message)
            _safe_empty_cache()


# ---------------------------------------------------------------------------
# Task-specific entry points (thin wrappers around _run_inference_loop)
# ---------------------------------------------------------------------------
def save_sz_task(runner: InferenceRunner, configs: Any) -> None:
    """Save s/z representations for each sample."""
    valid_pdbs = _load_valid_pdbs(configs.valid_pdbs_path)
    dataloader = _load_dataloader(configs, runner.error_dir)
    if dataloader is None:
        return
    for seed in configs.seeds:
        _run_inference_loop(
            runner,
            configs,
            dataloader,
            seed,
            dump_results=False,
            valid_pdbs=valid_pdbs,
            sz_dir=configs.sz_dir,
        )


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """Iface inference: load per-seed JSON and predict with dump."""
    input_json_path = configs.input_json_path
    input_dir_name = os.path.dirname(input_json_path)

    for seed in configs.seeds:
        exp, model = configs.exp, configs.ckpt_name
        configs.input_json_path = os.path.join(input_dir_name, f"{exp}_{model}_{seed}.json")
        logger.info("Loading data from %s", configs.input_json_path)

        dataloader = _load_dataloader(configs, runner.error_dir)
        if dataloader is None:
            return
        _run_inference_loop(runner, configs, dataloader, seed, skip_existing=True, dump_results=True)


def infer_predict_1b1(runner: InferenceRunner, configs: Any) -> None:
    """1-by-1 inference: single dataloader, skip existing, dump results."""
    dataloader = _load_dataloader(configs, runner.error_dir)
    if dataloader is None:
        return
    for seed in configs.seeds:
        _run_inference_loop(runner, configs, dataloader, seed, skip_existing=True, dump_results=True)


def infer_predict_default(runner: InferenceRunner, configs: Any) -> None:
    """Default inference: single dataloader, always run, dump results."""
    dataloader = _load_dataloader(configs, runner.error_dir)
    if dataloader is None:
        return
    for seed in configs.seeds:
        _run_inference_loop(runner, configs, dataloader, seed, dump_results=True)


# ---------------------------------------------------------------------------
# Runner factories
# ---------------------------------------------------------------------------
def run_inference(configs: Any) -> None:
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def run_inference_1b1(configs: Any) -> None:
    runner = InferenceRunner(configs)
    infer_predict_1b1(runner, configs)


def run_inference_default(configs: Any) -> None:
    runner = InferenceRunner(configs)
    infer_predict_default(runner, configs)


def run_save_sz(configs: Any) -> None:
    configs.task = "save"
    runner = InferenceRunner(configs)
    save_sz_task(runner, configs)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_infercence_cache(configs: Any) -> None:
    """Download CCD cache files and model checkpoints if not present."""

    def _progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        filled = int(30 * percent // 100)
        bar = "=" * filled + "-" * (30 - filled)
        print(f"\r[{bar}] {percent:.1f}%", end="", flush=True)
        if downloaded >= total_size:
            print()

    def _download(tos_url, dest, check_weight=True):
        urllib.request.urlretrieve(tos_url, dest, reporthook=_progress_callback)
        if check_weight:
            try:
                ckpt = torch.load(dest)
                del ckpt
            except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError):
                os.remove(dest)
                raise RuntimeError(
                    f"Download model checkpoint failed. Please download manually:\n  wget {tos_url} -O {dest}"
                )

    # CCD caches
    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name mismatch: `{tos_url}` vs `{cur_cache_fpath}`"
            )
            logger.info("Downloading %s from %s to %s", cache_name, tos_url, cur_cache_fpath)
            _download(tos_url, cur_cache_fpath, check_weight=False)

    # ESM checkpoints
    checkpoint_dir = configs.load_checkpoint_dir
    _ESM_CKPTS = {
        "esm": [
            ("esm2_t36_3B_UR50D", "esm2_t36_3B_UR50D.pt"),
            (
                "esm2_t36_3B_UR50D-contact-regression",
                "esm2_t36_3B_UR50D-contact-regression.pt",
            ),
        ],
        "ism": [
            ("esm2_t36_3B_UR50D_ism", "esm2_t36_3B_UR50D_ism.pt"),
            (
                "esm2_t36_3B_UR50D_ism-contact-regression",
                "esm2_t36_3B_UR50D_ism-contact-regression.pt",
            ),
        ],
    }
    for keyword, ckpt_list in _ESM_CKPTS.items():
        if keyword in configs.model_name:
            for url_key, filename in ckpt_list:
                path = os.path.join(checkpoint_dir, filename)
                if not opexists(path):
                    tos_url = URL[url_key]
                    logger.info("Downloading %s to %s", tos_url, path)
                    _download(tos_url, path)


# ---------------------------------------------------------------------------
# Multiprocessing worker for iface input construction
# ---------------------------------------------------------------------------
def _worker_func(args):
    """Worker for parallel iface input generation (runs in subprocess)."""
    (
        pdb_name,
        base_json,
        seed,
        data_root,
        surface_finder,
        binding_type,
        iface_version,
        valid_pdbs_path,
    ) = args

    valid_pdbs = _load_valid_pdbs(valid_pdbs_path)
    if valid_pdbs is not None and pdb_name not in valid_pdbs:
        return {"pdb": pdb_name, "iface_inputs": [], "error": None}

    logger.info(
        "pid: %s; pdb_name: %s; seed: %s; iface_version: %s",
        os.getpid(),
        pdb_name,
        seed,
        iface_version,
    )

    iface_func = generate_iface_inputs if iface_version == "manual" else generate_iface_inputs_at

    try:
        iface_inputs, _, _ = iface_func(
            base_json,
            pdb_name,
            seed,
            data_root,
            surface_finder,
            binding_type=binding_type,
            num_samples=1,
        )
        if iface_inputs is None:
            iface_inputs = []
        elif not isinstance(iface_inputs, list):
            iface_inputs = [iface_inputs]
        return {"pdb": pdb_name, "iface_inputs": iface_inputs, "error": None}

    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as e:
        error_message = f"pdb: {pdb_name}, seed: {seed}, binding_type: {binding_type}, error: {e}"
        logger.error(error_message)
        return {"pdb": pdb_name, "iface_inputs": [], "error": error_message}


def construct_inputs(configs):
    """Build per-seed iface input JSONs via multiprocessing."""
    exp, model = configs.exp, configs.ckpt_name
    input_json_path = configs.input_json_path
    dirname = os.path.dirname(input_json_path)
    error_dir = os.path.join(configs.dump_dir, "ERR")
    os.makedirs(error_dir, exist_ok=True)

    iface_version = configs.iface_version if configs.iface_version in ("manual", "auto") else "manual"

    with open(input_json_path, "r") as f:
        base_inputs = json.load(f)

    all_pdbs = []
    base_inputs_dict = {}
    for entry in base_inputs:
        name = entry["name"]
        all_pdbs.append(name)
        base_inputs_dict[name] = entry

    for seed in configs.seeds:
        iface_json_path = os.path.join(dirname, f"{exp}_{model}_{seed}.json")
        if os.path.exists(iface_json_path):
            logger.info("Iface JSON already exists, skip: %s", iface_json_path)
            continue

        task_args = [
            (
                pdb,
                base_inputs_dict[pdb],
                seed,
                configs.sz_dir,
                configs.surface_finder,
                configs.binding_type,
                iface_version,
                configs.valid_pdbs_path,
            )
            for pdb in all_pdbs
        ]

        with Pool(processes=min(100, len(task_args))) as pool:
            results = pool.map(_worker_func, task_args)

        all_inputs = []
        error_path = os.path.join(error_dir, f"missing_pdb_seed_{seed}.txt")
        for res in results:
            if res["error"]:
                _write_error(error_dir, f"missing_pdb_seed_{seed}.txt", res["error"] + "\n")
            else:
                all_inputs.extend(res["iface_inputs"])

        with open(iface_json_path, "w") as f:
            json.dump(all_inputs, f, indent=4)
        logger.info("Saved iface JSON: %s", iface_json_path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def update_configs(configs):
    """Merge experiment-specific configs into the global config."""
    exp_configs = generate_exp_configs(configs.exp)
    configs.update(ConfigDict(exp_configs))


def _apply_constraint_embedder_flags(model_name, configs):
    """Propagate embedder enable/disable flags into model_configs."""
    if model_name not in ("protenix_base_constraint_v0.5.0",):
        return
    embedder_cfg = model_configs[model_name]["model"]["constraint_embedder"]
    for key in (
        "pocket_embedder",
        "contact_embedder",
        "substructure_embedder",
        "contact_atom_embedder",
    ):
        embedder_cfg[key]["enable"] = getattr(configs, key)


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------
_TASK_DISPATCH = {
    "iface": run_inference,
    "save": run_save_sz,
    "input": construct_inputs,
    "1b1": run_inference_1b1,
    "default": run_inference_default,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    configs_base["use_deepspeed_evo_attention"] = os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true"
    configs = {
        **configs_base,
        **{"data": data_configs},
        **inference_configs,
        **dyna_configs,
    }
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )

    # Merge experiment-specific configs
    update_configs(configs)

    model_name = configs.model_name
    _, model_size, model_feature, model_version = model_name.split("_")
    logger.info(
        "Inference by Protenix: model_size=%s, features=%s, version=%s",
        model_size,
        model_feature.replace("-", ", "),
        model_version,
    )

    # Apply constraint embedder flags
    _apply_constraint_embedder_flags(model_name, configs)

    # Log key settings
    logger.info(
        "task=%s | input_json_path=%s | ckpt_name=%s | model_name=%s | load_checkpoint_path=%s",
        configs.task,
        configs.input_json_path,
        configs.ckpt_name,
        configs.model_name,
        configs.load_checkpoint_path,
    )
    if model_name in ("protenix_base_constraint_v0.5.0",):
        logger.info(
            "constraint_embedder: %s",
            model_configs[model_name]["model"]["constraint_embedder"],
        )

    # Apply model-specific configs & download caches
    configs.update(ConfigDict(model_configs[model_name]))
    download_infercence_cache(configs)

    # Dispatch task
    task_fn = _TASK_DISPATCH.get(configs.task)
    if task_fn is not None:
        # construct_inputs is a CPU-only task; under torchrun only rank 0
        # should execute it while other ranks exit immediately.
        if configs.task == "input":
            if is_main_process():
                task_fn(configs)
            else:
                logger.info(
                    "Rank %d: skipping construct_inputs (CPU-only task, rank 0 handles it).",
                    dist.get_rank(),
                )
            return
        task_fn(configs)
    else:
        logger.warning("Unknown task: %s. No action taken.", configs.task)


if __name__ == "__main__":
    run()
