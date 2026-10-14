#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp_base_configs.yaml"
REQUIRED_MSA_FILES = ("pairing.a3m", "non_pairing.a3m")
PIPELINE_TASKS = ("save", "input", "iface")
CHECKPOINT_BY_BINDING_TYPE = {
    "abag": "base_constraint",
    "protein-protein": "base_constraint",
    "protein-dna": "dyna_55199",
}


class PipelineProgress:
    def __init__(self, total: int) -> None:
        self.overall = tqdm(total=total, desc="Overall progress", unit="step", position=0)

    @contextmanager
    def stage(self, name: str):
        tqdm.write(f"[Start] {name}")
        with tqdm(total=1, desc=name, unit="step", position=1, leave=True) as progress:
            try:
                yield
            except Exception:
                progress.set_postfix_str("failed")
                tqdm.write(f"[Failed] {name}")
                raise
            else:
                progress.update(1)
                progress.set_postfix_str("completed")
                tqdm.write(f"[Completed] {name}")
                self.overall.update(1)

    def close(self) -> None:
        self.overall.close()


def _load_yaml(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    return data


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_msa_dir(msa_dir: str, input_file: Path) -> Path:
    path = Path(msa_dir).expanduser()
    if path.is_absolute():
        return path

    candidates = (PROJECT_ROOT / path, input_file.parent / path)
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_dir()),
        candidates[0].resolve(),
    )


def inspect_protein_msas(input_file: Path) -> Tuple[int, List[str]]:
    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON root must be a list: {input_file}")

    protein_count = 0
    issues = []
    for sample_index, sample in enumerate(data):
        if not isinstance(sample, dict) or not isinstance(sample.get("sequences"), list):
            raise ValueError(f"Input JSON sample {sample_index} is missing a sequences list")
        sample_name = sample.get("name", f"sample_{sample_index}")
        for sequence_index, sequence_entry in enumerate(sample["sequences"]):
            if not isinstance(sequence_entry, dict) or "proteinChain" not in sequence_entry:
                continue

            protein_count += 1
            protein_chain = sequence_entry["proteinChain"]
            label = f"{sample_name}.sequences[{sequence_index}].proteinChain"
            if not isinstance(protein_chain, dict) or not protein_chain.get("sequence"):
                issues.append(f"{label}: missing protein sequence")
                continue

            msa = protein_chain.get("msa")
            if not isinstance(msa, dict):
                issues.append(f"{label}: missing msa configuration")
                continue

            msa_dir_value = msa.get("precomputed_msa_dir")
            if not isinstance(msa_dir_value, str) or not msa_dir_value.strip():
                issues.append(f"{label}: missing precomputed_msa_dir")
                continue

            msa_dir = _resolve_msa_dir(msa_dir_value, input_file)
            if not msa_dir.is_dir():
                issues.append(f"{label}: MSA directory does not exist: {msa_dir}")
                continue

            missing_files = [
                filename
                for filename in REQUIRED_MSA_FILES
                if not (msa_dir / filename).is_file() or (msa_dir / filename).stat().st_size == 0
            ]
            if missing_files:
                issues.append(f"{label}: MSA directory {msa_dir} is missing valid files {missing_files}")

    return protein_count, issues


def _sample_names(input_file: Path) -> List[str]:
    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Input JSON must contain a non-empty sample list: {input_file}")

    names = []
    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"Input JSON sample {index} must be an object")
        name = sample.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"Input JSON sample {index} has an invalid name: {name!r}")
        names.append(name)
    return sorted(set(names))


def _experiment_seeds(experiment_config: dict) -> List[int]:
    seeds = experiment_config.get("seeds")
    if isinstance(seeds, list) and seeds and all(type(seed) is int for seed in seeds):
        return seeds

    seed_range = experiment_config.get("seed_range")
    if not isinstance(seed_range, dict):
        raise ValueError("Experiment must define either seeds or seed_range")
    start = seed_range.get("start")
    stop = seed_range.get("stop")
    step = seed_range.get("step", 1)
    if any(type(value) is not int for value in (start, stop, step)) or step == 0:
        raise ValueError("seed_range requires integer start/stop and a non-zero step")
    seeds = list(range(start, stop, step))
    if not seeds:
        raise ValueError("seed_range must produce at least one seed")
    return seeds


def _checkpoint_name(experiment_config: dict) -> str:
    binding_type = experiment_config.get("binding_type")
    try:
        return CHECKPOINT_BY_BINDING_TYPE[binding_type]
    except KeyError as error:
        raise ValueError(f"Unsupported binding_type: {binding_type!r}") from error


def expected_save_outputs(experiment_config: dict, input_file: Path) -> List[Path]:
    checkpoint = _checkpoint_name(experiment_config)
    output_dir = PROJECT_ROOT / "outputs" / checkpoint / "sz"
    return [
        output_dir / f"{sample_name}-{seed}.pt"
        for sample_name in _sample_names(input_file)
        for seed in _experiment_seeds(experiment_config)
    ]


def expected_iface_inputs(experiment: str, experiment_config: dict, input_file: Path) -> List[Path]:
    if Path(experiment).name != experiment:
        raise ValueError(f"Invalid experiment name: {experiment!r}")
    checkpoint = _checkpoint_name(experiment_config)
    return [
        input_file.parent / f"{experiment}_{checkpoint}_{seed}.json" for seed in _experiment_seeds(experiment_config)
    ]


def _missing_files(paths: Sequence[Path]) -> List[Path]:
    return [path for path in paths if not path.is_file() or path.stat().st_size == 0]


def invalid_iface_inputs(paths: Sequence[Path]) -> List[Path]:
    """Return missing, empty, malformed, or empty-list iface input files."""
    invalid = []
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            invalid.append(path)
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            invalid.append(path)
            continue
        if not isinstance(data, list) or not data:
            invalid.append(path)
    return invalid


def iface_inputs_are_ready(paths: Sequence[Path]) -> bool:
    return not invalid_iface_inputs(paths)


def run_pipeline_task(experiment: str, task: str) -> None:
    if task not in PIPELINE_TASKS:
        raise ValueError(f"Unsupported pipeline task: {task}")

    exp_script = PROJECT_ROOT / "exp.sh"
    if not exp_script.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {exp_script}")

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(PROJECT_ROOT)
    )
    subprocess.run(
        ["bash", str(exp_script), experiment, task],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MSAs when needed and run save/input/iface.")
    parser.add_argument(
        "--exp",
        default="8ekz",
        help="Experiment name in exp_base_configs.yaml (default: 8ekz).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "msa", *PIPELINE_TASKS),
        default="all",
        help=(
            "Stage to run: all, msa, save, input, or iface (default: all). "
            "The msa stage validates and updates the original input JSON in place."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--msa-out-dir",
        default=None,
        help="MSA output directory (default: <input JSON directory>/msa).",
    )
    return parser


def _run_msa_stage(
    progress: PipelineProgress,
    input_file: Path,
    msa_out_dir: Path,
) -> None:
    with progress.stage("Validate MSA"):
        protein_count, msa_issues = inspect_protein_msas(input_file)
        tqdm.write(f"Found {protein_count} protein chain(s)")
        for issue in msa_issues:
            tqdm.write(f"  - {issue}")
        if not msa_issues:
            tqdm.write("All protein chains have valid MSA paths")

    with progress.stage("Search MSA"):
        if msa_issues:
            from msa_search import run_msa

            updated_input = Path(run_msa(input_file, msa_out_dir)).resolve()
            if updated_input != input_file.resolve():
                raise RuntimeError("MSA search must update the original input JSON in place")
            _, remaining_issues = inspect_protein_msas(input_file)
            if remaining_issues:
                raise RuntimeError("Invalid MSA paths remain after the search:\n" + "\n".join(remaining_issues))
            tqdm.write(f"MSA paths written to the original JSON: {input_file}")
        else:
            tqdm.write("MSA paths are valid; skipping the search")


def _require_valid_msa(input_file: Path, stage: str) -> None:
    _, issues = inspect_protein_msas(input_file)
    if issues:
        details = "\n".join(f"  - {issue}" for issue in issues)
        raise RuntimeError(f"Cannot run '{stage}': valid MSA data is required. Run '--stage msa' first.\n{details}")
    tqdm.write(f"Prerequisite check passed for {stage}: MSA data is ready")


def _require_save_outputs(paths: Sequence[Path], stage: str) -> None:
    missing = _missing_files(paths)
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"Cannot run '{stage}': save-stage PT files are missing or empty. Run '--stage save' first.\n{details}"
        )
    tqdm.write(f"Prerequisite check passed for {stage}: save-stage PT files are ready")


def _require_iface_inputs(paths: Sequence[Path]) -> None:
    invalid = invalid_iface_inputs(paths)
    if invalid:
        details = "\n".join(f"  - {path}" for path in invalid)
        raise RuntimeError(
            "Cannot run 'iface': iface input JSON files are missing, empty, or invalid. "
            f"Run '--stage input' first.\n{details}"
        )
    tqdm.write("Prerequisite check passed for iface: iface input files are ready")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve_project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment YAML does not exist: {config_path}")

    config = _load_yaml(config_path)
    experiment_config = config.get(args.exp)
    if not isinstance(experiment_config, dict):
        raise KeyError(f"Experiment is not defined in the YAML file: {args.exp}")
    input_value = experiment_config.get("input_json_f")
    if not isinstance(input_value, str) or not input_value:
        raise ValueError(f"Experiment {args.exp} does not define a valid input_json_f")

    input_file = _resolve_project_path(input_value)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {input_file}")
    msa_out_dir = _resolve_project_path(args.msa_out_dir) if args.msa_out_dir else input_file.parent / "msa"

    save_outputs = expected_save_outputs(experiment_config, input_file)
    iface_inputs = expected_iface_inputs(args.exp, experiment_config, input_file)
    invalid_iface = invalid_iface_inputs(iface_inputs)
    iface_ready = not invalid_iface
    if iface_ready:
        tqdm.write("Existing iface input file(s) detected:")
        for path in iface_inputs:
            tqdm.write(f"  - {path}")
        tqdm.write("Skipping MSA search, save, and input because iface inputs are ready")
    else:
        tqdm.write("Iface inputs are incomplete; prerequisite stages may be required")
        for path in invalid_iface:
            tqdm.write(f"  - missing or invalid: {path}")

    run_msa = args.stage in ("all", "msa") and not iface_ready
    tasks = PIPELINE_TASKS if args.stage == "all" else ()
    if args.stage in PIPELINE_TASKS:
        tasks = (args.stage,)
    if iface_ready:
        tasks = tuple(task for task in tasks if task not in ("save", "input"))

    total_steps = (2 if run_msa else 0) + len(tasks)
    if total_steps == 0:
        tqdm.write(f"Experiment {args.exp}: stage '{args.stage}' skipped")
        return

    progress = PipelineProgress(total=total_steps)
    try:
        if run_msa:
            _run_msa_stage(progress, input_file, msa_out_dir)

        for task in tasks:
            with progress.stage(f"Run {task}"):
                if task == "save":
                    _require_valid_msa(input_file, task)
                elif task == "input":
                    _require_save_outputs(save_outputs, task)
                elif task == "iface":
                    _require_iface_inputs(iface_inputs)

                run_pipeline_task(args.exp, task)

                if task == "save":
                    missing_outputs = _missing_files(save_outputs)
                    if missing_outputs:
                        details = "\n".join(f"  - {path}" for path in missing_outputs)
                        raise RuntimeError(
                            "The save stage completed without producing all expected PT files:\n" + details
                        )
                elif task == "input":
                    invalid_outputs = invalid_iface_inputs(iface_inputs)
                    if invalid_outputs:
                        details = "\n".join(f"  - {path}" for path in invalid_outputs)
                        raise RuntimeError(
                            "The input stage completed without producing all valid iface inputs:\n" + details
                        )
                tqdm.write(f"{task} stage completed successfully")
    finally:
        progress.close()

    tqdm.write(f"Experiment {args.exp}: stage '{args.stage}' completed")


if __name__ == "__main__":
    main()
