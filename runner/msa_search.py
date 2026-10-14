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
import os
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from protenix.utils.logger import get_logger
from protenix.web_service.colab_request_parser import RequestParser

logger = get_logger(__name__)


def _resolve_msa_dir(msa_dir: Union[str, os.PathLike], base_dir: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve an MSA directory using the current directory before the JSON directory."""
    path = Path(msa_dir).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path]
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def protein_chain_has_valid_msa(protein_chain: Mapping, base_dir: Optional[Union[str, os.PathLike]] = None) -> bool:
    """Return whether a protein chain points to usable paired and unpaired MSA files."""
    msa = protein_chain.get("msa")
    if not isinstance(msa, Mapping):
        return False

    msa_dir = msa.get("precomputed_msa_dir")
    if not isinstance(msa_dir, str) or not msa_dir.strip():
        return False

    resolved_dir = _resolve_msa_dir(msa_dir, base_dir)
    required_files = (resolved_dir / "pairing.a3m", resolved_dir / "non_pairing.a3m")
    return resolved_dir.is_dir() and all(path.is_file() and path.stat().st_size > 0 for path in required_files)


def need_msa_search(json_data: dict, base_dir: Optional[Union[str, os.PathLike]] = None) -> bool:
    """Return whether any protein chain is missing a usable precomputed MSA."""
    return any(
        "proteinChain" in sequence and not protein_chain_has_valid_msa(sequence["proteinChain"], base_dir)
        for sequence in json_data["sequences"]
    )


def msa_search(seqs: Sequence[str], msa_res_dir: str) -> Sequence[str]:
    """
    do msa search with mmseqs and return result subdirs.
    """
    os.makedirs(msa_res_dir, exist_ok=True)
    tmp_fasta_fpath = os.path.join(msa_res_dir, f"tmp_{uuid.uuid4().hex}.fasta")
    RequestParser.msa_search(
        seqs_pending_msa=seqs,
        tmp_fasta_fpath=tmp_fasta_fpath,
        msa_res_dir=msa_res_dir,
    )
    msa_res_subdirs = RequestParser.msa_postprocess(
        seqs_pending_msa=seqs,
        msa_res_dir=msa_res_dir,
    )
    return msa_res_subdirs


def update_seq_msa(infer_seq: dict, msa_res_dir: str) -> dict:
    protein_seqs = []
    for sequence in infer_seq["sequences"]:
        if "proteinChain" in sequence.keys():
            protein_seqs.append(sequence["proteinChain"]["sequence"])
    if len(protein_seqs) > 0:
        protein_seqs = sorted(set(protein_seqs))
        msa_res_subdirs = msa_search(protein_seqs, msa_res_dir)
        if len(msa_res_subdirs) != len(protein_seqs):
            raise RuntimeError(
                "MSA search returned an unexpected number of result directories: "
                f"expected {len(protein_seqs)}, got {len(msa_res_subdirs)}"
            )
        protein_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        for sequence in infer_seq["sequences"]:
            if "proteinChain" in sequence.keys():
                sequence["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": protein_msa_res[sequence["proteinChain"]["sequence"]],
                    "pairing_db": "uniref100",
                }
    return infer_seq


def update_infer_json(json_file: str, out_dir: str, use_msa: bool = True) -> str:
    """
    update json file for inference.
    for every infer_data, if it needs to inference with msa and
    msa is not complete or missing in the json file,
    it will run msa searching if use_msa is True,
    else it will pass.
    """
    if not os.path.exists(json_file):
        raise RuntimeError(f"`{json_file}` not exists.")
    json_file = os.path.abspath(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    actual_updated = False
    json_base_dir = os.path.dirname(json_file)
    for seq_idx, infer_data in enumerate(json_data):
        if use_msa and need_msa_search(infer_data, json_base_dir):
            actual_updated = True
            seq_name = infer_data.get("name", f"seq_{seq_idx}")
            logger.info(f"starting to update msa result for seq {seq_idx} in {json_file}")
            update_seq_msa(
                infer_data,
                os.path.join(out_dir, seq_name, f"msa_resmsa_seq_{seq_idx}"),
            )
    if actual_updated:
        file_mode = stat.S_IMODE(os.stat(json_file).st_mode)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=os.path.dirname(json_file),
                prefix=f".{os.path.basename(json_file)}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                json.dump(json_data, temp_file, indent=4)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = temp_file.name
            os.chmod(temp_path, file_mode)
            os.replace(temp_path, json_file)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)
        logger.info(f"update msa result success and save to {json_file}")
        return json_file
    elif not use_msa:
        logger.warning(
            f"the inference json file {json_file} \n"
            + "do not contain msa and will not be updated,\n"
            + "and you set not using msa, in this mode, \n"
            + "if you do not use esm feature, model performance might degrade significantly"
        )
        return json_file
    else:
        logger.info(f"do not need to update msa result, so return itself {json_file}")
        return json_file
