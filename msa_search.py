import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def init_logging() -> None:
    log_format = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )


def run_msa(input_path: Union[str, Path], out_dir: Optional[Union[str, Path]] = None) -> Union[str, dict]:
    """Run MMseqs2 MSA search for a Protenix JSON or protein FASTA file."""
    input_file = Path(input_path).expanduser()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_dir = Path(out_dir).expanduser() if out_dir else input_file.parent / "msa"
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    init_logging()
    logger.info("Run MSA with input=%s, out_dir=%s", input_file, output_dir)

    suffix = input_file.suffix.lower()
    if suffix == ".json":
        from runner.msa_search import update_infer_json

        msa_input_json = update_infer_json(str(input_file), str(output_dir), use_msa=True)
        logger.info("MSA results have been written to %s", msa_input_json)
        return msa_input_json

    if suffix in {".fasta", ".fa", ".faa"}:
        from Bio import SeqIO
        from runner.msa_search import msa_search

        protein_seqs = sorted({str(record.seq) for record in SeqIO.parse(input_file, "fasta")})
        if not protein_seqs:
            raise ValueError(f"No protein sequences found in FASTA file: {input_file}")

        msa_res_subdirs = msa_search(protein_seqs, str(output_dir))
        if len(msa_res_subdirs) != len(protein_seqs):
            raise RuntimeError(
                "MSA search returned an unexpected number of result directories: "
                f"expected {len(protein_seqs)}, got {len(msa_res_subdirs)}"
            )
        fasta_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        logger.info("MSA result is %s and was saved to %s", fasta_msa_res, output_dir)
        return fasta_msa_res

    raise ValueError(f"Unsupported input format '{suffix}'. Expected .json, .fasta, .fa, or .faa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMseqs2 MSA search for a Protenix JSON or protein FASTA file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input .json, .fasta, .fa, or .faa file.",
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=None,
        help="MSA output directory (default: <input directory>/msa).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_msa(args.input, args.out_dir)


if __name__ == "__main__":
    main()
