# 示例用法
import numpy as np
from biotite.structure.io import pdbx
from biotite.structure import get_residues, filter_amino_acids, filter_nucleotides
from biotite.structure import stack
from biotite.structure.io.pdbx import CIFBlock


def extract_amino_acids_and_dna(input_cif, output_cif):
    # 1. 使用 mmCIF 讀取器逐步處理大文件
    cif_file = pdbx.CIFFile.read(str(input_cif))
    # ccd_codes = list(cif_file.keys())
    print(type(cif_file))
    print(dir(cif_file))
    # 2. 獲取結構數據（自動處理原子數組或堆疊）

    # 3. 定義標準氨基酸和 DNA 的殘基名稱集合
    target_residues = [
        # 20種標準氨基酸
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        # DNA 核苷酸
        "DA",
        "DC",
        "DG",
        "DT",
        # RNA
        "A",
        "C",
        "G",
        "U",
    ]
    blocks = dict([(t, cif_file[t]) for t in target_residues])
    # 5. 寫入新 CIF 文件
    output_file = pdbx.CIFFile(blocks=blocks)

    output_file.write(output_cif)

    print(f"提取完成！結果保存至: {output_cif}")


input_cif = "./release_data/components.v20240608.cif"  # 输入 CIF 文件路径
output_cif = "./tmp/output.cif"  # 输出 CIF 文件路径
# 使用示例
extract_amino_acids_and_dna(input_cif, output_cif)
