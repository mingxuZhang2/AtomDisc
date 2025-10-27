# compute_metrics_smiles.py
"""分子字符串评估指标合集
------------------------------------------------
- **Exact Match**
- **BLEU**（字符级）
- **平均 Levenshtein 距离**
- **指纹 Tanimoto 相似度**：RDK、MACCS、Morgan(ECFP-4)
"""

import logging
from typing import List, Tuple

from nltk.translate.bleu_score import corpus_bleu
import Levenshtein  # pip install python-Levenshtein

# ------------------------------
# 尝试导入 RDKit
# ------------------------------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
except ImportError as e:
    raise ImportError(
        "RDKit is required for fingerprint-based metrics. Install via conda (conda install -c rdkit rdkit) "
        "or pip (pip install rdkit-pypi)."
    ) from e

# ================================================================
# 文本类指标
# ================================================================

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """严格字符串匹配比例。"""
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return correct / len(predictions) if predictions else 0.0


def compute_bleu_on_smiles(predictions: List[str], references: List[str]) -> float:
    """基于字符序列的 BLEU。"""
    refs = [[list(r.strip())] for r in references]
    hyps = [list(p.strip()) for p in predictions]
    try:
        return corpus_bleu(refs, hyps)
    except ZeroDivisionError:
        return 0.0


def compute_avg_levenshtein(predictions: List[str], references: List[str]) -> float:
    """平均 Levenshtein 编辑距离。"""
    total = sum(Levenshtein.distance(p.strip(), r.strip()) for p, r in zip(predictions, references))
    return total / len(predictions) if predictions else float("inf")

# ================================================================
# 指纹相似度指标
# ================================================================

_DEF_FAIL = (None, None, None)


def _mol(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _maccs_fp(mol):
    """兼容不同 RDKit 版本的 MACCS 指纹生成。"""
    if mol is None:
        return None
    if hasattr(MACCSkeys, "GenMACCSKeys"):
        return MACCSkeys.GenMACCSKeys(mol)  # 新版本 RDKit
    elif hasattr(MACCSkeys, "GetMACCSKeysFingerprint"):
        return MACCSkeys.GetMACCSKeysFingerprint(mol)  # 旧版本 RDKit
    else:
        raise AttributeError("当前 RDKit 版本不支持 MACCS 指纹，请升级 RDKit 或使用支持的版本。")


def _fps(mol):
    if mol is None:
        return None, None, None
    return (
        RDKFingerprint(mol),
        _maccs_fp(mol),  # 调用兼容函数
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    )


def compute_fingerprint_similarities(
    predictions: List[str], references: List[str]
) -> Tuple[float, float, float, List[Tuple[float, float, float]]]:
    """返回平均 (RDK, MACCS, Morgan) 相似度及逐样本分数列表。"""
    r_sum = m_sum = mo_sum = 0.0
    per_sample = []

    for p_smi, r_smi in zip(predictions, references):
        mol_p, mol_r = _mol(p_smi.strip()), _mol(r_smi.strip())
        fp_p_rdk, fp_p_maccs, fp_p_morgan = _fps(mol_p)
        fp_r_rdk, fp_r_maccs, fp_r_morgan = _fps(mol_r)

        rdk  = _tanimoto(fp_p_rdk, fp_r_rdk)
        macc = _tanimoto(fp_p_maccs, fp_r_maccs)
        morg = _tanimoto(fp_p_morgan, fp_r_morgan)

        per_sample.append((rdk, macc, morg))
        r_sum += rdk; m_sum += macc; morg_sum = morg
        mo_sum += morg

    n = len(predictions) if predictions else 1
    return r_sum / n, m_sum / n, mo_sum / n, per_sample
