# convert_text_smiles_to_mol_tokens.py
import re
import logging
import torch
from typing import List
from rdkit import Chem

from atomdisc.utils.gnn_vq_utils import safe_parse_mol, fix_smiles, mol_to_graph

logger = logging.getLogger("convert_text_smiles_to_mol_tokens")

__all__ = ["add_smiles_special_tokens"]

# ---------- 生成 SMILES 结构符号正则 ---------- #

# 1. RDKit 所有元素符号
_ELEMENT_SYMBOLS_CAPITALIZED = [
    Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)
]
# 2. “芳香小写”元素
_AROMATIC_SYMBOLS_LOWERCASE = ['c', 'n', 'o', 's', 'p', 'se']
# 3. 明确要优先匹配的多字符元素（不拆分）
_EXPLICIT_MULTI = ['Cl', 'Br']

elements_for_regex: list[str] = _EXPLICIT_MULTI + [
    el for el in _ELEMENT_SYMBOLS_CAPITALIZED if len(el) == 1
] + _AROMATIC_SYMBOLS_LOWERCASE

SORTED_ELEMENT_SYMBOLS = sorted(
    set(elements_for_regex), key=lambda x: (-len(x), x)
)
ELEMENT_PATTERN_PART = "|".join(map(re.escape, SORTED_ELEMENT_SYMBOLS))

# 与 pre-training 相同的 tokenizer 正则
SMILES_TOKENIZER_REGEX_PATTERN = re.compile(
    r"("
    r"\[[^\]]+\]|"              # bracket 原子
    f"{ELEMENT_PATTERN_PART}|"   # 元素符号
    r"@@|@|"                    # 手性符号
    r"[=\#\(\)\.\[\]\+\-\:\/\\%]|[0-9]"  # 其它结构符号 + 数字
    r")"
)
STRUCT_SYMBOLS_TO_PRESERVE = {
    '=', '#', '(', ')', '.', '%', '+', '-', ':', '/', '\\',
    '[', ']', '@', '@@', *[str(i) for i in range(10)]
}
# ----------- 将 SMILES 字符串切分为 token 列表 ----------- #
def _tokenize_smiles_string(smi: str) -> List[str]:
    tokens, cur = [], 0
    while cur < len(smi):
        # 跳过空白
        while cur < len(smi) and smi[cur].isspace():
            cur += 1
        if cur >= len(smi):
            break

        m = SMILES_TOKENIZER_REGEX_PATTERN.match(smi, cur)
        if m:
            tokens.append(m.group(1))
            cur = m.end()
        else:
            # 理论上不会到这里；保险处理
            tokens.append(smi[cur])
            cur += 1
    return tokens

# ----------- 主函数，名称/签名保持不变 ----------- #
def convert_text_smiles_to_mol_tokens(text: str, gnn, vq, device):
    """
    将 text 中每段  <smiles>SMI</smiles>  替换成
        <mol> token… </mol>  形式，
    其中 token 同 pre-training 逻辑：
       - 元素 / [bracket] 原子 → 对应 <atom_k>
       - 结构符号 (= # ( ) … digits) 原样保留
    函数仍然 **只保留 text 第一个空行前的部分**（\n\n 截断规则不变）。
    """
    smiles_re = re.compile(r'<smiles>(.*?)</smiles>', flags=re.DOTALL | re.IGNORECASE)
    out_text  = text

    for m in smiles_re.finditer(text):
        raw_smi = m.group(1).strip()
        mol = (
            safe_parse_mol(raw_smi)
            or (lambda fs=fix_smiles(raw_smi): safe_parse_mol(fs) if fs != raw_smi else None)()
        )

        # ------ 默认失败回退为空串 ------
        mol_token_seq = ""

        if mol:
            graph = mol_to_graph(mol)
            if graph and graph[0].nelement() > 0:
                x, ei, ea = [t.to(device) for t in graph]
                with torch.no_grad():
                    node_repr = gnn(x, ei, ea)
                    _, code_ids, _ = vq(node_repr)

                # 1) 先用正则切 SMILES（用 canonical 形式，效果更稳定）
                try:
                    canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
                except Exception:
                    canon_smi = raw_smi  # canonical 失败就用原串

                smi_tokens = _tokenize_smiles_string(canon_smi)
                code_iter  = iter(code_ids.cpu().tolist())
                pieces     = ["<mol>"]

                for tk in smi_tokens:
                    if tk in STRUCT_SYMBOLS_TO_PRESERVE:
                        pieces.append(tk)
                    else:
                        try:
                            pieces.append(f"<atom_{next(code_iter)}>")
                        except StopIteration:
                            logger.error(
                                f"SMILES '{raw_smi}' atom-VQ 数量不匹配；直接终止替换。"
                            )
                            pieces = ["<mol>", "</mol>"]
                            break

                if pieces != ["<mol>", "</mol>"]:
                    pieces.append("</mol>")
                mol_token_seq = " ".join(pieces)

        # ---- 把整个 <smiles>…</smiles> 块替换成生成的 token 序列 ----
        out_text = out_text.replace(m.group(0), mol_token_seq)

    # --------- \n\n 之前截断 ---------
    out_text = out_text.split("\n\n")[0].strip()
    return out_text