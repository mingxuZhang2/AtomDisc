# convert_text_smiles_to_mol_tokens.py
import re
import logging
import torch
from typing import List
from rdkit import Chem

from atomdisc.utils.gnn_vq_utils import safe_parse_mol, fix_smiles, mol_to_graph

logger = logging.getLogger("convert_text_smiles_to_mol_tokens")

__all__ = ["add_smiles_special_tokens"]

# ---------- Build SMILES structural symbol regex ---------- #

# 1. All element symbols from RDKit
_ELEMENT_SYMBOLS_CAPITALIZED = [
    Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)
]
# 2. Aromatic lowercase elements
_AROMATIC_SYMBOLS_LOWERCASE = ['c', 'n', 'o', 's', 'p', 'se']
# 3. Multi-character elements to match first (do not split)
_EXPLICIT_MULTI = ['Cl', 'Br']

elements_for_regex: list[str] = _EXPLICIT_MULTI + [
    el for el in _ELEMENT_SYMBOLS_CAPITALIZED if len(el) == 1
] + _AROMATIC_SYMBOLS_LOWERCASE

SORTED_ELEMENT_SYMBOLS = sorted(
    set(elements_for_regex), key=lambda x: (-len(x), x)
)
ELEMENT_PATTERN_PART = "|".join(map(re.escape, SORTED_ELEMENT_SYMBOLS))

# Use the same tokenizer regex as pre-training
SMILES_TOKENIZER_REGEX_PATTERN = re.compile(
    r"("
    r"\[[^\]]+\]|"              # bracket atoms
    f"{ELEMENT_PATTERN_PART}|"   # element symbols
    r"@@|@|"                    # chirality symbols
    r"[=\#\(\)\.\[\]\+\-\:\/\\%]|[0-9]"  # other structural symbols + digits
    r")"
)
STRUCT_SYMBOLS_TO_PRESERVE = {
    '=', '#', '(', ')', '.', '%', '+', '-', ':', '/', '\\',
    '[', ']', '@', '@@', *[str(i) for i in range(10)]
}
# ----------- Split a SMILES string into a token list ----------- #
def _tokenize_smiles_string(smi: str) -> List[str]:
    tokens, cur = [], 0
    while cur < len(smi):
        # Skip whitespace
        while cur < len(smi) and smi[cur].isspace():
            cur += 1
        if cur >= len(smi):
            break

        m = SMILES_TOKENIZER_REGEX_PATTERN.match(smi, cur)
        if m:
            tokens.append(m.group(1))
            cur = m.end()
        else:
            # Should not happen; include the character as-is for robustness
            tokens.append(smi[cur])
            cur += 1
    return tokens

# ----------- Main function; keep the signature unchanged ----------- #
def convert_text_smiles_to_mol_tokens(text: str, gnn, vq, device):
    """
    Replace each <smiles>SMI</smiles> segment with
        <mol> token… </mol>, using the pre-training logic:
       - Elements / [bracket] atoms → corresponding <atom_k>
       - Structural symbols (= # ( ) … digits) remain as their original characters
    The function still truncates everything after the first blank line (\n\n).
    """
    smiles_re = re.compile(r'<smiles>(.*?)</smiles>', flags=re.DOTALL | re.IGNORECASE)
    out_text  = text

    for m in smiles_re.finditer(text):
        raw_smi = m.group(1).strip()
        mol = (
            safe_parse_mol(raw_smi)
            or (lambda fs=fix_smiles(raw_smi): safe_parse_mol(fs) if fs != raw_smi else None)()
        )

        # ------ Fallback to empty string when conversion fails ------
        mol_token_seq = ""

        if mol:
            graph = mol_to_graph(mol)
            if graph and graph[0].nelement() > 0:
                x, ei, ea = [t.to(device) for t in graph]
                with torch.no_grad():
                    node_repr = gnn(x, ei, ea)
                    _, code_ids, _ = vq(node_repr)

                # 1) Tokenize SMILES using the canonical form for stability
                try:
                    canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
                except Exception:
                    canon_smi = raw_smi  # If canonicalization fails, use the original string

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
                                f"SMILES '{raw_smi}' atom-VQ count mismatch; abort replacement."
                            )
                            pieces = ["<mol>", "</mol>"]
                            break

                if pieces != ["<mol>", "</mol>"]:
                    pieces.append("</mol>")
                mol_token_seq = " ".join(pieces)

        # ---- Replace the entire <smiles>…</smiles> block ----
        out_text = out_text.replace(m.group(0), mol_token_seq)

    # --------- Truncate before double newline ---------
    out_text = out_text.split("\n\n")[0].strip()
    return out_text